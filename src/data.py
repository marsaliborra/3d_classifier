"""
Descarga, preprocesado y carga de ModelNet10 como nubes de puntos.

CONTEXTO IMPORTANTE (por qué este módulo existe):
ModelNet10 NO es un dataset de nubes de puntos "de fábrica": son mallas 3D
(archivos .off, formato "Object File Format" — texto plano con vértices y
caras triangulares). PointNet, en cambio, opera sobre nubes de puntos sin
estructura de conectividad. El puente entre ambos es el muestreo de puntos
sobre la superficie de la malla, que hacemos aquí nosotros mismos (el paper
original de PointNet asume que ese muestreo ya está hecho).

Pipeline de este módulo:
    .off (malla) --sample_points_from_mesh--> nube de puntos cruda (N, 3)
                 --normalize_point_cloud-----> nube centrada y escalada
                 --ModelNet10Dataset----------> tensores listos para PyTorch
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from torch.utils.data import Dataset

# URL oficial de Princeton. ModelNet10 son las 10 clases "limpias" y
# alineadas por eje de las 40 de ModelNet40 - más manejable para un
# proyecto de unas pocas horas.
MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"

# Nombres de las 10 clases, en el orden en que aparecen las carpetas del zip.
# Los fijamos aquí (en vez de listarlos dinámicamente en cada run) para que
# el mapeo clase -> índice entero sea siempre el mismo entre entrenamiento,
# evaluación y cualquier notebook que cargue un checkpoint más tarde.
CLASSES = [
    "bathtub", "bed", "chair", "desk", "dresser",
    "monitor", "night_stand", "sofa", "table", "toilet",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def download_modelnet10(root: str = "data") -> Path:
    """
    Descarga y descomprime ModelNet10 en `root/ModelNet10` si no existe ya.

    Idempotente: si la carpeta ya está, no vuelve a descargar (~450MB,
    interesa no repetirlo cada vez que se reinicia el runtime de Colab
    dentro de la misma sesión).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "ModelNet10"

    if dataset_dir.exists():
        print(f"ModelNet10 ya existe en {dataset_dir}, no se descarga de nuevo.")
        return dataset_dir

    zip_path = root / "ModelNet10.zip"
    if not zip_path.exists():
        print("Descargando ModelNet10 (~450MB)... puede tardar unos minutos.")
        # Import local: requests no es una dependencia dura del resto del
        # proyecto, solo hace falta para esta descarga puntual.
        import requests

        response = requests.get(MODELNET10_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    print("Descomprimiendo...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    zip_path.unlink()  # no necesitamos guardar el .zip, solo los .off extraídos
    return dataset_dir


def list_off_files(dataset_dir: Path, split: str) -> list[tuple[Path, str]]:
    """
    Recorre `dataset_dir` y devuelve una lista de (ruta_al_off, nombre_clase)
    para el split pedido ("train" o "test").

    ModelNet10 organiza los archivos como:
        ModelNet10/<clase>/<train|test>/<clase>_XXXX.off
    así que la clase es simplemente el nombre de la carpeta de primer nivel.
    """
    dataset_dir = Path(dataset_dir)
    samples = []
    for class_name in CLASSES:
        split_dir = dataset_dir / class_name / split
        if not split_dir.exists():
            continue
        for off_path in sorted(split_dir.glob("*.off")):
            samples.append((off_path, class_name))
    return samples


def load_off_mesh(path: Path) -> trimesh.Trimesh:
    """
    Carga un archivo .off como malla de trimesh.

    Nota práctica: algunos .off de ModelNet10 tienen la cabecera mal
    formada (la palabra "OFF" pegada al primer número en vez de en su
    propia línea, un bug conocido y documentado del dataset original).
    trimesh ya maneja este caso, pero lo dejamos comentado aquí porque es
    el típico detalle que puede sorprender si se reimplementa el parser
    a mano.
    """
    return trimesh.load(str(path), file_type="off", force="mesh")


def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points: int, seed: int | None = None) -> np.ndarray:
    """
    Muestrea `n_points` puntos sobre la SUPERFICIE de la malla.

    Decisión de diseño propia (no del paper): usamos
    `trimesh.sample.sample_surface`, que muestrea puntos de forma uniforme
    respecto al ÁREA de cada triángulo (los triángulos grandes producen
    proporcionalmente más puntos que los pequeños). La alternativa más
    simple -y peor- sería tomar directamente los vértices de la malla:
    eso sesgaría la nube de puntos hacia zonas con mallado denso
    (p.ej. bordes redondeados) en vez de representar la forma de manera
    uniforme.

    El paper de PointNet no especifica cómo se generó su muestreo de
    ModelNet - asumen que la nube de puntos ya existe - así que esta
    función es responsabilidad nuestra.
    """
    rng = np.random.default_rng(seed)
    # trimesh.sample.sample_surface acepta un generador de números
    # aleatorios opcional a través del argumento `seed` en versiones
    # recientes; si no, usamos el estado global de numpy.
    points, _face_indices = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float32)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Centra la nube de puntos en el origen y la escala para que quepa en
    una esfera unitaria (radio 1).

    Por qué hace falta:
    - Sin centrar: la posición absoluta del objeto en el espacio de la
      malla original no aporta información sobre SU FORMA (que es lo que
      queremos clasificar) y solo añade varianza irrelevante que la red
      tendría que aprender a ignorar.
    - Sin escalar: los distintos objetos de ModelNet10 vienen en escalas
      de malla arbitrarias (un "bathtub" y un "night_stand" no están a la
      misma escala real-world consistente en el .off). Sin normalizar,
      la red podría aprender a discriminar clases por tamaño de la malla
      en vez de por forma, lo cual no generalizaría.

    Esto es una normalización estándar en la literatura de nubes de
    puntos (el propio paper de PointNet la menciona como preprocesado
    típico), no es una decisión exótica nuestra.
    """
    centroid = points.mean(axis=0)
    points = points - centroid

    # Radio de la esfera mínima que contiene todos los puntos = la mayor
    # distancia euclídea de cualquier punto al centroide.
    max_dist = np.linalg.norm(points, axis=1).max()
    points = points / max_dist
    return points.astype(np.float32)


def build_point_cloud_arrays(
    dataset_dir: Path,
    split: str,
    n_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorre todos los .off de un split, los convierte en nubes de puntos
    normalizadas y devuelve dos arrays de numpy:
        points: (num_samples, n_points, 3)
        labels: (num_samples,)  -- índices enteros según CLASSES

    Se hace una sola vez y se cachea en disco (ver `get_dataset_arrays`)
    porque muestrear una malla con trimesh en cada época de entrenamiento
    sería el cuello de botella del pipeline: es mucho más barato leer un
    .npz ya preprocesado que volver a muestrear ~4000 mallas por época.
    """
    samples = list_off_files(dataset_dir, split)
    if not samples:
        raise RuntimeError(
            f"No se encontraron archivos .off en {dataset_dir} para split='{split}'. "
            "¿Se ha descargado el dataset correctamente?"
        )

    all_points = np.zeros((len(samples), n_points, 3), dtype=np.float32)
    all_labels = np.zeros(len(samples), dtype=np.int64)

    for i, (off_path, class_name) in enumerate(samples):
        mesh = load_off_mesh(off_path)
        raw_points = sample_points_from_mesh(mesh, n_points, seed=seed + i)
        all_points[i] = normalize_point_cloud(raw_points)
        all_labels[i] = CLASS_TO_IDX[class_name]

    return all_points, all_labels


def get_dataset_arrays(
    dataset_dir: Path,
    split: str,
    n_points: int = 1024,
    cache_dir: str = "data",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Punto de entrada principal para obtener los arrays (points, labels) de
    un split. Usa una caché en disco (.npz) indexada por split y n_points,
    para no repetir el preprocesado (que puede tardar varios minutos) cada
    vez que se reinicia el notebook.
    """
    cache_path = Path(cache_dir) / f"modelnet10_{split}_{n_points}pts.npz"

    if cache_path.exists():
        print(f"Cargando nubes de puntos preprocesadas desde caché: {cache_path}")
        cached = np.load(cache_path)
        return cached["points"], cached["labels"]

    print(f"No hay caché para split='{split}', n_points={n_points}. Preprocesando desde .off...")
    points, labels = build_point_cloud_arrays(dataset_dir, split, n_points)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, points=points, labels=labels)
    print(f"Guardado en caché: {cache_path}")

    return points, labels


class ModelNet10Dataset(Dataset):
    """
    Dataset de PyTorch que envuelve los arrays de numpy ya preprocesados.

    Nota: la resta del centroide y la división por radio máximo ya se
    hicieron en `normalize_point_cloud` al construir la caché, así que
    aquí solo convertimos a tensor - no repetimos preprocesado pesado en
    cada __getitem__.
    """

    def __init__(self, points: np.ndarray, labels: np.ndarray):
        assert points.shape[0] == labels.shape[0]
        self.points = points  # (num_samples, n_points, 3)
        self.labels = labels  # (num_samples,)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        import torch

        # El modelo (ver src/model.py) espera canales-primero, es decir
        # (3, n_points) en vez de (n_points, 3), porque usamos nn.Conv1d
        # para implementar el "MLP compartido por punto": Conv1d espera
        # (canales, longitud_de_secuencia), y aquí tratamos cada punto
        # como una posición de la "secuencia" y sus 3 coordenadas (x,y,z)
        # como los canales de entrada.
        points = torch.from_numpy(self.points[idx].T).float()  # (3, n_points)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return points, label
