"""
Evaluacion del modelo entrenado: accuracy en test, matriz de confusion,
curvas de aprendizaje y visualizacion de predicciones concretas.

Pensado para ejecutarse DESPUES de `python -m src.train` (necesita que
existan outputs/checkpoints/best_model.pt y outputs/checkpoints/history.json).
Igual que train.py, se ejecuta como modulo desde la raiz del repo:

    !python -m src.evaluate
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASSES, ModelNet10Dataset, download_modelnet10, get_dataset_arrays
from src.model import SimplifiedPointNet
from src.train import N_POINTS

CHECKPOINT_DIR = Path("outputs/checkpoints")
FIGURES_DIR = Path("outputs/figures")


def load_trained_model(checkpoint_path: Path, device: torch.device) -> SimplifiedPointNet:
    """Reconstruye la arquitectura y carga los pesos entrenados desde disco."""
    model = SimplifiedPointNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # desactiva dropout y usa las estadisticas acumuladas de BatchNorm
    return model


def predict_all(model: SimplifiedPointNet, points: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Corre el modelo sobre TODO el array de nubes de puntos y devuelve las
    clases predichas como array de numpy (una prediccion por muestra).

    Se hace por lotes (batch_size) en vez de en una sola pasada para no
    intentar meter el test set entero de golpe en memoria de GPU.
    """
    dataset = ModelNet10Dataset(points, np.zeros(len(points), dtype=np.int64))  # labels dummy, no se usan
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for batch_points, _ in loader:
            batch_points = batch_points.to(device)
            logits = model(batch_points)
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.append(predictions)

    return np.concatenate(all_predictions)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Matriz de confusion normalizada por fila: cada fila suma 1.0, asi que
    la celda [i, j] se lee como "de todos los ejemplos reales de la clase i,
    que fraccion se predijo como clase j". Normalizar por fila (en vez de
    mostrar conteos absolutos) es lo que permite comparar clases aunque
    tengan distinto numero de muestras en el test set - relevante aqui
    porque, como vimos en la exploracion de datos, ModelNet10 no esta
    perfectamente balanceado entre clases.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Prediccion del modelo")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusion (normalizada por fila) - test set")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Matriz de confusion guardada en: {save_path}")


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """
    Grafica loss y accuracy de train vs test a lo largo de las epocas.

    Por que interesa mirar esto (mas alla de la accuracy final): si
    train_acc sigue subiendo mientras test_acc se estanca o baja, es la
    senal clasica de overfitting - el modelo esta memorizando el train set
    en vez de generalizar. Con un dataset y un modelo tan pequenos como los
    de este proyecto, es un riesgo real a comentar en la entrevista, no
    solo una formalidad.
    """
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["test_loss"] for h in history], label="test")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss (CrossEntropy)")
    axes[0].set_title("Loss por epoca")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["test_acc"] for h in history], label="test")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy por epoca")
    axes[1].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Curvas de entrenamiento guardadas en: {save_path}")


def plot_example_predictions(
    points: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    n_examples: int = 4,
    seed: int = 0,
) -> None:
    """
    Visualiza n_examples nubes de puntos del test set junto a su etiqueta
    real y la prediccion del modelo, coloreando el titulo en verde si acerto
    y en rojo si fallo. Es una comprobacion cualitativa complementaria a la
    matriz de confusion: ver a ojo cuando el modelo se equivoca ayuda a
    entender si los errores son "razonables" (p.ej. confundir dos formas
    parecidas) o si hay algo raro en el pipeline.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=n_examples, replace=False)

    fig = plt.figure(figsize=(4 * n_examples, 4))
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, n_examples, i + 1, projection="3d")
        cloud = points[idx]
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2, alpha=0.6)

        true_label = CLASSES[y_true[idx]]
        pred_label = CLASSES[y_pred[idx]]
        is_correct = y_true[idx] == y_pred[idx]
        color = "green" if is_correct else "red"

        ax.set_title(f"real: {true_label}\npred: {pred_label}", color=color, fontsize=11)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Ejemplos de predicciones guardados en: {save_path}")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    dataset_dir = download_modelnet10(root="data")
    test_points, test_labels = get_dataset_arrays(dataset_dir, "test", n_points=N_POINTS, cache_dir="data")

    model = load_trained_model(CHECKPOINT_DIR / "best_model.pt", device)
    predictions = predict_all(model, test_points, device)

    accuracy = (predictions == test_labels).mean()
    print(f"\nAccuracy en test: {accuracy:.4f} ({(predictions == test_labels).sum()}/{len(test_labels)})\n")

    # classification_report da precision/recall/F1 POR CLASE, no solo la
    # accuracy global - util para detectar si el modelo falla sobre todo en
    # una o dos clases concretas (p.ej. confundir "desk" con "table", dos
    # formas geometricamente muy parecidas) en vez de fallar de forma
    # uniforme en todas.
    print(classification_report(test_labels, predictions, target_names=CLASSES))

    plot_confusion_matrix(test_labels, predictions, FIGURES_DIR / "confusion_matrix.png")

    history_path = CHECKPOINT_DIR / "history.json"
    if history_path.exists():
        import json

        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, FIGURES_DIR / "training_curves.png")
    else:
        print(f"Aviso: no se encontro {history_path}, se omiten las curvas de entrenamiento.")

    plot_example_predictions(test_points, test_labels, predictions, FIGURES_DIR / "example_predictions.png")

    return accuracy


if __name__ == "__main__":
    evaluate()
