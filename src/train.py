"""
Loop de entrenamiento del clasificador SimplifiedPointNet sobre ModelNet10.

Pensado para ejecutarse en Colab con GPU T4, desde la RAIZ del repo, con:

    !python -m src.train

(no `!python src/train.py` directamente: al usar `from src.data import ...`
el script necesita ejecutarse como modulo del paquete `src`, para que Python
resuelva el import relativo a la raiz del repo. Ver
notebooks/01_data_exploration.ipynb para el mismo patron de setup/clonado).
Tambien funciona en CPU, solo que mucho mas lento.

SIMPLIFICACION CONSCIENTE (documentada tambien en README.md):
ModelNet10 no trae un split de VALIDACION separado, solo train/test. Lo
correcto en un proyecto de produccion seria reservar una parte del train
como validacion, y usar el test solo una vez al final. Aqui, por presupuesto
de tiempo (no hay una fase de busqueda de hiperparametros que justifique
mantener el test "ciego"), medimos accuracy sobre el test set AL FINAL DE
CADA EPOCA para poder graficar la curva de aprendizaje y guardar el mejor
checkpoint. Esto significa que el numero final de accuracy esta ligeramente
optimista respecto a lo que seria un test set nunca visto durante el
entrenamiento - un trade-off aceptable aqui, pero no valido en un proyecto
real con espacio para tunear hiperparametros.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import ModelNet10Dataset, download_modelnet10, get_dataset_arrays
from src.model import SimplifiedPointNet

# --- Hiperparametros ---
# Valores simples y razonables, sin busqueda de hiperparametros (Optuna,
# grid search, etc.) - decision consciente de alcance: con 9h de proyecto,
# el tiempo se invierte mejor en entender y documentar bien el pipeline que
# en optimizar decimas de accuracy. Si hiciera falta justificarlo en la
# entrevista: Adam con lr=1e-3 es el punto de partida estandar de facto para
# este tipo de arquitecturas, y estos valores siguen de cerca los del paper
# original de PointNet (que usa Adam, batch_size=32, lr inicial 1e-3).
N_POINTS = 1024
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
DATA_ROOT = "data"
CHECKPOINT_DIR = Path("outputs/checkpoints")


def build_dataloaders(n_points: int = N_POINTS, batch_size: int = BATCH_SIZE):
    """
    Descarga (si hace falta), preprocesa/cachea y envuelve ModelNet10 en
    DataLoaders de PyTorch listos para entrenar.
    """
    dataset_dir = download_modelnet10(root=DATA_ROOT)

    train_points, train_labels = get_dataset_arrays(dataset_dir, "train", n_points=n_points, cache_dir=DATA_ROOT)
    test_points, test_labels = get_dataset_arrays(dataset_dir, "test", n_points=n_points, cache_dir=DATA_ROOT)

    train_dataset = ModelNet10Dataset(train_points, train_labels)
    test_dataset = ModelNet10Dataset(test_points, test_labels)

    # shuffle=True SOLO en train: queremos que cada epoca vea los ejemplos en
    # un orden distinto (evita que la red aprenda patrones espurios ligados
    # al orden fijo del dataset). En test no importa el orden, asi que lo
    # dejamos fijo (mas rapido y reproducible al comparar epocas).
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    Ejecuta una pasada completa sobre `loader`.

    Si `optimizer` no es None, estamos en modo ENTRENAMIENTO: activamos
    dropout/batchnorm en modo train, calculamos gradientes y actualizamos
    pesos. Si es None, estamos en modo EVALUACION: desactivamos dropout,
    usamos las estadisticas acumuladas de BatchNorm (no las del batch
    actual) y no tocamos los pesos. Compartir esta funcion entre train y
    test evita duplicar la logica de "calcular loss y accuracy sobre un
    DataLoader" en dos sitios distintos.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # torch.no_grad() en evaluacion: no necesitamos gradientes (no vamos a
    # hacer backward), asi que desactivarlos ahorra memoria y computo.
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for points, labels in loader:
            points = points.to(device)   # (batch, 3, N)
            labels = labels.to(device)   # (batch,)

            if is_training:
                optimizer.zero_grad()  # limpiar gradientes acumulados del batch anterior

            logits = model(points)          # (batch, num_classes)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()   # calcula dL/dW para cada peso via backprop
                optimizer.step()  # actualiza los pesos: W -= lr * dL/dW (con el estado de Adam)

            # .item() saca el escalar de loss como float de Python (evita
            # acumular tensores de PyTorch en la CPU/GPU innecesariamente)
            total_loss += loss.item() * points.size(0)
            predictions = logits.argmax(dim=1)  # clase con mayor logit = prediccion del modelo
            total_correct += (predictions == labels).sum().item()
            total_samples += points.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    if device.type == "cpu":
        print("AVISO: no se detecto GPU. El entrenamiento sera mucho mas lento (usar Colab con GPU T4).")

    train_loader, test_loader = build_dataloaders()

    model = SimplifiedPointNet(num_classes=10).to(device)

    # CrossEntropyLoss combina internamente log-softmax + negative log
    # likelihood. Por eso el modelo devuelve logits "en crudo" (sin softmax
    # aplicado) en su forward: aplicar softmax nosotros mismos y luego
    # volver a aplicarlo dentro de la loss seria redundante y ademas menos
    # estable numericamente que la version combinada de PyTorch.
    criterion = nn.CrossEntropyLoss()

    # Adam en vez de SGD: mantiene un learning rate adaptativo por parametro
    # (usando momentos de primer y segundo orden del gradiente), lo que en
    # la practica converge mas rapido y con menos ajuste manual de
    # hiperparametros que SGD plano - relevante cuando no hay presupuesto
    # de tiempo para tunear un scheduler de learning rate a mano.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_test_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device, optimizer=None)

        elapsed = time.time() - start_time
        print(
            f"Epoca {epoch:3d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        # Guardamos el mejor checkpoint segun accuracy de test, no el ultimo
        # sin mas: en pocas epocas la curva puede oscilar, y queremos poder
        # evaluar/visualizar predicciones (Hora 6) con el mejor punto
        # observado durante el entrenamiento, no con el ultimo al azar.
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

    # El historial completo (no solo el mejor modelo) se guarda para poder
    # graficar loss/accuracy por epoca en la fase de evaluacion, sin tener
    # que volver a entrenar.
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), CHECKPOINT_DIR / "last_model.pt")

    print(f"\nMejor accuracy en test durante el entrenamiento: {best_test_accuracy:.4f}")
    print(f"Checkpoints guardados en: {CHECKPOINT_DIR}/")

    return model, history


if __name__ == "__main__":
    train()
