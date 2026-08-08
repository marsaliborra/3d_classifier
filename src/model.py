"""
Arquitectura PointNet SIMPLIFICADA para clasificacion de nubes de puntos.

Referencia: Qi et al., 2017, "PointNet: Deep Learning on Point Sets for 3D
Classification and Segmentation" (https://arxiv.org/abs/1612.00593).

IDEA CENTRAL DEL PAPER (la parte que SI implementamos):
Una nube de puntos es un CONJUNTO, no una secuencia ni una imagen: el orden
en que llegan los N puntos no tiene significado (si permuto los puntos, sigue
siendo el mismo objeto). Una arquitectura que clasifique nubes de puntos debe
ser, por tanto, invariante a permutaciones. PointNet consigue esto con dos
ingredientes:

  1. Un MLP "compartido" que se aplica de forma IDENTICA a cada punto por
     separado (mismos pesos para los N puntos) -> extrae una feature por
     punto sin mezclar informacion entre puntos todavia.
  2. Una funcion simetrica (aqui: max-pooling sobre la dimension de los N
     puntos) que agrega esas N features en un unico vector global. max()
     es simetrica porque max(a, b, c) = max(c, a, b): el resultado no
     depende del orden de los argumentos. Esa es la clave matematica de
     la invarianza a permutacion.

QUE SIMPLIFICAMOS respecto al paper (y por que; ver tambien README.md):
El paper original ANTES del MLP compartido inserta una "T-Net": una mini
red que predice una matriz de transformacion 3x3 y la aplica a los puntos
de entrada (y otra T-Net de 64x64 sobre las features intermedias), para
hacer la red tambien invariante a ROTACIONES/transformaciones afines de
entrada. Aqui la omitimos a proposito:
  - Anade una perdida de regularizacion extra (ortogonalidad de la matriz
    predicha) y una rama de red completa que hay que entrenar y depurar.
  - Los objetos de ModelNet10 ya vienen mayormente alineados por eje
    (no hay rotaciones arbitrarias que corregir en este dataset concreto).
  - Con un presupuesto de ~9h de proyecto, prioriza entender a fondo el
    mecanismo central (shared MLP + max-pooling simetrico) frente a anadir
    una pieza mas dificil de justificar en una entrevista si no se domina
    al 100%.
Si se quisiera anadir mas adelante, iria justo antes del primer Conv1d de
`PointNetFeatureExtractor` (ver comentario en el codigo mas abajo).
"""

import torch
import torch.nn as nn


class PointNetFeatureExtractor(nn.Module):
    """
    Extrae un vector de features GLOBAL (uno por nube de puntos) a partir
    de una nube de N puntos de entrada.

    Input:  (batch, 3, N)      -- coordenadas x,y,z de N puntos, canales-primero
    Output: (batch, 1024)      -- un unico vector que resume toda la nube

    Las dimensiones del MLP compartido (64, 64, 64, 128, 1024) son las
    mismas que usa el paper original -no las hemos cambiado- ya que no hay
    motivo particular para desviarnos de una configuracion que el paper ya
    valido empiricamente.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # --- MLP compartido por punto, implementado con Conv1d(kernel_size=1) ---
        #
        # Por que Conv1d y no nn.Linear:
        # Una Conv1d con kernel_size=1 aplicada a un tensor (batch, C_in, N)
        # produce (batch, C_out, N) aplicando LOS MISMOS pesos (C_in -> C_out)
        # a cada una de las N posiciones de forma independiente. Eso es
        # EXACTAMENTE "un MLP compartido aplicado a cada punto por separado":
        # no hay ninguna operacion (kernel_size=1, sin padding) que mezcle
        # informacion entre puntos distintos en esta fase. La mezcla entre
        # puntos ocurre unicamente, y de forma deliberada, en el max-pooling
        # de mas abajo.
        #
        # (Aqui es donde iria la T-Net del paper original si se quisiera
        # anadir mas adelante: como una capa extra que transforma `x` justo
        # antes de conv1.)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, N)
        x = self.relu(self.bn1(self.conv1(x)))   # (batch, 64, N)
        x = self.relu(self.bn2(self.conv2(x)))   # (batch, 64, N)
        x = self.relu(self.bn3(self.conv3(x)))   # (batch, 64, N)
        x = self.relu(self.bn4(self.conv4(x)))   # (batch, 128, N)
        x = self.bn5(self.conv5(x))              # (batch, 1024, N) -- sin ReLU aqui:
        # dejamos que el max-pooling opere sobre las features "en crudo" antes
        # de la ultima no-linealidad, tal como hace el paper original.

        # --- Max-pooling global: la funcion simetrica que da invarianza a permutacion ---
        # torch.max sobre la dimension N (la ultima) colapsa (batch, 1024, N)
        # a (batch, 1024): para cada uno de los 1024 canales, nos quedamos
        # con el valor maximo entre los N puntos. Intuicion: cada canal
        # aprende a "detectar" algun patron geometrico local (una esquina,
        # una superficie plana...) y el max-pooling se queda con la
        # respuesta mas fuerte de ESE patron en TODA la nube, sea cual sea
        # el punto donde aparezca. Da igual el orden de los puntos -> de
        # ahi la invarianza a permutacion.
        x, _ = torch.max(x, dim=2)  # (batch, 1024); descartamos los indices del max
        return x


class SimplifiedPointNet(nn.Module):
    """
    Clasificador completo: extractor de features global + cabeza fully
    connected que produce logits sobre las clases de ModelNet10.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = PointNetFeatureExtractor(in_channels=in_channels)

        # --- Cabeza de clasificacion ---
        # A partir de aqui ya NO hay ninguna nocion de "punto individual":
        # trabajamos sobre el vector global de 1024 valores que resume toda
        # la nube, exactamente igual que en un clasificador de imagenes tras
        # el global average/max pooling de una CNN convolucional 2D.
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # el paper usa dropout (keep_prob=0.7) en estas ultimas capas
            # para evitar que el clasificador memorice el vector global de
            # 1024 valores en vez de aprender a generalizar entre formas.
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, N) -> devuelve logits (batch, num_classes), SIN
        # softmax: usamos nn.CrossEntropyLoss en el entrenamiento, que ya
        # aplica log-softmax internamente de forma numericamente estable.
        global_features = self.feature_extractor(x)  # (batch, 1024)
        logits = self.classifier(global_features)      # (batch, num_classes)
        return logits


if __name__ == "__main__":
    # Sanity check rapido: construir el modelo y pasarle un batch aleatorio
    # para verificar que las formas de los tensores encajan en toda la red,
    # sin necesidad de tener el dataset descargado. Util para depurar la
    # arquitectura de forma aislada antes de conectarla al entrenamiento.
    batch_size, n_points, num_classes = 8, 1024, 10
    dummy_input = torch.randn(batch_size, 3, n_points)

    model = SimplifiedPointNet(num_classes=num_classes)
    output = model(dummy_input)

    print(f"Input:  {tuple(dummy_input.shape)}")
    print(f"Output: {tuple(output.shape)}  (esperado: ({batch_size}, {num_classes}))")
    assert output.shape == (batch_size, num_classes)
    print("OK: las formas de los tensores son correctas.")
