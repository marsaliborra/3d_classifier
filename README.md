# PointNet simplificado para clasificación de formas 3D (ModelNet10)

> **Estado:** en construcción — este README se irá completando a medida que
> avanza el proyecto (ver plan de 3 días más abajo). Las secciones marcadas
> como `TODO` se rellenarán con datos reales al final, no con placeholders
> inventados.

## Qué es esto

Proyecto de portfolio: un clasificador de nubes de puntos 3D entrenado sobre
[ModelNet10](https://modelnet.cs.princeton.edu/), usando una versión
**simplificada** de la arquitectura PointNet (Qi et al., 2017,
["PointNet: Deep Learning on Point Sets for 3D Classification and
Segmentation"](https://arxiv.org/abs/1612.00593)).

El objetivo no es igualar el estado del arte, sino demostrar que entiendo
**por qué** PointNet está diseñado como está diseñado (invarianza a
permutación de puntos vía max-pooling, MLPs compartidos por punto) y ser
explícito sobre qué he simplificado respecto al paper original y por qué.

## Por qué este proyecto (contexto)

Lo preparé como proyecto de portfolio para una entrevista como Application
Engineer en una empresa de deep learning 3D aplicado a datos CAD/CAE. Con
9 horas de trabajo repartidas en 3 días, prioricé:

- Entender y poder explicar cada componente del modelo, no maximizar accuracy.
- Ser honesto sobre las simplificaciones (ver más abajo) en vez de esconderlas.
- Dejar un camino claro de "qué haría con más tiempo".

## Simplificaciones conscientes respecto al paper original

| Aspecto | Paper original (PointNet 2017) | Este proyecto | Por qué |
|---|---|---|---|
| Alineación espacial | Usa una T-Net (mini red que predice una matriz de transformación 3x3 aplicada a los puntos de entrada, y otra 64x64 para features) | **Omitida** | La T-Net añade complejidad e inestabilidad de entrenamiento para ganar robustez a rotaciones arbitrarias. En ModelNet10 los modelos ya vienen mayormente alineados por eje, así que el beneficio es menor que el coste de explicar/depurar una red dentro de la red. Lo documento como omisión consciente, no como desconocimiento. |
| Dataset | ModelNet40 (40 clases) en el paper original | ModelNet10 (10 clases) | Subconjunto más pequeño y más limpio, pensado para entrenar en horas (no días) en una T4 gratuita de Colab. |
| Nº de épocas / búsqueda de hiperparámetros | Entrenamiento extenso con scheduler, batch norm cuidadosamente ajustado, etc. | Pocas épocas, configuración simple | Presupuesto de tiempo de 9h total, no una GPU dedicada. |
| Muestreo de puntos | El paper asume nubes de puntos ya muestreadas de forma consistente | Muestreo uniforme por área de superficie sobre la malla `.off` original (`trimesh.sample.sample_surface`) | Es una decisión de preprocesado propia, no del paper — ver `src/data.py`. |

## Estructura del repo

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py        # descarga, preprocesado y Dataset/DataLoader de PyTorch
│   ├── model.py        # arquitectura PointNet simplificada
│   ├── train.py         # loop de entrenamiento
│   └── evaluate.py    # accuracy, matriz de confusión, visualización de predicciones
├── notebooks/
│   └── 01_data_exploration.ipynb   # exploración del dataset (pensado para Colab)
├── data/               # dataset descargado (no versionado, ver .gitignore)
└── outputs/            # checkpoints y figuras generadas (no versionado)
```

## Cómo ejecutarlo (Google Colab)

1. Abre [colab.research.google.com](https://colab.research.google.com).
2. `File > Open notebook > GitHub`, pega la URL de este repo
   (`https://github.com/marsaliborra/3d_classifier`) y selecciona el
   notebook que quieras ejecutar (p.ej. `notebooks/01_data_exploration.ipynb`).
3. `Runtime > Change runtime type > T4 GPU` (necesario para el entrenamiento;
   la exploración de datos funciona igual en CPU, pero es más lenta).
4. Ejecuta las celdas en orden. La primera celda clona este repo dentro del
   propio Colab (`!git clone ...`) para poder importar `src/`.

## Resultados

TODO: accuracy en test, matriz de confusión y ejemplos de predicciones — se
rellenará tras el entrenamiento (Hora 6-7 del plan).

## Qué haría distinto con más tiempo

TODO.

## Next steps

TODO — incluirá una sección sobre llevar las predicciones a **NVIDIA
Omniverse** (exportando las nubes de puntos clasificadas a formato **USD**),
como extensión natural dado que la empresa a la que aplico trabaja en esa
dirección.

## Plan de trabajo (3 días, 9h)

- **Día 1 (3h):** setup del repo, carga/exploración de datos, preprocesado
  y Dataset de PyTorch.
- **Día 2 (3h):** arquitectura del modelo simplificado, loop de
  entrenamiento, primer entrenamiento en Colab.
- **Día 3 (3h):** evaluación (accuracy, matriz de confusión, visualización
  de predicciones), redacción final del README con resultados reales.
