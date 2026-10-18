# Laboratorio 10 - Deep Learning

Explicabilidad de dos modelos: Grad-CAM y CAM sobre ResNet-18 (visión) y SHAP, LIME y contrafactuales sobre un gradient boosting entrenado con German Credit (tabular). La semilla es **42** en todos los notebooks.

## Contenido

```text
.
|-- data/
|   |-- credit-g.arff           # German Credit (equivalente local de fetch_openml("credit-g", version=1))
|   `-- images/                 # Imágenes (a), (b), (c) y registro de fuentes
|-- docs/
|   `-- ai_usage.md             # Prompts de IA usados y por qué funcionaron
|-- notebooks/
|   |-- Lab10_Task1.ipynb       # Grad-CAM desde cero y verificación con CAM
|   |-- Lab10_Task2.ipynb       # SHAP: TreeSHAP, KernelSHAP y Shapley exacto
|   |-- Lab10_Task3.ipynb       # LIME desde cero, estabilidad y contrafactuales
|   `-- Lab10_Task4.ipynb       # Deletion, insertion y aleatorización en cascada de Grad-CAM
|-- results/
|   |-- task1/                  # Figuras y CSV del Task 1
|   |-- task2/                  # Figuras, CSV y summary.json del Task 2
|   |-- task3/                  # Figuras y CSV del Task 3
|   `-- task4/                  # Figuras y CSV del Task 4
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Ejecución

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Los notebooks detectan la raíz del repositorio si se abren desde `notebooks/`. Para reproducir los números reportados, ejecute cada notebook completo (*Restart & Run All*). La ejecución reportada usó NumPy 2.5.1, pandas 2.3.3, scikit-learn 1.9.0 y shap 0.54.1. Con otras versiones, los valores SHAP cambian en la cuarta o quinta cifra decimal. Los tiempos de 2.4 y 2.6d dependen de la máquina.

## Estado

| Task | Entrega | Estado |
|---|---|---|
| 1 - Grad-CAM y CAM | Parcial | Completo |
| 2 - SHAP | Parcial | Completo |
| 3 - LIME y contrafactuales | Final | Completo |
| 4 - Validación de Grad-CAM | Final | Completo |

## Lista de verificación

### Task 1
- [x] 1.1: top-5 de cada imagen y recuadro de (c), `(0, 80, 223, 208)`.
- [x] 1.1: fuente (URL o autor) de cada imagen en `data/images/README.md` y en la respuesta 1.1.
- [x] 1.2: Grad-CAM desde `layer4` con hooks y logit antes del softmax; mapas de las tres imágenes y de la segunda clase de (b) (`tabby`).
- [x] 1.3: CAM sin gradientes, verificación del logit y comparación con Grad-CAM antes y después de ReLU y normalización.
- [x] 1.4a: `alpha_k^c` contra `w_k^c / 49` en los cinco canales de mayor peso.
- [x] 1.4b: IoU del 20% superior entre las dos clases de (b) y límite de la resolución 7 × 7.
- [x] 1.4c: masa dentro del recuadro, prueba causal con fondo gris y conclusión sobre atajos.

### Task 2
- [x] 2.1: codificación ordinal (limitación documentada), división 80/20 estratificada, exactitud y AUC.
- [x] 2.2: eficiencia, valor base, importancia SHAP frente a permutación, waterfalls y beeswarm.
- [x] 2.3: variable duplicada con ruido, intervencional frente a `tree_path_dependent`.
- [x] 2.4: KernelSHAP frente a TreeSHAP con fondos de 10, 50 y 100.
- [x] 2.5: Shapley exacto con 64 coaliciones escrito sin `shap`.
- [x] 2.6a-d: respuestas de interpretación.

### Task 3
- [x] 3.1: LIME desde cero (solución cerrada verificada contra `sklearn.Ridge`, intercepto sin penalizar) en tres clientes.
- [x] 3.2: Jaccard y $N_{ef}$ para 9 combinaciones de $N$ y $\sigma_{rel}$ con 20 semillas.
- [x] 3.3: Spearman LIME frente a SHAP, coincidencia del top 3 y $R^2_w$ según $K$.
- [x] 3.4: contrafactuales del cliente rechazado, validez, proximidad y dispersión.
- [x] 3.5a-d: respuestas de análisis con corridas de verificación.

### Task 4
- [x] 4.1: deletion e insertion con 20 pasos y 10 órdenes aleatorios en las tres imágenes.
- [x] 4.2: aleatorización en cascada de `fc` a `conv1` con Spearman frente al mapa original.
- [x] 4.3a-c: respuestas, con deletion sobre referencia desenfocada y la etapa `fc` repetida con 20 semillas.

### General
- [x] Semilla aleatoria reportada.
- [x] Prompts de IA de cada task documentados en `docs/ai_usage.md`.
