# Artefactos procesados

Esta carpeta contiene el contrato de datos consumido por los modelos y por la
interfaz.

## Archivos principales

- `train.npz`, `val.npz`, `test.npz`: secuencias transformadas.
- `metadata.json`: shapes, configuración, prevalencias y criterio de split.
- `feature_names.json`: orden exacto de las 49 features.
- `scaler.pkl`: transformación ajustada exclusivamente con TRAIN.
- `sender_splits.csv.gz`: asignación de cada remitente.
- `val_context.csv.gz`, `test_context.csv.gz`: contexto legible por timestep.
- `audit/dataset_audit.json`: evidencia usada para seleccionar el dataset.

## Etapa A: aprendizaje de la normalidad

Generados por [`02_stage_a_autoencoder.ipynb`](../notebooks/02_stage_a_autoencoder.ipynb):

- `stage_a_model.pt`: autoencoder completo (encoder + decoder) entrenado
  exclusivamente sobre remitentes normales de TRAIN.
- `encoder.pt`: solo el encoder, que es lo que reutiliza la Etapa B por transfer
  learning. Se carga con `src.models.encoder.load_encoder`.
- `anomaly_threshold.json`: umbral elegido (F1 máximo sobre VALIDATION), los
  tres candidatos evaluados, métricas en VAL/TEST, media/desviación del error
  normal de TRAIN y los controles de honestidad (correlación con longitud,
  comparación contra baselines triviales).

```python
from src.evaluation.anomaly import get_anomaly_score

resultado = get_anomaly_score(secuencia, longitud_real)
# {"score": ..., "z_score": ..., "threshold": ..., "is_anomalous": ...}
```

## Etapa B: clasificador supervisado

Generados por [`03_stage_b_classifier.ipynb`](../notebooks/03_stage_b_classifier.ipynb):

- `stage_b_model.pt`: clasificador completo (encoder transferido + atención +
  cabeza) con su `ClassifierConfig`. Se carga con
  `src.models.classifier.load_classifier`.
- `ablation_results.csv`: una fila por `(brazo, semilla)` de la ablación, con
  métricas de VAL y TEST. Es la evidencia de que el preentrenamiento aporta.
- `stage_b_threshold.json`: umbral elegido y sus candidatos, métricas VAL/TEST,
  resumen de la ablación, coeficientes de la fusión y el lift de atención.
- `fusion_model.pkl`: regresión logística de la fusión tardía sobre
  `[score_A, logit_B]`, ajustada en validación. **No se recomienda para el MVP**:
  en TEST no mejora al clasificador solo (PR-AUC 0.804 contra 0.807). Se conserva
  como resultado experimental.

```python
from src.models.classifier import predict_aml

probabilidad, pesos_atencion = predict_aml(secuencia, longitud_real)
# pesos_atencion viene recortado a la longitud real y suma 1
```

El MVP debe combinar `get_anomaly_score` (contexto de anomalía, expresado en
sigmas) con `predict_aml` (decisión y heatmap), sin pasar por `fusion_model.pkl`.

## MVP: inferencia precomputada

- `mvp_data.csv.gz`: una fila por cada uno de los 8,694 remitentes de TEST, con
  el score de la Etapa A en bruto y en sigmas, la probabilidad de la Etapa B,
  los pesos de atención ya recortados a la longitud real, y un párrafo de
  explicación en lenguaje natural.

Existe para que la interfaz desplegada no tenga que cargar PyTorch. Lo genera
`scripts/build_mvp_data.py` y el contrato de columnas está en
[`docs/mvp_contrato.md`](../docs/mvp_contrato.md).

```python
import json
import pandas as pd

datos = pd.read_csv("artifacts/mvp_data.csv.gz")
contexto = pd.read_csv("artifacts/test_context.csv.gz")

fila = datos[datos.sender_id == "235874:80D76EB80"].iloc[0]
transacciones = contexto[contexto.sender_id == fila.sender_id].sort_values("timestep")
pesos = json.loads(fila.atencion)   # len(pesos) == len(transacciones)
```

## Carga segura

```python
from src.data.dataset import load_split

train = load_split("artifacts/train.npz")
X_train = train["X"]
y_train = train["y"]
mask_train = train["mask"]
```

`load_split` valida automáticamente shapes, etiquetas, valores finitos, máscara,
padding y trazabilidad. La componente de normalidad utiliza las secuencias con
`y == 0`; el clasificador utiliza ambas clases. Ningún consumidor debe volver a
ajustar `scaler.pkl` ni asumir dimensiones distintas de `metadata.json`.

Los archivos de contexto permiten enlazar `row_id` con la transacción original
para visualizaciones, pesos de atención y explicaciones de alertas.
