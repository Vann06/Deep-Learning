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
