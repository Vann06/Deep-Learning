# Módulo de datos

Este paquete contiene la implementación reutilizable del pipeline temporal.

- `preprocessing.py`: features causales y transformación ajustada solo con TRAIN.
- `sequences.py`: selección de una ventana por remitente y post-padding.
- `splits.py`: separación estratificada por remitente y submuestreo normal.
- `dataset.py`: contrato NPZ, lectura, escritura y validaciones.
- `pipeline.py`: orquestación, metadata, contextos y figuras.

## Contrato NPZ

```python
from src.data.dataset import load_split

data = load_split("artifacts/train.npz")
X = data["X"]
y = data["y"]
lengths = data["lengths"]
mask = data["mask"]
```

Cada archivo contiene `X`, `y`, `sender_id`, `lengths`, `mask`,
`transaction_y` y `row_id`. El padding está al final, usa ceros en `X`, `False`
en `mask` y `-1` en los arrays de trazabilidad.

El modelo que aprende normalidad debe filtrar `y == 0`. El clasificador utiliza
ambas clases y debe respetar `mask`. Para análisis de atención, `row_id` enlaza
cada timestep con `val_context.csv.gz` o `test_context.csv.gz`.
