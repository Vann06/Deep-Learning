# Sistema secuencial para detección AML

Proyecto orientado a detectar comportamiento sospechoso en
historias de transacciones y explicar qué movimientos contribuyeron a cada
alerta. El resultado final combina aprendizaje de normalidad, clasificación
supervisada con transferencia y atención, y una interfaz interactiva para
consultar remitentes del conjunto de prueba.

## Resultado esperado del sistema

La solución recibe el historial cronológico de un remitente y produce:

- un score de anomalía respecto del comportamiento financiero normal;
- una probabilidad supervisada de actividad sospechosa;
- una contribución por transacción mediante pesos de atención;
- una explicación legible para un analista de cumplimiento;
- una comparación experimental contra un clasificador entrenado desde cero.

La interfaz permite seleccionar un remitente, visualizar su secuencia, consultar
ambos scores y revisar un mapa de calor sobre las transacciones que activaron la
alerta.

## Arquitectura propuesta y orden de archivos

```text
Deep-Learning/
├── notebooks/
│   ├── 01_data_engineering.ipynb   # EDA y construcción reproducible
│   ├── 02_stage_a_autoencoder.ipynb # autoencoder de normalidad (Etapa A)
│   └── 03_stage_b_classifier.ipynb  # clasificador con transferencia (Etapa B)
├── scripts/
│   ├── audit_datasets.py           # compara PaySim e IBM AML
│   ├── run_data_pipeline.py        # genera secuencias y metadata
│   ├── validate_artifacts.py       # verifica el contrato guardado
│   └── README.md                   # instrucciones de ejecución
├── src/
│   ├── data/                       # carga, features, splits y secuencias
│   ├── models/                     # encoder, autoencoder, atención y clasificador
│   └── evaluation/                 # anomalías y métricas
├── artifacts/                      # NPZ, preprocesador, metadata y contextos
├── reports/figures/                # visualizaciones del EDA
├── app/                            # interfaz interactiva
├── docs/                           # decisiones, división y registro de IA
└── requirements.txt
```

El flujo de información sigue este orden:

```text
CSV originales
    -> auditoría y selección del dataset
    -> limpieza, features y orden temporal
    -> split por remitente y normalización con TRAIN
    -> secuencias NPZ + metadata + contextos
    -> aprendizaje de normalidad y encoder
    -> clasificador con transferencia y atención
    -> métricas, análisis de casos e interfaz
```

La separación mantiene el EDA en el notebook, la lógica reutilizable en `src/`,
los comandos reproducibles en `scripts/` y los archivos consumibles en
`artifacts/`. Los modelos leen dimensiones y nombres desde `metadata.json`; no
deben duplicar reglas de preprocesamiento.

## Datos seleccionados

Se analizaron los dos datasets propuestos en las instrucciones:

- **PaySim:** 6,362,620 transacciones y 6,353,307 remitentes. El percentil 99 es
  una transacción por remitente y el máximo es 3, por lo que no permite construir
  historias suficientemente informativas.
- **IBM AML HI-Small:** 5,078,345 transacciones, 496,999 remitentes compuestos y
  3,376 remitentes con al menos una transacción etiquetada. Este dataset sí
  permite modelar comportamiento temporal.

La identidad de remitente se define como `From Bank:Account`, evitando colisiones
de cuentas entre bancos. Cada ejemplo contiene una ventana cronológica de hasta
32 transacciones; se exige un mínimo de 3 y se utiliza post-padding con máscara.

## Representación

Cada transacción se representa con 49 features:

- montos pagado y recibido, y su diferencia absoluta;
- tiempo desde la transacción anterior;
- hora y día de semana con codificación cíclica;
- coincidencia de banco y moneda;
- cambio y novedad del destino;
- formato de pago y monedas codificados con one-hot.

Los montos y gaps utilizan `log1p` seguido de `StandardScaler`. El preprocesador
se ajusta únicamente con timesteps reales de TRAIN. Identificadores y etiquetas
no forman parte de las features.

## Dataset procesado

| Split | Shape de `X` | Positivos | Negativos |
|---|---:|---:|---:|
| Train | `(40551, 32, 49)` | 1,931 | 38,620 |
| Validation | `(8673, 32, 49)` | 413 | 8,260 |
| Test | `(8694, 32, 49)` | 414 | 8,280 |

Los splits se construyen 70/15/15 por remitente y no comparten identidades. Se
conservan todos los remitentes positivos elegibles y se seleccionan 20 normales
por positivo dentro de cada split. La prevalencia procesada de 4.762% facilita
el entrenamiento, pero no representa la prevalencia operativa original.

Cada NPZ contiene:

```text
X, y, sender_id, lengths, mask, transaction_y, row_id
```

`transaction_y` y `row_id` mantienen trazabilidad por timestep para explicar
predicciones y relacionarlas con los archivos de contexto de validación y prueba.

## Estructura

```text
notebooks/                 análisis ejecutado y experimentos
src/data/                  pipeline temporal reutilizable
src/models/                modelos secuenciales, atención y clasificación
src/evaluation/            métricas y evaluación de anomalías
scripts/                   comandos reproducibles y validación
artifacts/                 NPZ, metadata, preprocesador y contextos
reports/figures/           visualizaciones del EDA
docs/                      decisiones, división y registro de uso de IA
app/                       interfaz interactiva
```

## Etapa A: aprendizaje de la normalidad

[`notebooks/02_stage_a_autoencoder.ipynb`](notebooks/02_stage_a_autoencoder.ipynb)
entrena un autoencoder GRU exclusivamente sobre remitentes normales de TRAIN.
El error de reconstrucción enmascarado es el score de anomalía; el umbral se
justifica con F1 máximo sobre VALIDATION (44 alertas por cada 1,000 remitentes,
45.5% de precisión) y se evalúa una sola vez en TEST (ROC-AUC 0.767,
PR-AUC 0.308). El notebook demuestra empíricamente, antes de entrenar, por qué
la máscara es indispensable, y después, que el score supera ampliamente a
baselines triviales de longitud y monto, que son nuestros controles
anti-confound explícitos.
Solo consume `artifacts/{train,val,test}.npz`; no vuelve a tocar el CSV
original. Produce `stage_a_model.pt`, `encoder.pt` (el contrato que reutiliza
la Etapa B) y `anomaly_threshold.json`.

## Etapa B: clasificador supervisado con transferencia y atención

[`notebooks/03_stage_b_classifier.ipynb`](notebooks/03_stage_b_classifier.ipynb)
reutiliza `encoder.pt` y le añade atención aditiva enmascarada y una cabeza
binaria, entrenando ya con ambas clases y `pos_weight=20`. El *early stopping*
usa PR-AUC de validación, no la pérdida.

| Señal | ROC-AUC (TEST) | PR-AUC (TEST) |
|---|---:|---:|
| Etapa A (autoencoder) | 0.767 | 0.308 |
| **Etapa B (clasificador)** | **0.972** | **0.807** |
| Fusión tardía | 0.971 | 0.804 |

Con el umbral congelado desde validación se alertan 31 remitentes de cada 1,000,
con 94.0% de precisión y 60.6% de recall.

**Ablación (6 corridas: 2 brazos × 3 semillas).** El encoder preentrenado obtiene
0.8107 ± 0.0044 de PR-AUC en TEST frente a 0.7751 ± 0.0098 del encoder
inicializado al azar: **+0.0356 sin traslape entre desviaciones**, así que la
transferencia desde la Etapa A está justificada empíricamente y no es ruido de
inicialización.

Dos resultados negativos que se reportan tal cual: la **fusión tardía no mejora**
al clasificador solo (0.804 contra 0.807), por lo que el MVP debe usar la Etapa B
directamente; y la atención solo se alinea parcialmente con las transacciones
etiquetadas (lift mediano 0.772, 45.4% por encima de uniforme), de modo que el
heatmap muestra dónde miró el modelo pero no prueba cuál transacción fue la
culpable.

Produce `stage_b_model.pt`, `ablation_results.csv`, `stage_b_threshold.json` y
`fusion_model.pkl`.

## Reproducción

La ruta recomendada es ejecutar
[`notebooks/01_data_engineering.ipynb`](notebooks/01_data_engineering.ipynb)
desde la raíz del repositorio. El notebook descarga los datos públicos con
`kagglehub`, audita ambos datasets, reconstruye los artefactos, genera las figuras
y ejecuta las validaciones. Después pueden ejecutarse
[`notebooks/02_stage_a_autoencoder.ipynb`](notebooks/02_stage_a_autoencoder.ipynb)
y [`notebooks/03_stage_b_classifier.ipynb`](notebooks/03_stage_b_classifier.ipynb),
que no requieren descargar nada adicional: el 02 consume los `.npz` versionados y
el 03 consume además `encoder.pt`. Cada notebook es independiente.

Los tres se ejecutan en Colab desde la rama `Proyecto2`; la primera celda clona el
repositorio e instala `requirements.txt`.

[![Abrir 01 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vann06/Deep-Learning/blob/Proyecto2/notebooks/01_data_engineering.ipynb)
[![Abrir 02 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vann06/Deep-Learning/blob/Proyecto2/notebooks/02_stage_a_autoencoder.ipynb)
[![Abrir 03 en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vann06/Deep-Learning/blob/Proyecto2/notebooks/03_stage_b_classifier.ipynb)

También puede utilizarse la línea de comandos:

```bash
pip install -r requirements.txt
python scripts/audit_datasets.py --paysim /ruta/PaySim.csv --ibm /ruta/HI-Small_Trans.csv
python scripts/run_data_pipeline.py --ibm /ruta/HI-Small_Trans.csv --sender-summary artifacts/audit/ibm_sender_summary.csv.gz --audit artifacts/audit/dataset_audit.json
python scripts/validate_artifacts.py --artifacts artifacts
```

La función de carga compartida valida automáticamente shapes, labels, máscara,
padding y valores no finitos:

```python
from src.data.dataset import load_split

train = load_split("artifacts/train.npz")
X_train, y_train = train["X"], train["y"]
```

Las decisiones y limitaciones se encuentran en
[`docs/decisions.md`](docs/decisions.md), y el uso de cada comando se documenta en
[`scripts/README.md`](scripts/README.md).
