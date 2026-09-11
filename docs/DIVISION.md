# Proyecto 2 — Deep Learning / AML Detection

## Objetivo del repositorio
Construir el proyecto en paralelo entre 3 personas sin bloquearse entre sí.

**Regla principal:** nadie cambia la estructura de datos compartida sin avisar al equipo.

---

# 1. División del trabajo

## Persona 1 — Data Engineering + Integración + MVP

### Debe entregar
1. Carga y exploración inicial del dataset.
2. Construcción de secuencias por remitente.
3. Features y normalización.
4. Split train/validation/test por remitente.
5. Manejo inicial del desbalance.
6. Visualizaciones requeridas antes del modelado.
7. Dataset procesado que usarán Persona 2 y Persona 3.
8. Estructura inicial de la app Streamlit.
9. Integración final de outputs de Etapa A y Etapa B.

### Archivos principales
- `notebooks/01_data_engineering.ipynb`
- `src/data/preprocessing.py`
- `src/data/sequences.py`
- `src/data/splits.py`
- `src/data/dataset.py`
- `app/app.py`
- `artifacts/metadata.json`

### Output obligatorio para los demás
Persona 1 debe dejar una interfaz común:

```python
X_train, y_train, sender_train
X_val, y_val, sender_val
X_test, y_test, sender_test
```

Forma recomendada:

```text
X: [num_sequences, sequence_length, num_features]
y: [num_sequences]
sender_id: [num_sequences]
```

También debe guardar:

```text
artifacts/
├── train.npz
├── val.npz
├── test.npz
├── scaler.pkl
├── metadata.json
└── feature_names.json
```

`metadata.json` debe indicar al menos:
- sequence_length
- num_features
- nombres de features
- cantidad de secuencias train/val/test
- proporción normal/sospechoso
- criterio de split
- padding utilizado

---

## Persona 2 — Etapa A: Aprendizaje de la normalidad

### Objetivo
Entrenar un modelo secuencial únicamente con comportamiento normal.

### Debe entregar
1. Arquitectura Encoder/Decoder.
2. Entrenamiento solo con secuencias normales del TRAIN.
3. Error de reconstrucción por secuencia.
4. Selección y justificación del threshold usando VALIDATION.
5. Métricas de detección de anomalías.
6. Guardar encoder y modelo de Etapa A.
7. Función que devuelva un `anomaly_score`.

### Archivos
- `notebooks/02_stage_a_autoencoder.ipynb`
- `src/models/autoencoder.py`
- `src/models/encoder.py`
- `src/evaluation/anomaly.py`

### Debe dejar
```python
def get_anomaly_score(sequence):
    ...
    return score
```

y guardar:

```text
artifacts/
├── stage_a_model.pt
├── encoder.pt
└── anomaly_threshold.json
```

No debe modificar archivos de Persona 1 salvo coordinación previa.

---

## Persona 3 — Etapa B: Clasificador + Transfer Learning + Ablation

### Objetivo
Reutilizar el encoder aprendido en Etapa A para construir el clasificador supervisado.

### Debe entregar
1. Cargar `encoder.pt`.
2. Implementar transfer learning.
3. Implementar attention.
4. Clasificador normal/sospechoso.
5. Elegir y justificar función de pérdida para clases desbalanceadas.
6. Crear baseline supervisado entrenado desde cero.
7. Comparar baseline vs modelo con pretraining.
8. Guardar modelo final.
9. Dejar función para devolver probabilidad y pesos de atención.

### Archivos
- `notebooks/03_stage_b_classifier.ipynb`
- `src/models/classifier.py`
- `src/models/attention.py`
- `src/evaluation/metrics.py`

### Debe dejar
```python
def predict_aml(sequence):
    ...
    return probability, attention_weights
```

y guardar:

```text
artifacts/
├── stage_b_model.pt
└── ablation_results.csv
```

---

# 2. Trabajo compartido

## Reporte
Cada persona redacta SU sección técnica.

### Persona 1
- Dataset
- Construcción de secuencias
- Features
- Normalización
- Splits
- Desbalance
- Limitaciones de datos

### Persona 2
- Etapa A
- Arquitectura
- Entrenamiento normal
- Reconstruction error
- Threshold
- Resultados de Etapa A

### Persona 3
- Transfer learning
- Etapa B
- Attention
- Función de pérdida
- Baseline
- Ablation
- Resultados comparativos

### Entre los 3
- Contexto AML
- Interpretabilidad de 5 casos
- Limitaciones
- Conclusiones
- Qué sería necesario para producción
- Revisión final


---

# 3. MVP

Persona 1 crea la app base.

La app espera estas dos funciones:

```python
from src.evaluation.anomaly import get_anomaly_score
from src.models.classifier import predict_aml
```

Flujo esperado:

```text
Seleccionar remitente
        ↓
Mostrar secuencia
        ↓
get_anomaly_score()
        ↓
predict_aml()
        ↓
Probability + Attention
        ↓
Heatmap
        ↓
Explicación en lenguaje natural
```

Persona 2 y Persona 3 NO necesitan programar Streamlit.
Solo deben respetar las funciones acordadas.

---

# 4. Orden de trabajo

## Fase 1 — Persona 1 primero
Antes de entrenar modelos deben existir:

```text
train.npz
val.npz
test.npz
metadata.json
```

Persona 2 y Persona 3 pueden preparar sus clases/modelos mientras tanto,
pero no deben asumir dimensiones definitivas hasta leer `metadata.json`.

## Fase 2 — Persona 2
Entrena Etapa A y entrega:

```text
encoder.pt
stage_a_model.pt
anomaly_threshold.json
```

## Fase 3 — Persona 3
Usa `encoder.pt` y entrena Etapa B.

## Fase 4 — Integración
Persona 1 conecta ambos modelos al MVP.

---

# 5. Reglas para evitar conflictos

1. No trabajar los tres en el mismo notebook.
2. Cada persona tiene su propio notebook.
3. Código reusable va en `src/`.
4. Modelos y resultados generados van en `artifacts/`.
5. No cambiar nombres de archivos compartidos sin avisar.
6. No cambiar el formato de `train/val/test` sin avisar.
7. Hacer commits pequeños y descriptivos.
8. Antes de mergear, hacer `git pull`.
9. Nunca subir datasets masivos al repositorio.
10. Guardar pesos grandes solo si el repositorio/plataforma lo permite.


---

# 8. Uso de IA

Registrar durante el proyecto:

`docs/ai_usage.md`

Formato:

```markdown
## Task
Ayuda para diseñar el pipeline de secuencias.

## Prompt utilizado
"..."

## Qué sugerencia usamos
...

## Qué decisión tomó el equipo
...

## Por qué
...
```

El equipo debe poder distinguir claramente entre sugerencias de IA y decisiones propias.

---

# 9. Definición de terminado

## Persona 1 termina cuando
- [ ] Dataset procesado
- [ ] Secuencias construidas
- [ ] Splits sin leakage
- [ ] Visualizaciones iniciales
- [ ] Archivos `.npz`
- [ ] `metadata.json`
- [ ] MVP base corre

## Persona 2 termina cuando
- [ ] Autoencoder entrena
- [ ] Entrena solo normalidad
- [ ] Reconstruction score calculado
- [ ] Threshold justificado
- [ ] `encoder.pt` guardado
- [ ] Función `get_anomaly_score()` funciona

## Persona 3 termina cuando
- [ ] Clasificador usa encoder preentrenado
- [ ] Attention funciona
- [ ] Baseline entrenado
- [ ] Ablation completo
- [ ] Métricas comparativas
- [ ] `predict_aml()` funciona

## Proyecto termina cuando
- [ ] Notebook reproducible
- [ ] Reporte terminado
- [ ] 5 casos interpretados
- [ ] MVP desplegado
- [ ] README actualizado
- [ ] Uso de IA documentado
