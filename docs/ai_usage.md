# Uso de IA: 6 consultas representativas

## Consulta 1
**Prompt:** “¿Cómo verifico si PaySim permite secuencias por remitente sin asumirlo?”

**Sugerencia usada:** contar historiales y positivos repetidos por `nameOrig`.

**Decisión del equipo:** descartarlo al medir p99=1 y máximo=3.

## Consulta 2
**Prompt:** “¿Cómo defino remitente, longitud y padding en IBM AML?”

**Sugerencia usada:** identidad banco+cuenta, percentiles y máscara.

**Decisión del equipo:** mínimo 3, longitud 32 y post-padding porque preserva 81.69% de positivos elegibles.

## Consulta 3
**Prompt:** “¿Cómo evito leakage y normalizo correctamente?”

**Sugerencia usada:** split por entidad y ajuste del preprocesador solo con TRAIN.

**Decisión del equipo:** 70/15/15 estratificado por remitente; sin IDs ni etiqueta entre features.

## Consulta 4
**Prompt:** “¿Cómo entrego datos trazables a las otras personas?”

**Sugerencia usada:** NPZ con máscara, longitudes y etiquetas por timestep.

**Decisión del equipo:** guardar además contexto val/test, metadata, vocabulario, scaler y un validador de round-trip.

## Consulta 5

**Prompt:** "Planifica e implementa la Etapa A (autoencoder de normalidad): arquitectura, contrato con la Etapa B, notebook celda por celda con justificación, y entrénalo hasta tener artefactos reales."

**Sugerencia usada:** encoder GRU que expone estados ocultos por timestep (no solo el vector latente) para que la atención de la Etapa B tenga entrada; MSE enmascarada dividida entre pasos reales; tres candidatos de umbral (F1 máximo, presupuesto de alertas, percentil de TRAIN) con justificación de negocio; controles explícitos anti-confound (correlación longitud↔error con/sin máscara, comparación contra baselines triviales).

**Decisión del equipo:** se implementó el código en `src/models/encoder.py`, `src/models/autoencoder.py`, `src/evaluation/anomaly.py` y `notebooks/02_stage_a_autoencoder.ipynb`, con 8 pruebas unitarias (`tests/test_stage_a.py`). El entrenamiento se ejecutó localmente tres veces (60, 150 y 90 épocas) para decidir el punto de corte: se detectó que extender el entrenamiento más allá de 60 épocas bajaba el error de reconstrucción pero empeoraba levemente la separación normal/sospechoso (ROC-AUC 0.756→0.751, PR-AUC 0.284→0.272), un efecto conocido en autoencoders de anomalías. Se documentó como hallazgo honesto en el notebook en vez de perseguir más épocas. Umbral final: F1 máximo (0.436), ROC-AUC en TEST 0.767, PR-AUC 0.308.

**Por qué:** el enunciado exige que el encoder sea reutilizable por la Etapa B (transfer learning) y que el umbral esté justificado con una métrica sobre validación, no elegido arbitrariamente; los controles anti-confound responden directamente a que las secuencias positivas son más cortas en promedio (13.48 vs 16.03 transacciones), un riesgo real de inflar el desempeño de forma espuria.

## Consulta 6

**Prompt:** "Planifica e implementa la Etapa B: clasificador supervisado que reutilice el encoder de la Etapa A, con atención, pérdida para clases desbalanceadas y la ablación obligatoria contra un baseline entrenado desde cero."

**Sugerencia usada:** atención aditiva con enmascarado *antes* del softmax (si se enmascara después, el padding recibe probabilidad y el mapa de calor señala transacciones inexistentes); cabeza alimentada solo por el contexto de la atención y no por el vector latente, para que los pesos expliquen la totalidad de la predicción; early stopping sobre PR-AUC de validación en vez de la pérdida; ablación con varias semillas por brazo para poder separar la mejora real del ruido de inicialización; y aprovechar `transaction_y` (etiqueta por timestep, ya presente en los `.npz`) para medir cuantitativamente si la atención se concentra en las transacciones realmente marcadas.

**Decisión del equipo:** se implementaron `src/models/attention.py`, `src/models/classifier.py` y `src/evaluation/metrics.py`, con 7 pruebas unitarias (`tests/test_stage_b.py`). Se adoptó la ablación de 2 brazos × 3 semillas y el fine-tune con learning rate diferenciado (encoder 1e-4, atención y cabeza 1e-3). Resultados: el encoder preentrenado obtiene 0.8107 ± 0.0044 de PR-AUC en TEST contra 0.7751 ± 0.0098 del brazo desde cero, sin traslape entre desviaciones. Se reportaron dos resultados negativos en lugar de omitirlos: la fusión tardía no mejora al clasificador solo (0.8035 contra 0.8070 en TEST), y el lift de atención sobre transacciones etiquetadas tiene mediana 0.772, por debajo de uniforme, aunque su media sea 5.18, un promedio inflado por la cola de la distribución.

**Por qué:** el enunciado exige explícitamente transfer learning, atención, una función de pérdida justificada para clases desbalanceadas y una ablación contra un baseline entrenado desde cero. Correr una sola semilla por brazo habría dejado la conclusión de la ablación indistinguible del ruido; y revisar la mediana además de la media evitó afirmar una interpretabilidad que los datos no sostienen.
