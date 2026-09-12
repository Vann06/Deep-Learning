# Uso de IA — 4 consultas representativas

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
