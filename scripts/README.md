# Scripts reproducibles

Estos comandos reconstruyen y verifican los datos procesados. El notebook
`notebooks/01_data_engineering.ipynb` los utiliza o reproduce el mismo flujo.

## Orden de ejecución

1. Auditar los datasets completos:

   ```bash
   python scripts/audit_datasets.py --paysim /ruta/PaySim.csv --ibm /ruta/HI-Small_Trans.csv --output-dir artifacts/audit
   ```

2. Construir secuencias y artefactos:

   ```bash
   python scripts/run_data_pipeline.py --ibm /ruta/HI-Small_Trans.csv --sender-summary artifacts/audit/ibm_sender_summary.csv.gz --audit artifacts/audit/dataset_audit.json
   ```

3. Validar archivos ya guardados:

   ```bash
   python scripts/validate_artifacts.py --artifacts artifacts
   ```

4. Precomputar los datos de la interfaz, después de entrenar ambas etapas:

   ```bash
   python scripts/build_mvp_data.py --artifacts artifacts
   ```

`audit_datasets.py` debe ejecutarse cuando cambie el dataset. El constructor debe
ejecutarse cuando cambien features, longitud, muestreo o semilla. El validador es
seguro y rápido; conviene ejecutarlo antes de entrenar o compartir artefactos.

`build_mvp_data.py` corre la inferencia de las dos etapas sobre TEST y guarda el
resultado en `artifacts/mvp_data.csv.gz`, para que la interfaz desplegada no
necesite PyTorch. Debe volver a ejecutarse cada vez que se reentrene un modelo o
cambie un umbral. El contrato de columnas está en `docs/mvp_contrato.md`.

Los CSV originales y los resúmenes grandes están ignorados por Git. Los NPZ,
metadata, preprocesador y contextos procesados constituyen el contrato compartido.
