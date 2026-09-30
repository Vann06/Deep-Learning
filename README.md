# CC3092 - Laboratorio 8

Implementación de la entrega parcial (Task 1 y Task 2): un Mini-GPT a nivel
de caracteres entrenado con Tiny Shakespeare y tres estrategias de muestreo.

## Estructura

- `Lab8_Entrega_Parcial.ipynb`: notebook que descarga datos, entrena, genera
  las 25 muestras y deja la evidencia para responder Task 2.2.
- `lab8_minigpt.py`: arquitectura, entrenamiento, greedy, top-k y top-p.
- `tests/test_lab8_minigpt.py`: pruebas rapidas de shapes, mascara y muestreo.

Los archivos del Proyecto 1 no se duplican en esta rama. Permanecen
recuperables en la rama `Proyecto1` y en el historial de Git.

## Ejecución

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Abra `Lab8_Entrega_Parcial.ipynb` y ejecute las celdas en orden. La descarga
se guarda en `data/` y los pesos, grafica y muestras en `outputs/`; ambas
carpetas son artefactos reproducibles y no se versionan.

## Verificación rápida

```powershell
python -m unittest discover -s tests -v
```

Antes de entregar, complete en el notebook el análisis cualitativo con
referencias a sus muestras ejecutadas y revise el registro del uso de IA.
