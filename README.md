# CC3092 - Laboratorio 8

Entrega parcial (Task 1 y Task 2): un Mini-GPT a nivel de caracteres
entrenado con Tiny Shakespeare y tres estrategias de muestreo.

Entrega final (Task 3 y Task 4): embeddings contextuales de BERT frente a
embeddings estáticos de Word2Vec, y preguntas de análisis sobre GPT y BERT.

## Estructura

- `notebooks/Lab8_Entrega_Parcial.ipynb`: notebook que descarga datos, entrena,
  genera las 25 muestras y deja la evidencia para responder Task 2.2.
- `notebooks/Lab8_Entrega_Final.ipynb`: extracción de embeddings con
  `bert-base-multilingual-cased`, similitud coseno, comparación con
  `word2vec-google-news-300`, PCA y respuestas del Task 4.
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

Abra los notebooks de `notebooks/` y ejecute las celdas en orden. La primera
celda de código cambia el directorio de trabajo a la raíz del repositorio, así
que las rutas `data/` y `outputs/` y el import de `lab8_minigpt` funcionan
igual desde Jupyter o VS Code. La descarga se guarda en `data/` y los pesos,
gráficas y muestras en `outputs/`; ambas carpetas son artefactos reproducibles
y no se versionan.

La entrega final descarga la primera vez `bert-base-multilingual-cased`
(~700 MB, caché de Hugging Face) y `word2vec-google-news-300` (~1.6 GB, caché
de gensim en `~/gensim-data`). Ninguno de los dos se guarda en el repositorio.

## Verificación rápida

```powershell
python -m unittest discover -s tests -v
```

Antes de entregar, complete en el notebook el análisis cualitativo con
referencias a sus muestras ejecutadas y revise el registro del uso de IA.
