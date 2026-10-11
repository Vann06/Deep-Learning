# Laboratorio 9 – Deep Learning

Este repositorio contiene la implementación y análisis del **Laboratorio 9 de CC3045 – Deep Learning**, enfocado en modelos generativos aplicados al conjunto de datos **Fashion-MNIST**.

El laboratorio trabaja principalmente con dos familias de modelos:

- **Autoencoder Variacional (VAE)**, incluyendo el análisis del efecto de distintos valores de `beta` sobre la reconstrucción y el espacio latente.
- **Modelos de difusión**, desde la construcción del proceso forward hasta el entrenamiento de una U-Net condicional y el uso de classifier-free guidance.

También se incluyen verificaciones numéricas de conceptos estudiados en clase, como:

- Reparametrización en VAE.
- Divergencia KL entre distribuciones gaussianas.
- Proceso forward de difusión.
- Relación señal a ruido (SNR).
- Muestreo ancestral.
- Classifier-free guidance.
- Métricas de fidelidad, diversidad y nitidez.
- Estimación de `pass@k`.

## Dataset

Se utiliza **Fashion-MNIST**, compuesto por imágenes en escala de grises de `28 x 28` píxeles distribuidas en 10 clases de prendas y accesorios.

## Contenido general

El trabajo incluye:

1. Preparación y separación de los datos de entrenamiento y validación.
2. Implementación de un VAE con espacio latente de dimensión 2.
3. Entrenamiento y comparación de modelos con distintos valores de `beta`.
4. Visualización y análisis del espacio latente.
5. Generación e interpolación de imágenes.
6. Construcción del proceso forward de difusión.
7. Comparación de diferentes calendarios de ruido.
8. Implementación y entrenamiento de una U-Net condicional.
9. Generación de imágenes mediante difusión.
10. Aplicación de classifier-free guidance.
11. Evaluación de fidelidad, diversidad, nitidez y tiempo de generación.
12. Comparación entre VAE y difusión.
13. Cálculo y análisis de métricas `pass@1`, `pass@5` y `pass@10`.

## Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook

# Preparación del equipo 

El repositorio usa un entorno virtual y versiones fijas para que todos ejecuten el mismo código. Se recomienda Python 3.10 o 3.11.

## Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lab9 --display-name "Python 3 (Lab 9)"
jupyter lab
```

### Con GPU NVIDIA (opcional)

Antes de instalar `requirements.txt`, instale la misma versión de PyTorch con CUDA. `requirements.txt` acepta esa variante sin reinstalar.

```powershell
python -m pip install torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

## macOS o Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lab9 --display-name "Python 3 (Lab 9)"
jupyter lab
```

## Notas

Las implementaciones principales del VAE, el proceso forward de difusión, el muestreo y las fórmulas asociadas fueron desarrolladas directamente con tensores de PyTorch, siguiendo las restricciones indicadas en el laboratorio.

Los resultados, gráficas y análisis se obtienen a partir de la ejecución del notebook y se utilizan para justificar las conclusiones del laboratorio.

## Task 4 — Comparación y pass@k

- [Notebook ejecutado](notebooks/04_comparacion_passk.ipynb).
- [Informe PDF de Task 4](report/RESPUESTAS_TASK4.pdf).
- [Mediciones y resultados](artifacts/task4/resultados.json).

Se reutilizan las 500 muestras de difusión con `w=1` y los pesos de la U-Net de
`artifacts/task3/`. Como el repositorio no incluía pesos del VAE, se repitió únicamente
el entrenamiento de `beta=1` con la arquitectura y configuración de la Task 1.
Los pesos y el historial de esta ejecución se conservan en `artifacts/task4/`.
No se modificaron los notebooks de las Tasks 1–3.

La comparación usa 500 imágenes por grupo, todas en `[0,1]`. Los tiempos se miden
en CPU, con cuatro hilos, lotes de 100 y tres repeticiones; no se mezclan con los
tiempos de GPU de la Task 3. La semilla es `3045`.

Para reconstruir y ejecutar el notebook y después generar el PDF, desde la raíz:

```bash
python scripts/build_task4.py
python scripts/render_task4.py
python -m unittest discover -s tests -v
```

El primer comando descarga Fashion-MNIST si hace falta, carga los pesos guardados
del VAE (o lo entrena si faltan) y vuelve a cronometrar la difusión. Copia las
definiciones del VAE directamente del notebook 01 para no mantener otra implementación.
Las figuras temporales quedan en `outputs/task4/` y también se incluyen en el notebook
y el PDF. La ejecución local registrada usó PyTorch 2.13.0; la versión y el sistema
exactos se reportan en el notebook y en el JSON, sin atribuirlos al entorno fijado arriba.
