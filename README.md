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
