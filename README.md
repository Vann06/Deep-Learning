# Laboratorio 10 - Deep Learning

Estructura inicial para la **entrega parcial del Task 1**: Grad-CAM implementado desde cero y verificación numérica de su relación con CAM sobre ResNet-18.

## Contenido

```text
.
|-- data/
|   `-- images/                 # Imágenes (a), (b), (c) y registro de fuentes
|-- notebooks/
|   `-- Lab10_Task1.ipynb      # Desarrollo y respuestas del Task 1
|-- results/
|   `-- task1/                  # Figuras y tablas generadas por el notebook
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Antes de ejecutar

1. Coloca tres imágenes en `data/images/` con los nombres `a_objeto_claro.jpg`, `b_dos_objetos.jpg` y `c_contexto_enganoso.jpg`.
2. Completa `data/images/README.md` con la fuente de cada imagen.
3. En la celda **Configuración de las imágenes** del notebook, define:
   - la clase del segundo objeto de la imagen (b);
   - el recuadro `(x_min, y_min, x_max, y_max)` de la imagen (c), medido después del preprocesamiento a 224 x 224.
4. Ejecuta el notebook de principio a fin y sustituye los espacios de análisis por conclusiones respaldadas por las tablas y figuras generadas.

## Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/Lab10_Task1.ipynb
```

También puede abrirse en Google Colab. El notebook detecta la raíz del repositorio cuando se ejecuta localmente; en Colab se debe ajustar `PROJECT_ROOT` a la carpeta donde se subió o clonó el proyecto.

## Lista de verificación del Task 1

- [ ] Task 1.1: cinco predicciones y probabilidades por imagen; fuente de cada imagen; recuadro de (c).
- [ ] Task 1.2: Grad-CAM desde `layer4`, usando el logit antes de softmax; mapas de las tres imágenes y segunda clase de (b).
- [ ] Task 1.3: CAM sin gradientes; verificación del logit; comparación CAM vs. Grad-CAM antes y después de ReLU/normalización.
- [ ] Task 1.4a: comparación de `alpha_k^c` con `w_k^c / 49` para cinco canales.
- [ ] Task 1.4b: IoU del 20% superior entre las dos clases de (b) y discusión de resolución espacial.
- [ ] Task 1.4c: masa dentro del recuadro, prueba causal ocultando el fondo y conclusión sobre atajos.
- [ ] Semilla aleatoria reportada.
- [ ] Código comentado y respuestas basadas en números/figuras producidos.
