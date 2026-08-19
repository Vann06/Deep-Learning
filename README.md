# Hoja de Trabajo 2 - DCGAN de sprites de Pokemon

Proyecto de **CC3092 - Deep Learning y Sistemas Inteligentes**. El objetivo es
implementar una DCGAN desde cero con PyTorch, estudiar experimentalmente sus
fallas y conectar los resultados con la literatura de GANs.

## Estado actual

- [x] Dataset de 898 sprites RGB de 64x64 descargado y validado.
- [x] Task 1 entrenado durante 50 epocas.
- [x] Historial, 50 grids, grilla final, evolucion y curvas generadas.
- [x] Analisis de Task 1 documentado en `reports/HDT2.md`.
- [ ] Task 2: colapso de modo y estimacion JSD.
- [ ] Task 3: investigacion del paper.

## Estructura del proyecto

```text
Deep-Learning/
├── README.md
├── S7 - Hoja de Trabajo 2.pdf
├── scripts/
│   └── download_pokemon.py       # Descarga y normaliza los 898 sprites
├── notebooks/
│   ├── task1.ipynb               # DCGAN base, entrenamiento y visualizaciones
│   └── task2.ipynb               # Por crear: colapso de modo y JSD
├── data/
│   └── pokemon/                  # 001.png ... 898.png; no se versiona
├── outputs/
│   ├── task1/                    # Historial, grillas y checkpoints base
│   └── task2/                    # Experimento de colapso y curva JSD
└── reports/
    └── HDT2.md                   # Respuestas y evidencia de Tasks 1, 2 y 3
```



## 2. Descargar el dataset

El script obtiene los sprites 1 a 898 del repositorio publico de PokeAPI. Cada
imagen se compone sobre un fondo negro, se convierte a RGB y se redimensiona a
64x64 con vecino mas cercano para preservar el pixel art.

```powershell
.\.venv\Scripts\python.exe scripts\download_pokemon.py
```

Resultado esperado:

```text
Descargados: 898 | existentes: 0 | fallidos: 0
PNG disponibles: 898 (esperados para este rango: 898)
```

El comando es idempotente: al ejecutarlo nuevamente valida y omite los archivos
correctos. Algunas opciones utiles son:

```powershell
# Descargar un rango pequeno para probar la conexion
.\.venv\Scripts\python.exe scripts\download_pokemon.py --start 1 --end 20

# Reemplazar archivos existentes
.\.venv\Scripts\python.exe scripts\download_pokemon.py --overwrite

# Usar fondo blanco
.\.venv\Scripts\python.exe scripts\download_pokemon.py --background white
```

Los datos se guardan en `data/pokemon/` y estan excluidos de Git.

## 3. Fases del trabajo

### Fase 1 - Task 1: DCGAN base y entrega parcial

Ubicacion principal: `notebooks/task1.ipynb`.

1. Cargar y normalizar los 898 sprites al rango `[-1, 1]`.
2. Implementar Generator y Discriminator con las formas exigidas.
3. Verificar las salidas `(batch, 3, 64, 64)` y `(batch,)`.
4. Entrenar alternadamente D y G durante 50 epocas.
5. Guardar `loss_G`, `loss_D` y una grilla fija de 16 imagenes por epoca.
6. Producir la grilla final 4x4 y las curvas de perdida.
7. Documentar metodologia y resultados en `reports/HDT2.md`.

El notebook mantiene `RUN_TRAINING = False` para evitar iniciar accidentalmente
las 50 epocas. Despues de ejecutar y revisar las pruebas, cambiar a:

```python
RUN_TRAINING = True
```

Los resultados se escriben en:

```text
outputs/task1/history.csv
outputs/task1/grids/epoch_001.png ... epoch_050.png
outputs/task1/final_grid.png
outputs/task1/losses.png
outputs/task1/generator_final.pt
outputs/task1/discriminator_final.pt
```

### Fase 2 - Task 2: fallas experimentales y divergencia JSD

Ubicacion prevista: `notebooks/task2.ipynb`, reutilizando el codigo y los
checkpoints de Task 1.

1. Task 2.1: entrenar D cinco pasos por cada paso de G durante 20 epocas.
2. Registrar perdida, salida de D, gradientes y diversidad de las muestras.
3. Mostrar la grilla de baja diversidad y analizar el colapso de modo.
4. Task 2.2: estimar JSD en las 50 epocas del entrenamiento base.
5. Graficar la evolucion de JSD y contrastarla con el valor teorico esperado.
6. Escribir las respuestas matematicas en `reports/HDT2.md`.

Resultados previstos:

```text
outputs/task2/collapse_grid.png
outputs/task2/collapse_history.csv
outputs/task2/jsd_curve.png
```

### Fase 3 - Task 3: investigacion

Ubicacion principal: `reports/HDT2.md`.

1. Elegir una opcion: WGAN, Inception Score/FID o Unrolled GANs/MinibatchGAN.
2. Verificar que el paper pertenezca a NeurIPS, ICML, ICLR o CVPR.
3. Explicar la formulacion matematica del problema.
4. Presentar la modificacion al objetivo o al entrenamiento.
5. Conectar la propuesta con valores y observaciones reales de Task 2.
6. Mantener la seccion entre 400 y 600 palabras.

### Fase 4 - Integracion final

1. Ejecutar los notebooks desde un kernel limpio.
2. Confirmar que las figuras referenciadas existan.
3. Completar todos los espacios pendientes de `reports/HDT2.md`.
4. Agregar los prompts de IA utilizados y explicar como se verificaron.
5. Exportar el reporte a PDF.
6. Entregar el PDF y el archivo `.ipynb` o enlace al repositorio.

## Verificaciones rapidas

```powershell
# Revisar dependencias instaladas
.\.venv\Scripts\python.exe -m pip check

# Validar sintaxis del descargador
.\.venv\Scripts\python.exe -m py_compile scripts\download_pokemon.py

# Consultar archivos pendientes en Git
git status --short
```
