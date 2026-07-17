# Deep-Learning
Deep Learning y Sistemas Inteligentes

Este repositorio contiene el código y los recursos utilizados en el curso de Deep Learning y Sistemas Inteligentes. Aquí encontrarás implementaciones de redes neuronales, ejemplos prácticos y ejercicios para reforzar los conceptos aprendidos en clase.

## Contenido
- `Lab_Semana2_CNN_v2_Estudiante.ipynb`: Notebook con ejercicios y ejemplos de redes neuronales convolucionales (CNN).

## Requisitos
- Python 3.x
- Bibliotecas: numpy, matplotlib, tensorflow, keras (según sea necesario)

## Notas de estudio 

###  Forward Pass
El objetivo del *Forward Pass* es transformar una imagen de entrada (una matriz de números) en una predicción final (un único número entre 0 y 1).

| Fase | Operación | ¿Qué hace en la práctica? | ¿Por qué es importante? |
| :--- | :--- | :--- | :--- |
| **1. Convolución** | Filtrado local | Desliza un pequeño molde (kernel) sobre la imagen buscando patrones específicos (bordes, texturas). | Extrae las características visuales clave. |
| **2. ReLU** | Activación | Apaga las neuronas muertas. Todo número negativo se convierte en `0`, los positivos se quedan igual. | Rompe la linealidad para que la red aprenda patrones complejos. |
| **3. Max Pooling** | Reducción | Se queda solo con el valor más alto de cada cuadrante de la imagen (por ejemplo, de 2x2). | Achica el mapa, ahorra memoria y tolera pequeñas variaciones. |
| **4. Flatten** | Aplanado | Estira la cuadrícula bidimensional de datos para convertirla en una sola fila larga. | Prepara los datos para la decisión final. |
| **5. Capa Densa** | Clasificación | Mezcla todos los datos del vector usando pesos ($W$) y sesgos ($b$) para dar un puntaje ($z$). | Conecta todas las pistas encontradas para tomar una decisión. |
| **6. Sigmoid** | Normalización | Aplastamiento matemático que toma el puntaje y lo comprime estrictamente entre `0` y `1`. | Transforma el resultado en una probabilidad entendible. |

---

###  Backward Pass (Regla de la Cadena)
El *Backward Pass* es el camino de regreso. Mide qué tan mala fue la predicción y distribuye la culpa hacia atrás para corregir el modelo.

1. **Calcular el Error Total (Loss):** Se compara la predicción final ($\hat{y}$) con la etiqueta real ($y$).
2. **Repartir la Culpa (Gradientes):** Viajamos al revés usando la regla de la cadena matemática. 
   * *El truco del Max Pooling:* Como en el camino de ida solo sobrevivió el número máximo de cada cuadrante, el error de regreso **solo se le asigna a ese píxel ganador**. El resto recibe un cero, porque no aportaron nada al resultado.
3. **Ajustar las Tuercas (Optimización):** Se modifican ligeramente los filtros de la convolución y los pesos de la capa densa mediante Descenso de Gradiente. La próxima vez que pase la misma imagen, el error será menor.


## Integrantes 

- Vianka Vanessa Castro Ordoñez 23201
- Ricardo Arturo Godínez Sánchez 23247

