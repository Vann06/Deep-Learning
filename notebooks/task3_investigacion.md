# Reporte: WGAN y el problema del gradiente cero

**Opción seleccionada:** A
**Paper:** Wasserstein Generative Adversarial Networks
**Autores:** Martin Arjovsky, Soumith Chintala y Léon Bottou (2017, publicado en ICML)

## El problema: El gradiente o nabla cero

Principalmente, el problema que soluciona esta investigación es el del gradiente cero (o $\nabla = 0$). ¿Qué nos quiere decir esto? Básicamente, que en las GAN originales, cuando el discriminador puede reconocer muy fácilmente que un conjunto de datos del generador es evidentemente falso, el entrenamiento se estanca.

Al usar una función sigmoide en las GAN clásicas, devolvemos un valor entre $0$ y $1$. Esto quiere decir que si los datos son muy malos o muy evidentemente falsos, el modelo va a devolver un $0$. El problema real al que este equipo se enfrentó es que si el generador mejora un poco, pero no mejora lo suficiente, el modelo va a regresar nuevamente $0$.

Al no haber un cambio en la respuesta, la métrica matemática (la divergencia de Jensen-Shannon) se vuelve una constante y el gradiente o nabla pasa a ser prácticamente cero:

$$\nabla_{\theta} JSD(P_{real} \| P_{falso}) = 0$$

Al ser cero, el modelo no lograba aprender de manera correcta, por lo que no había una solución clara para esto.

## La solución: Distancias en lugar de probabilidades

Sin embargo, lo que ellos propusieron fue hacer algo diferente. En vez de tratar esto como una sigmoide que devuelve un valor entre $0$ y $1$, la idea fue devolver un número real continuo para poder decir cuál es la distancia exacta a la que están los datos falsos de los verdaderos. Esta nueva métrica se conoce como la Distancia de Wasserstein:

$$W(P_{real}, P_{falso}) = \sup_{\|f\|_{L} \leq 1} \mathbb{E}_{x \sim P_{real}}[f(x)] - \mathbb{E}_{x \sim P_{falso}}[f(x)]$$

Podemos tomarlo como un sistema de coordenadas. Es decir, podemos nosotros decir que el modelo mide que los datos falsos están a cierta distancia numérica de los verdaderos. Entonces, al medirlo con un número continuo, podemos ver realmente las mejoras paso a paso.

Gracias a esto, tenemos un gradiente que nunca va a llegar a ser nulo, ya que siempre vamos a tener las distancias para guiar al modelo. Esto es lo que se conoce como WGAN, y como se puede ver en las fórmulas, en la práctica mejoró y resolvió por completo el problema del gradiente cero.

## La conexión con lo que vimos en la Task 2

En la Task 2.1 provocamos este problema a propósito: entrenamos a D cinco veces por cada vez que entrenamos a G, para que se volviera "demasiado bueno" rápido. Y efectivamente eso fue lo que pasó: a partir de la época 7, `loss_D` se desplomó a valores de $10^{-4}$-$10^{-5}$, y `D(G(z))` promedio llegó a 0.0000113, prácticamente 0.

Ahí medimos directamente la norma del gradiente de G (`grad_norm_G`) en cada época, y se ve clarísimo el problema del gradiente cero descrito arriba: en las primeras 6 épocas ese gradiente promediaba 78.29, pero apenas D se volvió demasiado confiado (época 7 en adelante) el promedio cayó a 0.22 en las 14 épocas restantes — una caída de más de 350 veces. G se quedó sin señal útil para mejorar, y la diversidad entre las 16 imágenes finales cayó de 0.4098 (entrenamiento normal de Task 1) a 0.0449 (colapso de modo de Task 2.1).

Si en vez de la sigmoide de D hubiéramos usado el crítico de WGAN, ese gradiente no se habría aplanado igual: aunque el crítico también "ganara" casi todas las comparaciones entre reales y falsas, seguiría devolviendo la distancia real (Wasserstein) entre lo que genera G y los datos reales, en vez de simplemente un $0$. Con esa señal continua en vez de una probabilidad saturada, `grad_norm_G` probablemente se habría mantenido cerca de los valores altos que vimos al inicio (~78) en vez de desplomarse a 0.22, y el generador no se habría quedado atrapado produciendo siempre la misma salida repetida.

**Conteo de palabras (secciones anteriores):** ~540 palabras.

## Referencias

1. Arjovsky, M., Chintala, S. y Bottou, L. (2017). _Wasserstein GAN_. Proceedings of the 34th International Conference on Machine Learning (ICML). arXiv:1701.07875.
2. Evidencia experimental propia: `outputs/task2/collapse_history.csv`, `outputs/task2/collapse_summary.json` (Task 2, este repositorio).
