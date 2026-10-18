# Uso de IA en el Laboratorio 10

Usamos la IA principalmente para entender los conceptos y las fórmulas de cada task y relacionarlos con nuestros propios resultados. El código lo ejecutamos y revisamos nosotros, y cada número citado en las respuestas sale de nuestras celdas.

## Task 1: Grad-CAM y CAM

**Prompt:** "Ayúdame a entender la fórmula de Grad-CAM, $L^c=\text{ReLU}(\sum_k\alpha_k^cA^k)$, y por qué en nuestro ResNet-18 $\alpha_k^c$ termina siendo igual a $w_k^c/Z$. ¿Cómo se relaciona con los errores que obtuvimos en el Task 1.3?"

**Por qué funcionó:** pedimos la explicación conectada a nuestros números y no en abstracto. Así entendimos que, como la red termina en pooling global y una capa lineal, el gradiente del logit es el mismo $w_k^c/Z$ en las 49 posiciones, y por eso Grad-CAM y CAM coinciden. Eso nos permitió interpretar los errores de $10^{-6}$ a $10^{-7}$ como redondeo de `float32` y no como una diferencia real.

## Task 2: SHAP

**Prompt:** "Ayúdame a entender la fórmula de Shapley con los pesos factoriales y por qué los valores SHAP suman exactamente en log-odds pero no en probabilidad. ¿Cómo se ve eso con nuestro cliente rechazado?"

**Por qué funcionó:** al pedir el ejemplo con nuestro cliente, vimos la propiedad de eficiencia con números reales: el error máximo fue de $9.7\times10^{-8}$ en log-odds, mientras que en probabilidad la suma de efectos individuales (−0.946) no coincide con el cambio real (−0.755), porque la sigmoide no es lineal. Con eso pudimos responder el 2.6a con nuestros propios valores.

## Task 3: LIME y contrafactuales

**Prompt:** "Antes del task 3, explícame todas las fórmulas y sumatorias que aparecen (el kernel $\pi_x(z)$, la regresión ponderada, $R^2_w$, Jaccard, $N_{ef}$ y la distancia con MAD) con una explicación clara y natural, y cómo se relacionan con lo que tenemos."

**Por qué funcionó:** entender las fórmulas antes de programar nos dejó leer los resultados con sentido. Por ejemplo, que $N$ es el número de clientes inventados, que $N_{ef}=4808$ de 5000 significa que los pesos casi no varían entre esos clientes, y que $R^2_w=0.30$ indica que la explicación lineal solo captura el 30% del comportamiento del modelo. Las dudas que fueron saliendo (qué es $K$, qué es el MAD, qué es un contrafactual) las resolvimos sobre la marcha con los mismos datos.

## Task 4: Validación de Grad-CAM

**Prompt:** "Ayúdame a entender qué miden deletion, insertion y la prueba de aleatorización en cascada, y cómo leer el AUC y el Spearman con nuestros resultados."

**Por qué funcionó:** la explicación con analogías (comparar el mapa contra cavar al azar y hacer una prueba de cordura del método) nos permitió interpretar los casos inesperados con datos. Así explicamos por qué deletion empataba con el azar en la imagen (b) y por qué el 0.694 de la cascada en (a) era una tirada favorable, lo que nos llevó a agregar las dos corridas de verificación del 4.3.
