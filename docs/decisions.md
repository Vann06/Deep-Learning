# Decisiones de ingeniería de datos

## 1. Dataset

PaySim se evaluó primero y se descartó para el objetivo secuencial por remitente. En 6,362,620 transacciones hay 6,353,307 remitentes; la mediana, p95 y p99 son una transacción, el máximo es 3, y solo 28 remitentes fraudulentos repiten al menos una vez. Construir secuencias de PaySim implicaría padding casi total y no representaría comportamiento histórico.

Se eligió IBM AML `HI-Small_Trans.csv`: 5,078,345 transacciones, 496,999 remitentes compuestos y 3,376 remitentes positivos. La variante Small conserva escala suficiente y cabe con holgura en el límite de 30 minutos de Colab.

## 2. Identidad y orden temporal

El remitente es `From Bank:Account`; `Account` por sí sola podría colisionar entre bancos. El archivo no resultó globalmente monotónico en tiempo, por lo que el pipeline ordena explícitamente cada historial por `sender_id`, `Timestamp` y número de fila original antes de crear ventanas.

## 3. Longitud y remitentes cortos

Se exige un mínimo de 3 transacciones: retiene 81.69% de los remitentes positivos, mientras que exigir 4 reduciría esa cobertura a 61.76%. Se usa longitud máxima 32 porque el p75 de los historiales elegibles es 30 y así se cubre completo 77.46% de ellos. Los historiales de 3–31 usan post-padding en cero y máscara. Los de menos de 3 se excluyen y quedan documentados como limitación.

Se guarda una ventana por remitente para evitar que cuentas extremadamente activas dominen el dataset. En positivos, la ventana incluye la última transacción etiquetada y el mayor contexto previo posible; en normales se conserva la ventana más reciente.

## 4. Features y normalización

Se incluyen monto pagado/recibido, diferencia absoluta, tiempo desde la operación previa, hora y día cíclicos, mismo banco, misma moneda, cambio/novedad de destino y one-hot de formato y monedas. No se usan IDs de cuenta/banco como features ni `Is Laundering`, evitando memorización y leakage explícito.

Montos y gap usan `log1p` por sus colas largas; luego `StandardScaler`. Variables cíclicas y binarias ya están acotadas. Categóricas usan one-hot con desconocidos ignorados. Scaler y vocabulario se ajustan exclusivamente con timesteps reales de TRAIN.

## 5. Splits y desbalance

El split 70/15/15 se hace por remitente y se estratifica por etiqueta. Se conservan los 2,758 remitentes positivos elegibles. Dentro de cada split se seleccionan 20 remitentes normales por positivo: reduce costo sin fingir balance perfecto y deja `pos_weight=20` como referencia inicial. Validation/test heredan el mismo muestreo para comparación controlada, pero no estiman precisión operativa con la prevalencia original de 0.102% por transacción.

## 6. Limitaciones

- IBM AML es sintético y no representa directamente remesas Guatemala–Estados Unidos.
- El mínimo de 3 excluye 618 remitentes positivos con uno o dos movimientos.
- Una ventana por remitente pierde historia en cuentas muy largas.
- La tasa positiva procesada está alterada por submuestreo; probabilidades futuras deben calibrarse con prevalencia real.
- Las etiquetas marcan transacciones conocidas, no toda conducta ilícita posible.
