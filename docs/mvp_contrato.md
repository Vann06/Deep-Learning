# Contrato del MVP: todo lo que la interfaz necesita

Este documento es la entrega para quien arme la página de Streamlit. La inferencia ya está
corrida y guardada, así que la app solo tiene que leer dos archivos y maquetar.

## La decisión de fondo: la app no ejecuta modelos

Los 8,694 remitentes del conjunto de prueba se conocen de antemano, así que corrimos la
inferencia una sola vez en local y guardamos el resultado. La interfaz no necesita PyTorch,
no carga pesos y no calcula nada.

Esto responde directamente a la advertencia del enunciado de que un MVP que no corre vale
cero puntos. Sin torch, la app arranca en segundos y no puede quedarse sin memoria.

Los modelos siguen versionados en `artifacts/` y los notebooks ejecutados muestran que
producen exactamente estos números, así que no se pierde nada del entregable técnico.

**No hace falta base de datos.** Streamlit Cloud despliega desde GitHub, así que el
repositorio es la capa de persistencia y la app solo lee.

---

## Los dos archivos que consume

### `artifacts/mvp_data.csv.gz` (727 KB, 8,694 filas)

Una fila por remitente del conjunto de prueba.

| Columna | Tipo | Contenido |
|---|---|---|
| `sender_id` | texto | Identidad. Es la llave para unir con el contexto |
| `longitud` | entero | Transacciones reales, entre 3 y 32 |
| `y_real` | 0 o 1 | La verdad conocida. Sirve para mostrar aciertos y fallos |
| `score_a` | decimal | Error de reconstrucción de la Etapa A |
| `z_a` | decimal | El mismo score en desviaciones sobre el cliente normal promedio |
| `alerta_a` | booleano | Si `score_a` supera el umbral 0.11778 |
| `prob_b` | decimal | Probabilidad de lavado de la Etapa B, entre 0 y 1 |
| `alerta_b` | booleano | Si `prob_b` supera el umbral 0.95214 |
| `atencion` | JSON | Lista de pesos, uno por transacción real. Suma 1 |
| `explicacion` | texto | Párrafo en lenguaje natural ya redactado |

La lista de `atencion` viene recortada a `longitud`, o sea que su posición `i` corresponde
al `timestep = i` del contexto. No hay que filtrar relleno.

### `artifacts/test_context.csv.gz` (2.2 MB)

Las transacciones legibles, ya versionado desde el notebook 01. Una fila por transacción.

Columnas útiles para la tabla: `timestep`, `timestamp`, `amount_paid`, `payment_currency`,
`amount_received`, `receiving_currency`, `payment_format`, `from_bank`, `to_bank`,
`destination_account`, `is_laundering`.

---

## Cómo cargar los datos de un remitente

```python
import json
import pandas as pd
import streamlit as st

@st.cache_data
def cargar():
    datos = pd.read_csv("artifacts/mvp_data.csv.gz")
    contexto = pd.read_csv("artifacts/test_context.csv.gz")
    return datos, contexto

datos, contexto = cargar()

def remitente(sender_id):
    fila = datos[datos.sender_id == sender_id].iloc[0]
    transacciones = (
        contexto[contexto.sender_id == sender_id]
        .sort_values("timestep")
        .reset_index(drop=True)
    )
    pesos = json.loads(fila.atencion)          # len(pesos) == len(transacciones)
    return fila, transacciones, pesos
```

Los pesos quedan alineados fila por fila con la tabla de transacciones, así que el mapa de
calor se arma agregando `pesos` como columna o coloreando con `background_gradient`.

---

## Los cinco requisitos y de dónde sale cada uno

| Requisito del enunciado | De dónde sale |
|---|---|
| Seleccionar un remitente del conjunto de prueba | `datos.sender_id`, los 8,694 |
| Ver su secuencia con información básica | `contexto` filtrado por `sender_id` y ordenado por `timestep` |
| Score de anomalía de la Etapa A | `z_a`, que es lo interpretable. Conviene mostrarlo como "2.7 desviaciones sobre un cliente normal" en vez del `score_a` crudo |
| Probabilidad de lavado de la Etapa B | `prob_b`, y `alerta_b` para el semáforo |
| Mapa de calor sobre la secuencia | la lista de `atencion` |
| Párrafo de explicación automático | `explicacion` |

La columna `explicacion` está lista para mostrarse tal cual. Se arma con los números reales
de cada remitente: nivel de alerta, probabilidad, desviaciones, en qué transacción se
concentró la atención y qué tiene de particular esa operación. Se puede usar directamente o
generar una propia.

### Sugerencia para la demostración

Conviene agregar filtros por `alerta_b` y por `y_real`, porque con 8,694 remitentes en una
lista es difícil encontrar un caso interesante durante la presentación. Tres casos que vale
la pena tener a mano:

| `sender_id` | Qué muestra |
|---|---|
| `235874:80D76EB80` | Caso de lavado detectado por ambas etapas, atención concentrada al 100% |
| `701:805B8DD30` | El cliente más atípico de los 8,694, a 41 desviaciones, y es limpio. La Etapa B lo descarta bien |
| `70:100428810` | Caso de lavado que se escapa por cuatro cienmilésimas de probabilidad |

---

## Dos cosas que pueden costar el despliegue

### 1. El `requirements.txt` de la raíz pide torch

Ese archivo instala `torch` sin especificar variante, y en Streamlit Cloud eso descarga la
rueda con CUDA, de unos 2.5 GB, contra aproximadamente 1 GB de RAM del plan gratuito. Es la
causa más común de que un despliegue con PyTorch falle.

**No hay que modificarlo**, porque Colab sí necesita la versión con GPU para los notebooks.

La solución es crear un `app/requirements.txt`, junto al archivo de entrada de la app, con
solo lo que la interfaz usa:

```
streamlit
pandas
numpy
plotly
```

La documentación de Streamlit confirma la precedencia:

> "Community Cloud will search the directory where your entrypoint file is, then it will
> search the root of your repository."

Y aclara que solo se usa uno de los dos archivos, con el del directorio de entrada ganando.
Así que la raíz queda intacta.

### 2. Hay que desplegar desde la rama `Proyecto2`

La rama `main` del repositorio está vacía, solo tiene el commit inicial. Todo el proyecto
vive en `Proyecto2`, así que al configurar la app en share.streamlit.io:

```
Repository:     Vann06/Deep-Learning
Branch:         Proyecto2
Main file path: app/app.py
```

---

## Qué falta entregar además de la app

El enunciado pide la URL de la interfaz en un archivo de texto. Una vez desplegada, hay que
crear un `.txt` en la raíz del repositorio con el enlace y commitearlo.

## Cómo regenerar el precómputo

Si los modelos se reentrenan, este archivo queda desactualizado. El script que lo genera
carga `stage_a_model.pt` y `stage_b_model.pt`, calcula scores, probabilidades y pesos de
atención sobre `test.npz`, y redacta las explicaciones cruzando con `test_context.csv.gz`.
Basta volver a correrlo para regenerar `mvp_data.csv.gz`.
