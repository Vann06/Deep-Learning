"""mvp streamlit del sistema aml, ve docs/mvp_contrato.md para el contrato completo.

la app no ejecuta ningun modelo: toda la inferencia ya se corrio una vez con
scripts/build_mvp_data.py y quedo guardada en artifacts/mvp_data.csv.gz. por
eso aqui nunca se importa torch ni las funciones predict_aml/get_anomaly_score
(esas si cargan pesos de pytorch y son justo lo que rompe un deploy gratuito
en streamlit cloud por ram). docs/DIVISION.md sugeria usarlas, pero
mvp_contrato.md es la decision vigente y mas reciente del equipo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
UMBRAL_B = 0.95214

# el umbral de etapa a esta en la escala del error crudo (mse), pero z_a ya
# viene en sigmas en mvp_data.csv.gz, asi que convertimos el umbral una sola
# vez aqui usando las mismas stats que uso el script que genero los datos
_INFO_A = json.loads((ARTIFACTS / "anomaly_threshold.json").read_text(encoding="utf-8"))
_STATS_A = _INFO_A["normal_train_error_stats"]
UMBRAL_A_SIGMAS = (_INFO_A["selected"]["value"] - _STATS_A["mean"]) / _STATS_A["std"]

# columnas de test_context que si le sirven al analista; el csv trae mas
# (sequence_id, source_row_id, sender_account) que son solo de trazabilidad
# interna y no aportan nada a la tabla que ve el usuario
COLUMNAS_TABLA = [
    "timestep",
    "timestamp",
    "amount_paid",
    "payment_currency",
    "amount_received",
    "receiving_currency",
    "payment_format",
    "from_bank",
    "to_bank",
    "destination_account",
    "is_laundering",
]


@st.cache_data
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """lee los dos csv precomputados. streamlit cachea esto, asi que solo se lee una vez por sesion."""
    datos = pd.read_csv(ARTIFACTS / "mvp_data.csv.gz")
    contexto = pd.read_csv(ARTIFACTS / "test_context.csv.gz")
    return datos, contexto


def datos_remitente(
    datos: pd.DataFrame, contexto: pd.DataFrame, sender_id: str
) -> tuple[pd.Series, pd.DataFrame, list[float]]:
    """recorta todo lo que hace falta mostrar para un remitente puntual."""
    fila = datos.loc[datos.sender_id == sender_id].iloc[0]
    transacciones = (
        contexto.loc[contexto.sender_id == sender_id]
        .sort_values("timestep")
        .reset_index(drop=True)
    )
    # atencion[i] corresponde al timestep=i, ya viene recortada a la longitud real
    pesos = json.loads(fila["atencion"])
    return fila, transacciones, pesos


def texto_alerta_a(alerta: bool, z_a: float) -> str:
    """etapa a se mide en sigmas, no como probabilidad. score_a crudo (mse de
    reconstruccion) no tiene rango 0-1, asi que mostrarlo como % confunde."""
    icono = "🔴" if alerta else "🟢"
    return f"{icono} **Etapa A (anomalía):** {z_a:.1f}σ sobre un cliente normal (umbral {UMBRAL_A_SIGMAS:.1f}σ)"


def texto_alerta_b(alerta: bool, prob_b: float) -> str:
    """etapa b si es una probabilidad real, aqui el porcentaje tiene sentido."""
    icono = "🔴" if alerta else "🟢"
    return f"{icono} **Etapa B (lavado):** {prob_b:.1%} (umbral {UMBRAL_B:.1%})"


def main() -> None:
    st.set_page_config(page_title="Detección AML", layout="wide")
    st.title("Sistema secuencial de detección AML")
    st.caption(
        "Datos precomputados sobre el conjunto de prueba (8,694 remitentes). "
        "La app solo lee resultados ya calculados, no ejecuta modelos."
    )

    datos, contexto = cargar_datos()

    st.sidebar.header("Filtros")
    # con casi 9 mil remitentes hace falta poder acotar la lista para encontrar
    # casos interesantes en vivo, tal como sugiere el contrato
    filtro_alerta_b = st.sidebar.selectbox(
        "Alerta de Etapa B", ["Todas", "Solo alertadas", "Solo no alertadas"]
    )
    filtro_y_real = st.sidebar.selectbox(
        "Etiqueta real", ["Todas", "Solo lavado confirmado", "Solo limpias"]
    )

    filtrados = datos
    if filtro_alerta_b == "Solo alertadas":
        filtrados = filtrados[filtrados.alerta_b]
    elif filtro_alerta_b == "Solo no alertadas":
        filtrados = filtrados[~filtrados.alerta_b]
    if filtro_y_real == "Solo lavado confirmado":
        filtrados = filtrados[filtrados.y_real == 1]
    elif filtro_y_real == "Solo limpias":
        filtrados = filtrados[filtrados.y_real == 0]

    if filtrados.empty:
        st.warning("Ningún remitente cumple esa combinación de filtros.")
        return

    sender_id = st.sidebar.selectbox(
        f"Remitente ({len(filtrados):,} disponibles)", filtrados.sender_id.tolist()
    )

    fila, transacciones, pesos = datos_remitente(datos, contexto, sender_id)

    columnas = st.columns(3)
    columnas[0].metric("Transacciones", int(fila["longitud"]))
    columnas[1].metric("Etiqueta real", "Lavado" if fila["y_real"] else "Limpia")
    columnas[2].metric("Desviaciones (Etapa A)", f"{fila['z_a']:.1f}σ")

    st.markdown(texto_alerta_a(fila["alerta_a"], fila["z_a"]))
    st.markdown(texto_alerta_b(fila["alerta_b"], fila["prob_b"]))

    st.subheader("Secuencia de transacciones")
    st.dataframe(transacciones[COLUMNAS_TABLA], use_container_width=True)

    st.subheader("Dónde miró el modelo (atención de la Etapa B)")
    heatmap = pd.DataFrame(
        {
            "timestep": transacciones["timestep"],
            "atencion": pesos,
            "fecha": transacciones["timestamp"],
            "monto": transacciones["amount_paid"],
            "moneda": transacciones["payment_currency"],
            "formato": transacciones["payment_format"],
        }
    )
    figura = px.bar(
        heatmap,
        x="timestep",
        y="atencion",
        color="atencion",
        color_continuous_scale="Reds",
        hover_data=["fecha", "monto", "moneda", "formato"],
    )
    figura.update_layout(yaxis_title="peso de atención", xaxis_title="transacción (timestep)")
    st.plotly_chart(figura, use_container_width=True)

    st.subheader("Explicación")
    st.info(fila["explicacion"])


if __name__ == "__main__":
    main()
