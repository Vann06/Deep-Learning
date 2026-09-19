"""Precomputa scores, atención y explicaciones para la interfaz del MVP.

La interfaz no ejecuta modelos. Como los remitentes del conjunto de prueba se
conocen de antemano, la inferencia se corre una vez aquí y se guarda en
`artifacts/mvp_data.csv.gz`. Así la app desplegada no necesita PyTorch, que es
la dependencia que más suele romper un despliegue en Streamlit Cloud.

El contrato de columnas está documentado en `docs/mvp_contrato.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_split
from src.evaluation.anomaly import reconstruction_scores
from src.models.autoencoder import SequenceAutoencoder
from src.models.classifier import load_classifier, make_supervised_loader, predict_scores
from src.models.encoder import EncoderConfig


def describir_transaccion(fila: pd.Series, grupo: pd.DataFrame) -> str:
    """Qué tiene de particular una transacción dentro de su propio historial."""
    rasgos = []
    montos = grupo["amount_paid"]
    if len(montos) > 2 and fila["amount_paid"] >= montos.quantile(0.9):
        rasgos.append("es de los montos más altos del historial")
    if fila["payment_currency"] != fila["receiving_currency"]:
        rasgos.append(f"convierte {fila['payment_currency']} a {fila['receiving_currency']}")
    previos = grupo[grupo["timestep"] < fila["timestep"]]["destination_account"]
    if len(previos) and fila["destination_account"] not in set(previos):
        rasgos.append("va a un destinatario nuevo")
    if fila["from_bank"] != fila["to_bank"]:
        rasgos.append("sale hacia otro banco")
    return ", ".join(rasgos) if rasgos else "no presenta rasgos inusuales por sí sola"


def redactar(
    largo: int,
    prob: float,
    z: float,
    alerta_a: bool,
    alerta_b: bool,
    pesos: np.ndarray,
    grupo: pd.DataFrame,
    umbral_b: float,
) -> str:
    """Párrafo en lenguaje natural armado con los números reales del remitente."""
    principal = int(np.argmax(pesos))
    fila = grupo.iloc[principal]

    # Cuando la probabilidad casi toca el umbral, un solo decimal los muestra iguales.
    casi = abs(prob - umbral_b) < 0.005
    prob_fmt = f"{prob:.4%}" if casi else f"{prob:.1%}"
    umbral_fmt = f"{umbral_b:.4%}" if casi else f"{umbral_b:.1%}"

    if alerta_b:
        cabeza = (
            f"El sistema clasifica a este remitente como sospechoso, con una probabilidad "
            f"estimada de {prob_fmt}, por encima del umbral de alerta de {umbral_fmt}."
        )
    elif prob >= 0.5:
        cabeza = (
            f"El sistema no levanta alerta, pero queda muy cerca de hacerlo: la "
            f"probabilidad estimada es {prob_fmt} contra un umbral de {umbral_fmt}. "
            f"Es un caso que conviene revisar manualmente."
        )
    else:
        cabeza = (
            f"El sistema no levanta alerta. La probabilidad estimada de lavado es "
            f"{prob_fmt}, muy por debajo del umbral de {umbral_fmt}."
        )

    if alerta_a and alerta_b:
        contexto = (
            f"El comportamiento del cliente es además atípico: reconstruye {z:.1f} "
            f"desviaciones peor que un cliente normal promedio."
        )
    elif alerta_a:
        contexto = (
            f"Las dos señales apuntan en direcciones distintas. El cliente sí resulta "
            f"atípico, a {z:.1f} desviaciones del cliente normal promedio, pero ese "
            f"comportamiento inusual no coincide con los patrones de lavado que el sistema "
            f"aprendió de los casos confirmados."
        )
    else:
        contexto = (
            f"En cuanto a qué tan atípico resulta, se ubica a {z:.1f} desviaciones del "
            f"cliente normal promedio, dentro del rango habitual."
        )

    foco = (
        f"De las {largo} transacciones del historial, el modelo concentró "
        f"{pesos[principal]:.0%} de su atención en la número {principal + 1}, del "
        f"{pd.to_datetime(fila['timestamp']).strftime('%d/%m/%Y')}, por "
        f"{fila['amount_paid']:,.0f} {fila['payment_currency']} mediante "
        f"{fila['payment_format']}. Esa operación {describir_transaccion(fila, grupo)}."
    )

    cierre = (
        "La señalización indica dónde miró el modelo y sirve para orientar la revisión, "
        "no como prueba de que esa transacción concreta sea la operación ilícita."
    )
    return " ".join([cabeza, contexto, foco, cierre])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cpu")  # el precómputo corre una vez, la CPU basta
    test = load_split(args.artifacts / "test.npz")
    lengths = test["lengths"].astype(int)

    checkpoint = torch.load(
        args.artifacts / "stage_a_model.pt", map_location=device, weights_only=False
    )
    autoencoder = SequenceAutoencoder(EncoderConfig(**checkpoint["config"]))
    autoencoder.load_state_dict(checkpoint["state_dict"])
    autoencoder.eval()
    score_a = reconstruction_scores(
        autoencoder, test["X"], test["mask"], test["lengths"], device=device
    )

    info_a = json.loads((args.artifacts / "anomaly_threshold.json").read_text(encoding="utf-8"))
    stats = info_a["normal_train_error_stats"]
    umbral_a = float(info_a["selected"]["value"])
    z_a = (score_a - stats["mean"]) / stats["std"]

    clasificador, _ = load_classifier(
        args.artifacts / "stage_b_model.pt", map_location=device
    )
    clasificador.to(device).eval()
    prob_b, _, _ = predict_scores(
        clasificador, make_supervised_loader(test, batch_size=args.batch_size), device
    )
    umbral_b = float(
        json.loads((args.artifacts / "stage_b_threshold.json").read_text(encoding="utf-8"))[
            "selected"
        ]["value"]
    )

    pesos = np.zeros((len(lengths), test["X"].shape[1]), dtype=np.float32)
    with torch.no_grad():
        for inicio in range(0, len(lengths), args.batch_size):
            corte = slice(inicio, inicio + args.batch_size)
            _, w = clasificador(
                torch.from_numpy(test["X"][corte]).to(device),
                torch.from_numpy(test["lengths"][corte].astype(np.int64)),
                torch.from_numpy(test["mask"][corte]).to(device),
            )
            pesos[corte] = w.cpu().numpy()

    contexto = pd.read_csv(args.artifacts / "test_context.csv.gz").sort_values(
        ["sender_id", "timestep"]
    )
    por_remitente = {sid: grupo for sid, grupo in contexto.groupby("sender_id", sort=False)}

    filas = []
    for i, sender_id in enumerate(test["sender_id"]):
        largo = int(lengths[i])
        w = pesos[i, :largo].astype(float)
        alerta_a = bool(score_a[i] >= umbral_a)
        alerta_b = bool(prob_b[i] >= umbral_b)
        filas.append(
            {
                "sender_id": sender_id,
                "longitud": largo,
                "y_real": int(test["y"][i]),
                "score_a": round(float(score_a[i]), 6),
                "z_a": round(float(z_a[i]), 3),
                "alerta_a": alerta_a,
                "prob_b": round(float(prob_b[i]), 6),
                "alerta_b": alerta_b,
                "atencion": json.dumps([round(v, 5) for v in w]),
                "explicacion": redactar(
                    largo, float(prob_b[i]), float(z_a[i]), alerta_a, alerta_b,
                    w, por_remitente[sender_id], umbral_b,
                ),
            }
        )

    salida = args.artifacts / "mvp_data.csv.gz"
    pd.DataFrame(filas).to_csv(salida, index=False, compression="gzip")
    print(f"{salida}: {len(filas):,} filas, {salida.stat().st_size / 1024:.0f} KB")
    print(f"Umbrales aplicados: Etapa A {umbral_a:.5f}, Etapa B {umbral_b:.5f}")


if __name__ == "__main__":
    main()
