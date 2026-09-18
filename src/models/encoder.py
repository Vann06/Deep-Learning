"""Encoder secuencial compartido entre la Etapa A y la Etapa B.

El contrato que expone hacia la Etapa B es devolver, además del vector
comprimido, los estados ocultos por timestep: son la entrada que necesita el
mecanismo de atención para explicar qué transacciones pesaron más en una
alerta. Un encoder que solo devolviera el bottleneck rompería ese contrato.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class EncoderConfig:
    """Hiperparámetros del encoder, serializados junto con los pesos."""

    num_features: int
    hidden_size: int = 64
    latent_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0


class SequenceEncoder(nn.Module):
    """GRU con soporte nativo de longitud variable vía secuencias empaquetadas."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.num_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.bottleneck = nn.Linear(config.hidden_size, config.latent_size)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Codifica un batch `[B, T, F]` con longitudes reales `lengths [B]`.

        Devuelve `(hidden_states, latent)`:
        - `hidden_states [B, T, hidden_size]`: salida de la GRU en cada paso,
          re-rellenada con ceros más allá de la longitud real. Es la entrada
          que consume la atención de la Etapa B.
        - `latent [B, latent_size]`: representación comprimida del último
          paso real de cada secuencia, resultado de aplicar el bottleneck
          sobre el último estado oculto de la GRU.
        """
        total_length = x.shape[1]
        lengths_cpu = lengths.detach().to("cpu", dtype=torch.int64)
        packed = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.gru(packed)
        hidden_states, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=total_length
        )
        latent = self.bottleneck(h_n[-1])
        return hidden_states, latent


def save_encoder(path: str | Path, model: SequenceEncoder, config: EncoderConfig) -> None:
    """Guarda pesos y configuración juntos para que `load_encoder` sea autosuficiente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def load_encoder(
    path: str | Path, *, map_location: str | torch.device | None = None
) -> tuple[SequenceEncoder, EncoderConfig]:
    """Reconstruye el encoder desde disco. Es la función que consume la Etapa B."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = EncoderConfig(**checkpoint["config"])
    model = SequenceEncoder(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config
