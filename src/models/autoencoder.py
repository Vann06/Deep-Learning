"""Autoencoder secuencial para la Etapa A: aprender qué es un remitente normal.

Se entrena exclusivamente sobre secuencias normales. El error de reconstrucción
por secuencia, calculado únicamente sobre los pasos reales (enmascarados), es
el score de anomalía que consume `src/evaluation/anomaly.py`.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.models.encoder import EncoderConfig, SequenceEncoder


class SequenceDecoder(nn.Module):
    """Reconstruye la secuencia desde el latente, sin autorregresión.

    Repetir el latente en los `T` pasos y dejar que la GRU lo despliegue evita
    el *exposure bias* del teacher forcing: la reconstrucción no depende de
    los valores reales del paso anterior, solo del resumen comprimido.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.latent_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(config.hidden_size, config.num_features)

    def forward(self, latent: Tensor, sequence_length: int) -> Tensor:
        repeated = latent.unsqueeze(1).expand(-1, sequence_length, -1)
        hidden, _ = self.gru(repeated)
        return self.output_layer(hidden)


class SequenceAutoencoder(nn.Module):
    """Compone encoder y decoder; expone `.encoder` para transferirlo a la Etapa B."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config)
        self.decoder = SequenceDecoder(config)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states, latent = self.encoder(x, lengths)
        x_hat = self.decoder(latent, x.shape[1])
        return x_hat, hidden_states, latent


def masked_reconstruction_error(
    x: Tensor, x_hat: Tensor, mask: Tensor, *, per_sequence: bool = False
) -> Tensor:
    """MSE calculado solo sobre pasos reales (`mask=True`).

    Sin esta máscara, el padding en cero se reconstruye trivialmente y el
    error promedio baja mecánicamente con la longitud de la secuencia,
    confundiendo "historial corto" con "comportamiento normal".
    """
    num_features = x.shape[-1]
    mask_f = mask.to(dtype=x.dtype)
    squared_error = (x - x_hat).pow(2).sum(dim=-1)  # [B, T]
    masked_error = squared_error * mask_f
    if per_sequence:
        denom = mask_f.sum(dim=1).clamp(min=1.0) * num_features
        return masked_error.sum(dim=1) / denom
    denom = mask_f.sum().clamp(min=1.0) * num_features
    return masked_error.sum() / denom


def _run_epoch(
    model: SequenceAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    clip_grad: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, lengths, mask in loader:
            x = x.to(device)
            mask = mask.to(device)
            if training:
                optimizer.zero_grad()
            x_hat, _, _ = model(x, lengths)
            loss = masked_reconstruction_error(x, x_hat, mask)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def fit_autoencoder(
    model: SequenceAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    patience: int = 8,
    device: torch.device,
    clip_grad: float = 1.0,
    min_delta: float = 1e-5,
) -> dict:
    """Entrena con early stopping sobre la pérdida de reconstrucción en VAL.

    `val_loader` debe contener únicamente secuencias normales: el early
    stopping decide sobre la misma distribución que vio el entrenamiento,
    nunca sobre secuencias sospechosas.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, device, optimizer, clip_grad)
        val_loss = _run_epoch(model, val_loader, device, None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    return history
