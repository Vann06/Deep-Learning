"""Clasificador supervisado de la Etapa B: encoder de la Etapa A + atención + cabeza.

El encoder se reutiliza desde `src/models/encoder.py`, nunca se redefine: los dos
brazos de la ablación (preentrenado vs. inicialización aleatoria) deben ser
idénticos en todo salvo en cómo arrancan sus pesos, y eso solo se garantiza si
comparten la misma clase y la misma `EncoderConfig`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.attention import AdditiveAttention
from src.models.encoder import EncoderConfig, SequenceEncoder, load_encoder
from src.utils import get_device


@dataclass(frozen=True)
class ClassifierConfig:
    """Hiperparámetros del clasificador, serializados junto con los pesos."""

    encoder: EncoderConfig
    attention_size: int = 32
    head_hidden: int = 32
    dropout: float = 0.2


class AMLClassifier(nn.Module):
    """Encoder secuencial → atención enmascarada → cabeza binaria.

    La cabeza consume **solo** el contexto de la atención, no el vector latente
    del encoder. Es una decisión deliberada: así la probabilidad de salida es
    íntegramente una suma ponderada de representaciones por transacción, y los
    pesos de atención explican el total de la predicción. Concatenar el latente
    mejoraría quizá algún decimal, pero dejaría parte de la señal fuera del
    mapa de calor y volvería la interpretabilidad una verdad a medias.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequenceEncoder(config.encoder)
        self.attention = AdditiveAttention(config.encoder.hidden_size, config.attention_size)
        self.head = nn.Sequential(
            nn.Linear(config.encoder.hidden_size, config.head_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: Tensor, lengths: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve `(logits [B], attention_weights [B, T])`.

        Se devuelven logits y no probabilidades porque `BCEWithLogitsLoss` es
        numéricamente más estable que aplicar sigmoide y luego BCE.
        """
        hidden_states, _ = self.encoder(x, lengths)
        context, weights = self.attention(hidden_states, mask)
        logits = self.head(context).squeeze(-1)
        return logits, weights


def build_classifier(
    config: ClassifierConfig,
    *,
    pretrained_encoder_path: str | Path | None = None,
    map_location: str | torch.device | None = None,
) -> AMLClassifier:
    """Construye un brazo de la ablación.

    - `pretrained_encoder_path=None` → brazo *scratch*: el encoder arranca con
      la inicialización aleatoria de PyTorch.
    - ruta a `encoder.pt` → brazo *pretrained*: el encoder arranca con los pesos
      que la Etapa A aprendió sobre comportamiento normal.

    Todo lo demás (atención, cabeza, semilla, datos, optimizador) es idéntico,
    que es justamente lo que hace válida la comparación.
    """
    model = AMLClassifier(config)
    if pretrained_encoder_path is not None:
        encoder, encoder_config = load_encoder(pretrained_encoder_path, map_location=map_location)
        if encoder_config != config.encoder:
            raise ValueError(
                "La EncoderConfig guardada no coincide con la del clasificador: "
                f"{encoder_config} vs {config.encoder}"
            )
        model.encoder.load_state_dict(encoder.state_dict())
    return model


def make_supervised_loader(
    arrays: dict[str, np.ndarray], *, batch_size: int = 256, shuffle: bool = False
) -> DataLoader:
    """Empaqueta un split de `load_split` en el orden que espera `AMLClassifier`.

    A diferencia de la Etapa A, aquí entran **ambas clases**: el tensor `y`
    viaja con el batch porque el entrenamiento ya es supervisado.
    """
    dataset = TensorDataset(
        torch.from_numpy(arrays["X"]),
        torch.from_numpy(arrays["lengths"].astype(np.int64)),
        torch.from_numpy(arrays["mask"]),
        torch.from_numpy(arrays["y"].astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(
    model: AMLClassifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_grad: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for x, lengths, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, lengths, mask)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def predict_scores(
    model: AMLClassifier, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recorre un loader y devuelve `(probabilidades, logits, etiquetas)`."""
    model.eval()
    model.to(device)
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, lengths, mask, y in loader:
            x, mask = x.to(device), mask.to(device)
            logits, _ = model(x, lengths, mask)
            logits_chunks.append(logits.cpu().numpy())
            label_chunks.append(y.numpy())
    logits = np.concatenate(logits_chunks)
    labels = np.concatenate(label_chunks)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return probabilities, logits, labels


def fit_classifier(
    model: AMLClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 40,
    patience: int = 8,
    encoder_lr: float = 1e-4,
    head_lr: float = 1e-3,
    pos_weight: float = 20.0,
    device: torch.device,
    clip_grad: float = 1.0,
    min_delta: float = 1e-4,
) -> dict:
    """Entrena con early stopping sobre el **PR-AUC de validación**, no sobre la pérdida.

    Dos decisiones que van juntas:

    - `pos_weight=20` en `BCEWithLogitsLoss` compensa el submuestreo 20:1 del
      dataset (`metadata.json → imbalance.train_pos_weight_recommended`): sin
      él, predecir "todo normal" ya acierta el 95.2% y el gradiente casi no
      empuja hacia la clase positiva.
    - El criterio de parada es PR-AUC porque la Etapa A dejó una lección
      explícita: entrenar hasta minimizar la pérdida mejoró la reconstrucción
      pero empeoró la separación entre clases. Aquí se selecciona el checkpoint
      directamente por la métrica que interesa.

    El learning rate es diferenciado: el encoder se ajusta 10× más lento que la
    atención y la cabeza, para adaptarlo a la tarea supervisada sin borrar de
    golpe la normalidad que aprendió en la Etapa A.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.attention.parameters(), "lr": head_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    history: dict = {"train_loss": [], "val_pr_auc": [], "best_epoch": None}
    best_pr_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, device, optimizer, criterion, clip_grad)
        probabilities, _, labels = predict_scores(model, val_loader, device)
        val_pr_auc = float(average_precision_score(labels, probabilities))
        history["train_loss"].append(train_loss)
        history["val_pr_auc"].append(val_pr_auc)

        if val_pr_auc > best_pr_auc + min_delta:
            best_pr_auc = val_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_pr_auc"] = float(best_pr_auc)
    history["epochs_run"] = len(history["train_loss"])
    return history


def save_classifier(path: str | Path, model: AMLClassifier, config: ClassifierConfig) -> None:
    """Guarda pesos y configuración juntos, igual patrón que `save_encoder`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, path)


def load_classifier(
    path: str | Path, *, map_location: str | torch.device | None = None
) -> tuple[AMLClassifier, ClassifierConfig]:
    """Reconstruye el clasificador desde disco (la config viaja anidada)."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    raw = dict(checkpoint["config"])
    config = ClassifierConfig(encoder=EncoderConfig(**raw.pop("encoder")), **raw)
    model = AMLClassifier(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


_STAGE_B_CACHE: dict[str, tuple[AMLClassifier, ClassifierConfig]] = {}


def predict_aml(
    sequence: np.ndarray,
    length: int | None = None,
    *,
    model_path: str | Path = "artifacts/stage_b_model.pt",
    device: torch.device | None = None,
) -> tuple[float, np.ndarray]:
    """API pública de la Etapa B (contrato de `docs/DIVISION.md`).

    Recibe una secuencia `[T, F]` (o `[1, T, F]`) y su longitud real. Devuelve
    la tupla `(probabilidad, pesos_de_atención)` que espera la app del MVP.

    Los pesos vienen **recortados a la longitud real**, no a los 32 pasos
    acolchados: el heatmap debe tener exactamente una casilla por transacción
    que de verdad ocurrió. Suman 1.
    """
    device = device or get_device()
    key = str(model_path)
    if key not in _STAGE_B_CACHE:
        model, config = load_classifier(model_path, map_location=device)
        model.to(device)
        _STAGE_B_CACHE[key] = (model, config)
    model, _ = _STAGE_B_CACHE[key]

    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    total_length = x.shape[1]
    real_length = total_length if length is None else int(length)
    mask = np.arange(total_length)[None, :] < real_length

    x_t = torch.from_numpy(x).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    lengths_t = torch.tensor([real_length], dtype=torch.int64)

    with torch.no_grad():
        logits, weights = model(x_t, lengths_t, mask_t)
        probability = float(torch.sigmoid(logits)[0].item())
        attention = weights[0, :real_length].cpu().numpy()
    return probability, attention
