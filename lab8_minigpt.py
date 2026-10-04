"""Mini-GPT a nivel de caracteres para CC3092 - Laboratorio 8.

La implementacion evita ``nn.MultiheadAttention`` y las capas Transformer de
alto nivel.  De esa forma quedan visibles las ecuaciones de atencion, mascara
causal, conexiones residuales y normalizacion que se reutilizan del Proyecto 1.

Relacion con las diapositivas de S13 (NLP y Generacion de Texto):
- Diap. 6, Transformer: self-attention sobre toda la secuencia en paralelo
  (``CausalSelfAttention`` y ``make_positional_encoding``).
- Diap. 8, GPT: solo decoder, mascara causal y tarea de predecir el siguiente
  token dado el contexto previo (``causal_mask``, ``get_batch``, ``MiniGPT``).
- Diap. 10, temperatura y muestreo: P(siguiente token | contexto) = softmax(z/tau),
  greedy como el limite tau -> 0, top-k y top-p (funciones ``sample_*``).
"""

from __future__ import annotations

import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

SamplingStrategy = Literal["greedy", "top-k", "top-p"]


def set_seed(seed: int = 42) -> None:
    """Fija las semillas usadas por Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tiny_shakespeare(path: str | Path = "data/tiny_shakespeare.txt") -> str:
    """Descarga Tiny Shakespeare una vez y retorna el texto completo."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class CharacterCorpus:
    """Texto codificado y dividido 90/10 sin romper el orden temporal."""

    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"El texto contiene caracteres fuera del vocabulario: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: list[int] | torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().tolist()
        return "".join(self.itos[int(i)] for i in indices)


def build_corpus(text: str, train_fraction: float = 0.90) -> CharacterCorpus:
    """Construye el vocabulario ordenado y la division temporal requerida."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(train_fraction * len(encoded))
    return CharacterCorpus(encoded[:split], encoded[split:], stoi, itos)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crea pares x/y donde y es x desplazado un caracter al futuro."""
    # Diap. 8 (GPT): la tarea es predecir el siguiente token dado el contexto
    # previo. Por eso y_t = x_{t+1}: cada posicion de x tiene su objetivo en y.
    if len(data) <= block_size:
        raise ValueError("El split debe contener mas caracteres que block_size.")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def iter_epoch_batches(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Recorre una vez todos los bloques completos de un split.

    En entrenamiento se baraja el orden de los bloques, no el orden interno de
    cada secuencia. En validacion se mantiene un recorrido determinista.
    """
    if len(data) <= block_size:
        raise ValueError("El split debe contener mas caracteres que block_size.")

    usable_tokens = ((len(data) - 1) // block_size) * block_size
    starts = torch.arange(0, usable_tokens, block_size)
    if shuffle:
        starts = starts[torch.randperm(len(starts))]

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size].tolist()
        x = torch.stack([data[i : i + block_size] for i in batch_starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in batch_starts])
        yield x.to(device), y.to(device)


def layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm manual sobre la dimension de caracteristicas."""
    # LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta, con mu y sigma^2
    # calculadas sobre las d_model componentes de cada token (Proyecto 1).
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(variance + eps) + beta


def make_positional_encoding(block_size: int, d_model: int) -> torch.Tensor:
    """Codificacion posicional sinusoidal de Vaswani et al."""
    # Diap. 6: el Transformer procesa todas las posiciones en paralelo, asi que
    # el orden se inyecta sumando PE al embedding:
    #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    # ``frequencies`` es 1 / 10000^(2i/d_model), escrito como exp(-2i ln(10000) / d_model).
    pe = torch.zeros(block_size, d_model)
    positions = torch.arange(block_size).unsqueeze(1).float()
    frequencies = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * frequencies)
    pe[:, 1::2] = torch.cos(positions * frequencies)
    return pe


class CausalSelfAttention(nn.Module):
    """Self-attention multi-head implementada directamente con tensores."""

    # ``register_buffer`` crea el atributo dinamicamente. La anotacion explicita
    # permite que Pylance sepa que se puede indexar como un Tensor.
    causal_mask: torch.Tensor

    def __init__(self, d_model: int, n_heads: int, block_size: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible entre n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        scale = 0.02
        self.WQ = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.WK = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.WV = nn.Parameter(torch.randn(d_model, d_model) * scale)
        self.WO = nn.Parameter(torch.randn(d_model, d_model) * scale)
        # Diap. 8 (GPT): "cada token solo puede atender a los tokens anteriores".
        # triu(diagonal=1) marca con True las posiciones j > i (el futuro de i).
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape

        # Q, K y V: (B, T, C) -> (B, h, T, d_head).
        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, time, self.n_heads, self.d_head).transpose(1, 2)

        # Q = X W_Q, K = X W_K, V = X W_V, divididas en h cabezas de d_head = d_model / h.
        queries = split_heads(x @ self.WQ)
        keys = split_heads(x @ self.WK)
        values = split_heads(x @ self.WV)

        # Diap. 6: cada palabra atiende a las demas. score(q, k) = q k^T / sqrt(d_head);
        # dividir entre sqrt(d_head) evita que el softmax se sature.
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.d_head)
        # Diap. 8: mascara causal ANTES del softmax. Los scores del futuro valen -inf
        # y exp(-inf) = 0, asi que su peso de atencion queda exactamente en 0.
        scores = scores.masked_fill(self.causal_mask[:time, :time], float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V; luego se concatenan
        # las cabezas y se proyectan con W_O.
        attended = weights @ values
        attended = attended.transpose(1, 2).contiguous().view(batch, time, self.d_model)
        return attended @ self.WO, weights


class TransformerBlock(nn.Module):
    """Bloque decoder: atencion causal, residual, LayerNorm y FFN."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, block_size: int) -> None:
        super().__init__()
        scale = 0.02
        self.attention = CausalSelfAttention(d_model, n_heads, block_size)
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * scale)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * scale)
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.beta1 = nn.Parameter(torch.zeros(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.beta2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, weights = self.attention(x)
        # Residual + LayerNorm: x = LN(x + Atencion(x)).
        x = layer_norm(x + attention_output, self.gamma1, self.beta1)
        # FFN(x) = ReLU(x W1 + b1) W2 + b2, aplicada a cada posicion por separado.
        feed_forward = F.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        x = layer_norm(x + feed_forward, self.gamma2, self.beta2)
        return x, weights


class MiniGPT(nn.Module):
    """Decoder-only apilado para language modeling a nivel de caracter."""

    # Igual que causal_mask, PE es un buffer (no un parametro entrenable).
    PE: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.register_buffer("PE", make_positional_encoding(block_size, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, d_ff, n_heads, block_size) for _ in range(n_layers)]
        )
        self.Wlm = nn.Parameter(torch.randn(d_model, vocab_size) * 0.02)
        self.blm = nn.Parameter(torch.zeros(vocab_size))

    def forward(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]
    ):
        if indices.ndim != 2:
            raise ValueError("indices debe tener forma (batch, tiempo).")
        _, time = indices.shape
        if time > self.block_size:
            raise ValueError(f"La secuencia excede block_size={self.block_size}.")

        # Diap. 3 y 4: el modelo no procesa caracteres sino vectores. E es la tabla
        # de embeddings (una fila por caracter) y PE agrega la posicion.
        x = self.E[indices] + self.PE[:time]
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attention:
                attention_maps.append(weights)
        # z = h W_lm + b_lm: los logits z de la diap. 10, uno por caracter del vocabulario.
        logits = x @ self.Wlm + self.blm

        loss = None
        if targets is not None:
            # Cross-entropy L = -mean_t log P(x_{t+1} | x_1..x_t), con P = softmax(z):
            # la misma distribucion de la diap. 10 con tau = 1.
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        if return_attention:
            return logits, loss, attention_maps
        return logits, loss


@torch.no_grad()
def evaluate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Calcula cross-entropy media por token sobre un split completo."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in iter_epoch_batches(
        data, block_size, batch_size, device, shuffle=False
    ):
        _, loss = model(x, y)
        assert loss is not None
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    model.train(was_training)
    return total_loss / total_tokens


def train_model(
    model: MiniGPT,
    corpus: CharacterCorpus,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: torch.device | None = None,
    verbose: bool = True,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> list[dict[str, float]]:
    """Entrena por epocas y registra loss y perplejidad de validacion."""
    device = device or next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    start_epoch = 1

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = saved["history"]
        start_epoch = len(history) + 1
        if verbose:
            print(f"Reanudando desde la epoca {start_epoch}.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        for x, y in iter_epoch_batches(
            corpus.train_data,
            model.block_size,
            batch_size,
            device,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * y.numel()
            running_tokens += y.numel()

        val_loss = evaluate_loss(
            model,
            corpus.val_data,
            model.block_size,
            batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "optimization_loss": running_loss / running_tokens,
            "train_loss": running_loss / running_tokens,
            "val_loss": val_loss,
            # Perplejidad = exp(L_val): numero efectivo de opciones entre las que duda el modelo.
            "val_perplexity": math.exp(val_loss),
        }
        history.append(row)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint,
            )
        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs}: "
                f"train={row['train_loss']:.4f}, val={row['val_loss']:.4f}, "
                f"ppl={row['val_perplexity']:.3f}"
            )
    return history


def _distribution_metrics(logits: torch.Tensor, tau: float) -> dict[str, float | int]:
    # Diap. 10: P(siguiente token | contexto) = softmax(z / tau).
    probabilities = F.softmax(logits / tau, dim=-1)
    # Entropia H = -sum_i P_i log P_i: alta si la distribucion es plana (tau -> inf),
    # baja si se concentra en un token (tau -> 0).
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {
        "top_probability": float(probabilities.max().item()),
        "entropy": float(entropy.item()),
    }


def sample_greedy(logits: torch.Tensor) -> int:
    """Selecciona argmax; no introduce aleatoriedad."""
    # Diap. 10: con tau -> 0, softmax(z / tau) pone toda la masa en argmax_i z_i,
    # "siempre el token mas probable, texto repetitivo y predecible".
    return int(torch.argmax(logits).item())


def sample_top_k(
    logits: torch.Tensor, *, k: int, tau: float
) -> int:
    """Aplica temperatura, conserva los k logits mayores y renormaliza."""
    if tau <= 0:
        raise ValueError("tau debe ser mayor que cero.")
    if k <= 0:
        raise ValueError("k debe ser mayor que cero.")
    k = min(k, logits.numel())
    # Diap. 10: primero z / tau, que controla creatividad contra coherencia.
    scaled = logits / tau
    # Top-k: "solo considerar los k tokens mas probables". El resto queda en -inf,
    # y el softmax les asigna probabilidad 0 y renormaliza entre los k que quedan.
    top_values, top_indices = torch.topk(scaled, k)
    filtered_logits = torch.full_like(scaled, float("-inf"))
    filtered_logits.scatter_(0, top_indices, top_values)
    probabilities = F.softmax(filtered_logits, dim=-1)
    sampled_position = torch.multinomial(probabilities, num_samples=1)
    return int(sampled_position.item())


def sample_top_p(
    logits: torch.Tensor, *, p: float, tau: float
) -> int:
    """Muestreo nucleus: usa el conjunto minimo cuya masa alcanza p."""
    if tau <= 0:
        raise ValueError("tau debe ser mayor que cero.")
    if not 0 < p <= 1:
        raise ValueError("p debe estar en el intervalo (0, 1].")

    # Diap. 10: P = softmax(z / tau), ordenada de mayor a menor.
    probabilities = F.softmax(logits / tau, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    # Top-p: "solo considerar los tokens que acumulan probabilidad p". cumulative[j]
    # es la masa acumulada de los j+1 tokens mas probables.
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)

    # Se elimina un token solo si la masa anterior ya habia alcanzado p.
    remove = (cumulative - sorted_probabilities) >= p
    nucleus_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    nucleus_probabilities /= nucleus_probabilities.sum()
    sampled_position = torch.multinomial(nucleus_probabilities, num_samples=1)

    return int(sorted_indices[sampled_position].item())


def _candidate_count(
    logits: torch.Tensor,
    strategy: SamplingStrategy,
    **kwargs: float | int,
) -> int:
    """Cantidad de candidatos conservados; se usa solo para diagnostico."""
    if strategy == "greedy":
        return 1
    if strategy == "top-k":
        return min(int(kwargs["k"]), logits.numel())

    tau = float(kwargs["tau"])
    p = float(kwargs["p"])
    probabilities = F.softmax(logits / tau, dim=-1)
    sorted_probabilities = torch.sort(probabilities, descending=True).values
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    keep = (cumulative - sorted_probabilities) < p
    return int(keep.sum().item())


def _normalize_strategy(strategy: str) -> SamplingStrategy:
    normalized = strategy.lower().replace("_", "-")
    if normalized == "greedy":
        return "greedy"
    if normalized == "top-k":
        return "top-k"
    if normalized == "top-p":
        return "top-p"
    raise ValueError("strategy debe ser 'greedy', 'top-k' o 'top-p'.")


def sample_next_token(
    logits: torch.Tensor, strategy: str, **kwargs: float | int
) -> int:
    """Despacha a greedy, top-k o top-p usando nombres tolerantes."""
    normalized = _normalize_strategy(strategy)
    if normalized == "greedy":
        return sample_greedy(logits)
    if normalized == "top-k":
        return sample_top_k(logits, k=int(kwargs["k"]), tau=float(kwargs["tau"]))
    if normalized == "top-p":
        return sample_top_p(logits, p=float(kwargs["p"]), tau=float(kwargs["tau"]))
    raise AssertionError("Estrategia normalizada no contemplada.")


@overload
def generate_text(
    model: MiniGPT,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    *,
    stoi: dict[str, int],
    itos: dict[int, str],
    return_trace: Literal[False] = False,
    **kwargs: float | int,
) -> str: ...


@overload
def generate_text(
    model: MiniGPT,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    *,
    stoi: dict[str, int],
    itos: dict[int, str],
    return_trace: Literal[True],
    **kwargs: float | int,
) -> tuple[str, list[dict[str, float | int | str]]]: ...


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    *,
    stoi: dict[str, int],
    itos: dict[int, str],
    return_trace: bool = False,
    **kwargs: float | int,
) -> str | tuple[str, list[dict[str, float | int | str]]]:
    """Genera exactamente ``max_new_tokens`` caracteres de forma autoregresiva."""
    # Diap. 8 y 10: en cada paso el modelo da una distribucion sobre el vocabulario,
    # se elige un token con la estrategia y se agrega al contexto del siguiente paso.
    if not prompt:
        raise ValueError("prompt no puede estar vacio.")
    missing = sorted(set(prompt) - set(stoi))
    if missing:
        raise ValueError(f"El prompt contiene caracteres desconocidos: {missing!r}")

    device = next(model.parameters()).device
    generated = torch.tensor(
        [[stoi[ch] for ch in prompt]], dtype=torch.long, device=device
    )
    trace: list[dict[str, float | int | str]] = []
    model.eval()

    for step in range(max_new_tokens):
        context = generated[:, -model.block_size :]
        logits, _ = model(context)
        # Solo importan los logits z de la ultima posicion: P(x_{t+1} | x_1..x_t).
        next_logits = logits[0, -1]
        next_index = sample_next_token(next_logits, strategy, **kwargs)
        next_tensor = torch.tensor([[next_index]], dtype=torch.long, device=device)
        generated = torch.cat((generated, next_tensor), dim=1)
        if return_trace:
            normalized_strategy = _normalize_strategy(strategy)
            tau = 1.0 if normalized_strategy == "greedy" else float(kwargs["tau"])
            metrics = _distribution_metrics(next_logits, tau)
            metrics["candidate_count"] = _candidate_count(
                next_logits,
                normalized_strategy,
                **kwargs,
            )
            current_text = "".join(itos[int(i)] for i in generated[0].tolist())
            trace.append(
                {
                    "step": step + 1,
                    "context": current_text[:-1][-50:],
                    "chosen_character": itos[next_index],
                    **metrics,
                }
            )

    text = "".join(itos[int(i)] for i in generated[0].tolist())
    return (text, trace) if return_trace else text


TASK2_CONFIGS = {
    "A": {"strategy": "greedy"},
    "B": {"strategy": "top-k", "k": 5, "tau": 0.7},
    "C": {"strategy": "top-k", "k": 50, "tau": 1.0},
    "D": {"strategy": "top-p", "p": 0.70, "tau": 0.7},
    "E": {"strategy": "top-p", "p": 0.95, "tau": 1.0},
}


def run_task2_experiment(
    generate: Callable,
    prompt: str = "ROMEO: ",
    samples_per_configuration: int = 5,
    max_new_tokens: int = 200,
    seed: int = 2026,
) -> dict[str, list[dict]]:
    """Ejecuta las 25 generaciones requeridas con semillas reproducibles."""
    results: dict[str, list[dict]] = {}
    for config_index, (name, parameters) in enumerate(TASK2_CONFIGS.items()):
        results[name] = []
        strategy = parameters["strategy"]
        sampling_parameters = {
            key: value for key, value in parameters.items() if key != "strategy"
        }
        for sample_index in range(samples_per_configuration):
            torch.manual_seed(seed + 100 * config_index + sample_index)
            text, trace = generate(
                prompt,
                max_new_tokens,
                strategy,
                return_trace=True,
                **sampling_parameters,
            )
            results[name].append({"text": text, "trace": trace})
    return results


def repetition_rate(text: str, ngram_size: int = 4) -> float:
    """Fraccion de n-gramas repetidos; complementa el analisis cualitativo."""
    if len(text) < ngram_size:
        return 0.0
    ngrams = [text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def summarize_results(
    results: dict[str, list[dict]], prompt: str = "ROMEO: "
) -> list[dict[str, float | str]]:
    """Resume diversidad, entropia y tamano del conjunto candidato."""
    rows = []
    for name, samples in results.items():
        traces = [step for sample in samples for step in sample["trace"]]
        generated_parts = [sample["text"][len(prompt) :] for sample in samples]
        rows.append(
            {
                "configuration": name,
                "mean_repetition_rate": float(
                    np.mean([repetition_rate(text) for text in generated_parts])
                ),
                "mean_entropy": float(np.mean([step["entropy"] for step in traces])),
                "mean_top_probability": float(
                    np.mean([step["top_probability"] for step in traces])
                ),
                "mean_candidate_count": float(
                    np.mean([step["candidate_count"] for step in traces])
                ),
            }
        )
    return rows


def confidence_moments(
    sample: dict, count: int = 2
) -> dict[str, list[dict[str, float | str]]]:
    """Encuentra pasos de menor/mayor entropia para analizar top-p E."""
    ordered = sorted(sample["trace"], key=lambda step: step["entropy"])
    return {"confident": ordered[:count], "uncertain": ordered[-count:][::-1]}


def save_samples(results: dict[str, list[dict]], path: str | Path) -> None:
    """Guarda las 25 muestras en un archivo de texto facil de citar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for name, samples in results.items():
        sections.append(f"CONFIGURACION {name}\n{'=' * 80}")
        for index, sample in enumerate(samples, start=1):
            sections.append(f"Muestra {index}\n{'-' * 80}\n{sample['text']}")
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
