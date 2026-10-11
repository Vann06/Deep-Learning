"""Piezas reutilizables de la Task 3 del Lab 9: difusion condicional con guia.

Todo esta escrito con tensores de PyTorch. No se usa diffusers, ni
torch.distributions, ni calendarios o muestreadores prefabricados.
El calendario y q_sample son los mismos de la Task 2.1 (nb02).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

NUM_CLASSES = 10
NULL_CLASS = 10  # la clase 10 es la condicion nula


# ---------------------------------------------------------------------------
# calendario y proceso forward (task 2.1)
# ---------------------------------------------------------------------------

def linear_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    """beta_t lineal, alpha_t = 1 - beta_t y alpha_bar_t = prod_{s<=t} alpha_s."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample(x0, t, alpha_bars, noise=None, generator=None):
    """x_t = sqrt(a_bar_t) x_0 + sqrt(1 - a_bar_t) eps, con t de 1 a T (uno por imagen)."""
    if noise is None:
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
    t = torch.as_tensor(t, device=alpha_bars.device)
    a_bar = alpha_bars[t - 1].to(dtype=x0.dtype, device=x0.device)
    # forma (B, 1, 1, 1) para que cada imagen use su propio a_bar_t
    a_bar = a_bar.view(-1, *([1] * (x0.dim() - 1)))
    x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise
    return x_t, noise


# ---------------------------------------------------------------------------
# embedding del instante (misma formula del positional encoding de la semana 6)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """PE(t, 2i) = sin(t / 10000^(2i/d)) y PE(t, 2i+1) = cos(t / 10000^(2i/d))."""
    t = t.float().unsqueeze(1)  # (B, 1)
    # 1 / 10000^(2i/d) escrito como exp(-2i ln(10000) / d), igual que en la semana 6
    freqs = torch.exp(
        torch.arange(0, dim, 2, device=t.device).float() * (-math.log(10_000.0) / dim)
    )
    emb = torch.zeros(t.shape[0], dim, device=t.device)
    emb[:, 0::2] = torch.sin(t * freqs)
    emb[:, 1::2] = torch.cos(t * freqs)
    return emb


# ---------------------------------------------------------------------------
# u-net condicional
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """dos convoluciones 3x3 con groupnorm y silu; la condicion se suma en medio."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        # (B, C) -> (B, C, 1, 1): el mismo vector se suma en todos los pixeles (broadcasting)
        h = h + self.emb_proj(emb)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h)))


class UNet(nn.Module):
    """predice eps a partir de (x_t, t, c). niveles 28x28/32 canales y 14x14/64 canales."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(NUM_CLASSES + 1, emb_dim)  # 0..9 y la nula

        self.down1 = ConvBlock(1, 32, emb_dim)                            # 28x28, 32 canales
        self.downsample = nn.Conv2d(32, 32, 4, stride=2, padding=1)       # 28 -> 14
        self.down2 = ConvBlock(32, 64, emb_dim)                           # 14x14, 64 canales
        self.bottleneck = ConvBlock(64, 64, emb_dim)                      # cuello a 14x14
        self.up2 = ConvBlock(64 + 64, 64, emb_dim)                        # skip de down2
        self.upsample = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 14 -> 28
        self.up1 = ConvBlock(32 + 32, 32, emb_dim)                        # skip de down1
        self.out = nn.Conv2d(32, 1, 1)                                    # conv final a 1 canal

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # condicion = mlp(embedding sinusoidal de t) + embedding de la clase
        emb = self.time_mlp(sinusoidal_embedding(t, self.emb_dim)) + self.class_emb(c)
        h1 = self.down1(x, emb)
        h2 = self.down2(self.downsample(h1), emb)
        h = self.bottleneck(h2, emb)
        h = self.up2(torch.cat([h, h2], dim=1), emb)
        h = self.up1(torch.cat([self.upsample(h), h1], dim=1), emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# entrenamiento (task 3.2)
# ---------------------------------------------------------------------------

def diffusion_loss(model, x0, labels, alpha_bars, p_uncond: float) -> torch.Tensor:
    """t ~ U{1..T}, eps ~ N(0, I), clase nula con prob. p_uncond y MSE(eps, eps_theta)."""
    batch = x0.shape[0]
    T = alpha_bars.shape[0]
    t = torch.randint(1, T + 1, (batch,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t, _ = q_sample(x0, t, alpha_bars, noise=noise)
    drop = torch.rand(batch, device=x0.device) < p_uncond
    c = torch.where(drop, torch.full_like(labels, NULL_CLASS), labels)
    return F.mse_loss(model(x_t, t, c), noise)


def train_epoch(model, loader, optimizer, alpha_bars, p_uncond: float, device) -> float:
    """una pasada por el loader; devuelve la perdida media por imagen."""
    model.train()
    total, count = 0.0, 0
    for x0, labels in loader:
        x0, labels = x0.to(device), labels.to(device)
        loss = diffusion_loss(model, x0, labels, alpha_bars, p_uncond)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * x0.shape[0]
        count += x0.shape[0]
    return total / count


@torch.no_grad()
def interval_losses(model, loader, alpha_bars, device, n_intervals: int = 10, seed: int = 0):
    """MSE media por intervalo de t (de 1-100 a 901-1000), con la clase real."""
    model.eval()
    T = alpha_bars.shape[0]
    width = T // n_intervals
    losses = []
    for k in range(n_intervals):
        # mismo generador por intervalo para que la comparacion sea justa
        gen = torch.Generator(device=device).manual_seed(seed)
        total, count = 0.0, 0
        for x0, labels in loader:
            x0, labels = x0.to(device), labels.to(device)
            t = torch.randint(k * width + 1, (k + 1) * width + 1, (x0.shape[0],),
                              device=device, generator=gen)
            noise = torch.randn(x0.shape, device=device, generator=gen)
            x_t, _ = q_sample(x0, t, alpha_bars, noise=noise)
            mse = F.mse_loss(model(x_t, t, labels), noise, reduction="sum")
            total += mse.item()
            count += noise.numel()
        losses.append(total / count)
    return losses


# ---------------------------------------------------------------------------
# muestreo ancestral con guia sin clasificador (task 3.3 y 3.4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_eps(model, x, t, c, w: float) -> torch.Tensor:
    """eps_guiado = eps(x, t, nula) + w (eps(x, t, c) - eps(x, t, nula))."""
    if w == 1:
        # con w = 1 la formula se reduce exactamente a eps(x, t, c): una sola pasada
        return model(x, t, c)
    null = torch.full_like(c, NULL_CLASS)
    eps_c, eps_null = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, null])).chunk(2)
    return eps_null + w * (eps_c - eps_null)


def ancestral_step(x_t, eps, t: int, betas, alphas, alpha_bars, z=None):
    """x_{t-1} = (x_t - (1 - a_t)/sqrt(1 - a_bar_t) eps) / sqrt(a_t) + sigma_t z, sigma_t^2 = beta_t."""
    a, a_bar, beta = alphas[t - 1].item(), alpha_bars[t - 1].item(), betas[t - 1].item()
    mean = (x_t - (1 - a) / math.sqrt(1 - a_bar) * eps) / math.sqrt(a)
    if t == 1 or z is None:
        return mean  # en el ultimo paso z = 0
    return mean + math.sqrt(beta) * z


@torch.no_grad()
def sample(model, labels, w: float, betas, alphas, alpha_bars, generator=None):
    """parte de x_T ~ N(0, I) y aplica el paso ancestral de t = T a 1."""
    model.eval()
    device = labels.device
    T = betas.shape[0]
    x = torch.randn(len(labels), 1, 28, 28, device=device, generator=generator)
    for t in range(T, 0, -1):
        tt = torch.full((len(labels),), t, device=device, dtype=torch.long)
        eps = guided_eps(model, x, tt, labels, w)
        z = torch.randn(x.shape, device=device, generator=generator) if t > 1 else None
        x = ancestral_step(x, eps, t, betas, alphas, alpha_bars, z)
    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# evaluacion: clasificador, fidelidad, diversidad y confusion (task 3.4 y 3.5)
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    """cnn sencilla; baja resolucion con convoluciones de stride 2 en vez de pooling."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),   # 28 -> 14
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),   # 14 -> 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / count


@torch.no_grad()
def predict(model, images, batch_size: int = 500) -> torch.Tensor:
    """clase predicha para cada imagen."""
    model.eval()
    return torch.cat([model(images[i:i + batch_size]).argmax(dim=1)
                      for i in range(0, len(images), batch_size)])


def fidelity(classifier, images, labels) -> float:
    """fraccion de imagenes que el clasificador asigna a la clase pedida."""
    return (predict(classifier, images) == labels).float().mean().item()


def diversity_per_class(images, labels) -> torch.Tensor:
    """distancia euclidiana media entre pares distintos de la misma clase, una por clase."""
    flat = images.flatten(start_dim=1).float()
    result = torch.zeros(NUM_CLASSES)
    for k in range(NUM_CLASSES):
        x = flat[labels == k]
        n = x.shape[0]
        dist = torch.cdist(x, x)
        # cdist puede dejar residuos numericos en la diagonal; se excluye a mano
        off_diag = ~torch.eye(n, dtype=torch.bool, device=dist.device)
        result[k] = dist[off_diag].mean()
    return result


def confusion_matrix(preds, labels) -> torch.Tensor:
    """fila = clase pedida, columna = clase predicha."""
    idx = labels.cpu() * NUM_CLASSES + preds.cpu()
    return torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
