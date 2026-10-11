"""Métricas de la Task 4: nitidez horizontal y estimación de pass@k."""

import torch


def horizontal_sharpness(images: torch.Tensor) -> torch.Tensor:
    """Media de |x[..., j+1] - x[..., j]|; las imágenes deben estar en [0,1]."""
    return (images[..., 1:] - images[..., :-1]).abs().mean()


def pass_at_k(n: int, c: int, k: int) -> float:
    """1 - C(n-c,k)/C(n,k), calculado sin factoriales con tensores float64."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Se requiere 0 <= c <= n y 1 <= k <= n.")
    if n - c < k:
        return 1.0
    i = torch.arange(k, dtype=torch.float64)
    return (1 - ((n - c - i) / (n - i)).prod()).item()
