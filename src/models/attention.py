"""Atención aditiva sobre los estados ocultos del encoder (Etapa B).

Los pesos que devuelve esta capa no son un detalle interno: son la explicación
que consume el MVP para señalar qué transacción del historial pesó más en una
alerta. Por eso el enmascarado ocurre **antes** del softmax. Si se aplicara
después, los pasos de padding recibirían probabilidad y el mapa de calor
señalaría transacciones que no existen.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdditiveAttention(nn.Module):
    """Atención aditiva (Bahdanau) de una sola cabeza sobre `[B, T, H]`.

    Se elige la variante aditiva sobre la de producto punto porque aprende su
    propia proyección antes de puntuar: con `hidden_size=64` y secuencias
    cortas, es más estable y sus pesos resultan más interpretables, que es el
    objetivo aquí.
    """

    def __init__(self, hidden_size: int, attention_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.projection = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Resume `hidden_states [B, T, H]` usando solo los pasos reales de `mask [B, T]`.

        Devuelve `(context, weights)`:
        - `context [B, H]`: suma ponderada de los estados por timestep. Es lo
          único que ve la cabeza clasificadora, de modo que la predicción es
          explicable por completo a partir de `weights`.
        - `weights [B, T]`: suman 1 sobre los pasos reales y valen exactamente
          0 en el padding.

        Se rellena con `finfo.min` en vez de `-inf` para que una secuencia
        completamente enmascarada degrade a pesos uniformes en lugar de
        producir NaN. No debería ocurrir (el mínimo del dataset es 3
        transacciones), pero un NaN aquí envenenaría el gradiente entero.
        """
        projected = torch.tanh(self.projection(hidden_states))  # [B, T, A]
        scores = self.score(projected).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)  # [B, T]
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # [B, H]
        return context, weights
