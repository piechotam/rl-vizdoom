"""Dueling Q-network for stacked 60x45 RGB frames and game-state features."""

from __future__ import annotations

import torch
from torch import nn

from src.preprocessing import FRAME_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH

FEATURE_HIDDEN = 64
AUX_HIDDEN = 128


class DuelingQNetwork(nn.Module):
    """CNN encoder + separate value and advantage streams (Wang et al. 2016).

    ``Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)``. Frames are uint8
    ``(B, stack * 3, 45, 60)``, scaled to ``[0, 1]`` inside the network.

    Optional extras:

    - ``feature_dim > 0``: a game-state vector goes through a small MLP and is
      concatenated with the CNN features before the V and A streams;
    - ``aux_dim > 0``: an auxiliary head predicts binary game facts (e.g. "an
      enemy is visible") from the CNN features only, so its loss shapes the
      visual encoder (Lample & Chaplot 2017).
    """

    def __init__(
        self,
        num_actions: int,
        stack_size: int = 4,
        hidden: int = 512,
        feature_dim: int = 0,
        aux_dim: int = 0,
    ):
        super().__init__()
        in_channels = stack_size * FRAME_CHANNELS
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, FRAME_HEIGHT, FRAME_WIDTH)
            n_conv = self.encoder(dummy).shape[1]
        self.features = (
            nn.Sequential(nn.Linear(feature_dim, FEATURE_HIDDEN), nn.ReLU())
            if feature_dim
            else None
        )
        n_features = n_conv + (FEATURE_HIDDEN if feature_dim else 0)
        self.value = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, num_actions)
        )
        self.aux = (
            nn.Sequential(
                nn.Linear(n_conv, AUX_HIDDEN), nn.ReLU(), nn.Linear(AUX_HIDDEN, aux_dim)
            )
            if aux_dim
            else None
        )

    def _q(self, conv: torch.Tensor, features: torch.Tensor | None) -> torch.Tensor:
        h = conv
        if self.features is not None:
            if features is None:
                raise ValueError("this network needs game-state features")
            h = torch.cat([conv, self.features(features)], dim=1)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def q_and_aux(
        self, x: torch.Tensor, features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Q-values and auxiliary logits (``None`` without an aux head)."""
        conv = self.encoder(x.float() / 255.0)
        aux = self.aux(conv) if self.aux is not None else None
        return self._q(conv, features), aux

    def forward(
        self, x: torch.Tensor, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._q(self.encoder(x.float() / 255.0), features)
