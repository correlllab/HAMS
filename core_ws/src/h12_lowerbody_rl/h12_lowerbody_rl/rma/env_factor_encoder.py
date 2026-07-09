from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    if activation == "elu":
        act: type[nn.Module] = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class EnvFactorEncoderCfg:
    in_dim: int = 21  # e_t: 15 upper-body DoF + 3 left hand + 3 right hand (no torso)
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"


class EnvFactorEncoder(nn.Module):
    """Encodes privileged environment factors e_t into a compact latent z_t."""

    def __init__(self, cfg: EnvFactorEncoderCfg | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvFactorEncoderCfg()
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.latent_dim,
            activation=self.cfg.activation,
        )

    def forward(self, env_factors: torch.Tensor) -> torch.Tensor:
        """Args:
        env_factors: Tensor of shape (N, in_dim)
        Returns:
        z: Tensor of shape (N, latent_dim)
        """
        if env_factors.ndim != 2 or env_factors.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"env_factors must be (N, {self.cfg.in_dim}); got {tuple(env_factors.shape)}"
            )
        return self.net(env_factors)
