"""Small identity-initialized temporal conditioning modules."""

from __future__ import annotations

import math

import torch
from torch import nn


def _validate_discrete(values: torch.Tensor, maximum: int, name: str) -> torch.Tensor:
    values = values.to(dtype=torch.long)
    if values.numel() and (values.min() < 0 or values.max() > maximum):
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return values


class _FiLMHead(nn.Module):
    def __init__(self, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(embedding).chunk(2, dim=-1)
        return gamma, beta


class FeatureAgeConditioner(nn.Module):
    """Shared age table with independent FiLM heads for SAM2 feature scales."""

    CHANNELS = {"s0": 32, "s1": 64, "deep": 256, "post": 256}

    def __init__(self, max_age: int = 5, embedding_dim: int = 64) -> None:
        super().__init__()
        self.max_age = max_age
        self.embedding = nn.Embedding(max_age + 1, embedding_dim)
        self.heads = nn.ModuleDict(
            {
                name: _FiLMHead(embedding_dim, channels)
                for name, channels in self.CHANNELS.items()
            }
        )

    def age_embedding(self, age: torch.Tensor) -> torch.Tensor:
        return self.embedding(_validate_discrete(age, self.max_age, "feature age"))

    def condition_bchw(
        self,
        name: str,
        feature: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.CHANNELS[name]:
            raise ValueError(f"unexpected {name} BCHW shape: {tuple(feature.shape)}")
        checked_age = _validate_discrete(age, self.max_age, "feature age")
        gamma, beta = self.heads[name](self.embedding(checked_age))
        active = checked_age.gt(0).to(feature.dtype).view(-1, 1, 1, 1)
        return feature * (1 + active * gamma[:, :, None, None]) + active * beta[
            :, :, None, None
        ]

    def condition_seq(
        self,
        name: str,
        feature: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 3 or feature.shape[-1] != self.CHANNELS[name]:
            raise ValueError(f"unexpected {name} sequence shape: {tuple(feature.shape)}")
        checked_age = _validate_discrete(age, self.max_age, "feature age")
        if feature.shape[1] != checked_age.numel():
            raise ValueError("sequence batch and age lengths differ")
        gamma, beta = self.heads[name](self.embedding(checked_age))
        active = checked_age.gt(0).to(feature.dtype).view(1, -1, 1)
        return feature * (1 + active * gamma.unsqueeze(0)) + active * beta.unsqueeze(0)


class SinusoidalDeltaEncoding(nn.Module):
    def __init__(self, dim: int = 64, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("sinusoidal dimension must be even")
        frequencies = torch.exp(
            -math.log(base) * torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.numel() and delta.min() < 0:
            raise ValueError("memory recency must be non-negative")
        phase = delta.float().unsqueeze(-1) * self.frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class _TemporalResidualHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TwoClockMemoryConditioner(nn.Module):
    """Recency/freshness residuals with one, learnable zero-output boundary."""

    def __init__(
        self,
        *,
        max_freshness: int = 5,
        embedding_dim: int = 64,
        spatial_dim: int = 64,
        pointer_dim: int = 256,
        use_freshness: bool = True,
    ) -> None:
        super().__init__()
        self.max_freshness = max_freshness
        self.use_freshness = use_freshness
        self.recency_encoding = SinusoidalDeltaEncoding(embedding_dim)
        self.freshness_embedding = nn.Embedding(max_freshness + 1, embedding_dim)
        self.spatial_recency = _TemporalResidualHead(
            embedding_dim, embedding_dim, spatial_dim
        )
        self.pointer_recency = _TemporalResidualHead(
            embedding_dim, embedding_dim, pointer_dim
        )
        if use_freshness:
            self.spatial_freshness = _TemporalResidualHead(
                embedding_dim, embedding_dim, spatial_dim
            )
            self.pointer_freshness = _TemporalResidualHead(
                embedding_dim, embedding_dim, pointer_dim
            )

    def _encoded(
        self, recency: torch.Tensor, freshness: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        recency_encoded = self.recency_encoding(recency)
        if not self.use_freshness:
            return recency_encoded, None
        freshness = _validate_discrete(
            freshness, self.max_freshness, "memory freshness"
        )
        return recency_encoded, self.freshness_embedding(freshness)

    def spatial_delta(
        self, recency: torch.Tensor, freshness: torch.Tensor
    ) -> torch.Tensor:
        recency_encoded, freshness_encoded = self._encoded(recency, freshness)
        delta = self.spatial_recency(recency_encoded)
        if freshness_encoded is not None:
            delta = delta + self.spatial_freshness(freshness_encoded)
        return delta

    def pointer_delta(
        self, recency: torch.Tensor, freshness: torch.Tensor
    ) -> torch.Tensor:
        recency_encoded, freshness_encoded = self._encoded(recency, freshness)
        delta = self.pointer_recency(recency_encoded)
        if freshness_encoded is not None:
            delta = delta + self.pointer_freshness(freshness_encoded)
        return delta
