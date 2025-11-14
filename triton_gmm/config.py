"""Shared configuration dataclasses for tuning kernel launch parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatmulConfig:
    """Launch parameters for a Triton matmul kernel."""

    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2


@dataclass(frozen=True)
class GroupedMMConfig:
    """Launch parameters for the grouped matmul kernel."""

    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 2


DEFAULT_MATMUL_CONFIG = MatmulConfig()
DEFAULT_GROUPED_MM_CONFIG = GroupedMMConfig()

__all__ = [
    "MatmulConfig",
    "DEFAULT_MATMUL_CONFIG",
    "GroupedMMConfig",
    "DEFAULT_GROUPED_MM_CONFIG",
]

