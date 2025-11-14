"""Autograd-friendly drop-in replacement for :func:`torch._grouped_mm`."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function

from .config import DEFAULT_GROUPED_MM_CONFIG, GroupedMMConfig
from .kernel import (
    GroupedRowTiling,
    prepare_grouped_mm_tiling,
    _run_grouped_mm_forward,
    _run_grouped_mm_grad_weight,
)


class TritonGroupedMM(Function):
    """Custom autograd Function that dispatches to the Triton grouped matmul kernel."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: Tensor,
        weight: Tensor,
        offs: Tensor,
        config: GroupedMMConfig = DEFAULT_GROUPED_MM_CONFIG,
    ) -> Tensor:
        if input.ndim != 2:
            raise ValueError("input must be rank-2 with shape (total_tokens, in_features).")
        if weight.ndim != 3:
            raise ValueError("weight must have shape (num_groups, in_features, out_features).")
        if not input.is_cuda or not weight.is_cuda:
            raise ValueError("TritonGroupedMM expects CUDA tensors for input and weight.")
        if input.device != weight.device:
            raise ValueError("input and weight tensors must live on the same device.")
        if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError("TritonGroupedMM currently supports only torch.bfloat16 tensors.")

        num_groups, in_features, _ = weight.shape
        total_rows, input_dim = input.shape
        if input_dim != in_features:
            raise ValueError(
                f"input features {input_dim} must match weight features {in_features}."
            )

        mm_config = config or DEFAULT_GROUPED_MM_CONFIG
        _, tiling = prepare_grouped_mm_tiling(
            offs,
            num_groups=num_groups,
            total_rows=total_rows,
            block_m=mm_config.block_m,
            device=input.device,
        )
        output = _run_grouped_mm_forward(
            input,
            weight,
            tiling=tiling,
            config=mm_config,
            out=None,
        )

        ctx.save_for_backward(input, weight)
        ctx.tiling = tiling.detach()
        ctx.config = mm_config
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        input, weight = ctx.saved_tensors
        tiling: GroupedRowTiling = ctx.tiling
        config: GroupedMMConfig = ctx.config

        grad_input: Optional[Tensor]
        grad_weight: Optional[Tensor]

        grad_output_bf16 = grad_output.to(torch.bfloat16)

        if ctx.needs_input_grad[0]:
            grad_input_bf16 = _run_grouped_mm_forward(
                grad_output_bf16,
                weight.transpose(-2, -1),
                tiling=tiling,
                config=config,
                out=None,
            )
            grad_input = grad_input_bf16.to(ctx.input_dtype)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            grad_weight_bf16 = _run_grouped_mm_grad_weight(
                input,
                grad_output_bf16,
                tiling=tiling,
                config=config,
                out_dtype=torch.bfloat16,
            )
            grad_weight = grad_weight_bf16.to(ctx.weight_dtype)
        else:
            grad_weight = None

        return grad_input, grad_weight, None, None


def grouped_mm(
    input: Tensor,
    weight: Tensor,
    *,
    offs: Tensor,
    config: GroupedMMConfig = DEFAULT_GROUPED_MM_CONFIG,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Drop-in replacement for :func:`torch._grouped_mm` with Triton acceleration."""

    result = TritonGroupedMM.apply(input, weight, offs, config)
    if out is not None:
        if not isinstance(out, Tensor):
            raise TypeError("out must be a torch.Tensor when provided.")
        if out.shape != result.shape:
            raise ValueError(f"out tensor must have shape {tuple(result.shape)}.")
        if out.device != result.device:
            raise ValueError("out tensor must be on the same device as the result.")
        if out.dtype != result.dtype:
            raise ValueError("out tensor dtype must match the result dtype.")
        out.copy_(result)
        return out
    return result


__all__ = ["TritonGroupedMM", "grouped_mm"]

