"""Grouped matmul kernel that mirrors :func:`torch._grouped_mm`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

from .config import DEFAULT_GROUPED_MM_CONFIG, GroupedMMConfig


@dataclass
class GroupedRowTiling:
    """Precomputed metadata describing how rows are partitioned per expert."""

    offsets: Tensor
    group_rows: Tensor
    group_starts: Tensor
    tile_group_index: Tensor
    tile_row_start: Tensor

    def detach(self) -> "GroupedRowTiling":
        return GroupedRowTiling(
            offsets=self.offsets.detach(),
            group_rows=self.group_rows.detach(),
            group_starts=self.group_starts.detach(),
            tile_group_index=self.tile_group_index.detach(),
            tile_row_start=self.tile_row_start.detach(),
        )

    @property
    def total_rows(self) -> int:
        if self.offsets.numel() == 0:
            return 0
        return int(self.offsets[-1].item())


def _normalize_offsets(
    offs: Tensor,
    *,
    num_groups: int,
    total_rows: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(offs, Tensor):
        raise TypeError("offs must be a torch.Tensor.")
    if offs.ndim != 1:
        raise ValueError("offs must be a 1D tensor.")
    if offs.numel() != num_groups:
        raise ValueError(
            f"Expected offs to have {num_groups} elements but got {offs.numel()}."
        )
    if offs.dtype not in (torch.int32, torch.int64):
        raise TypeError("offs must use torch.int32 or torch.int64 dtype.")

    normalized = offs.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    if normalized.numel() == 0:
        raise ValueError("offs must contain at least one group.")
    if normalized[0].item() < 0:
        raise ValueError("offs entries must be non-negative.")
    if (normalized[1:] < normalized[:-1]).any():
        raise ValueError("offs must be non-decreasing.")
    # if normalized[-1].item() != total_rows:
    #     raise ValueError(
    #         f"offs last value {normalized[-1].item()} must equal the total rows {total_rows}."
    #     )
    if normalized[-1].item() > torch.iinfo(torch.int32).max:
        raise ValueError("Grouped matmul offsets exceed int32 range.")
    return normalized.to(dtype=torch.int32)


def _build_group_rows(offsets: Tensor) -> Tuple[Tensor, Tensor]:
    group_rows = torch.empty_like(offsets)
    group_rows[0] = offsets[0]
    if offsets.numel() > 1:
        group_rows[1:] = offsets[1:] - offsets[:-1]
    group_starts = torch.zeros_like(offsets)
    if offsets.numel() > 1:
        group_starts[1:] = offsets[:-1]
    return group_rows, group_starts


def build_grouped_mm_tiling(offsets: Tensor, block_m: int) -> GroupedRowTiling:
    if block_m <= 0:
        raise ValueError("block_m must be a positive integer.")
    group_rows, group_starts = _build_group_rows(offsets)
    rows_cpu = group_rows.cpu().tolist()

    tile_groups: list[int] = []
    tile_row_starts: list[int] = []
    for idx, rows in enumerate(rows_cpu):
        rows_int = int(rows)
        for row_start in range(0, rows_int, block_m):
            tile_groups.append(idx)
            tile_row_starts.append(row_start)

    device = offsets.device
    if tile_groups:
        tile_group_tensor = torch.tensor(tile_groups, dtype=torch.int32, device=device)
        tile_row_tensor = torch.tensor(tile_row_starts, dtype=torch.int32, device=device)
    else:
        tile_group_tensor = torch.empty(0, dtype=torch.int32, device=device)
        tile_row_tensor = torch.empty(0, dtype=torch.int32, device=device)

    return GroupedRowTiling(
        offsets=offsets,
        group_rows=group_rows,
        group_starts=group_starts,
        tile_group_index=tile_group_tensor,
        tile_row_start=tile_row_tensor,
    )


def prepare_grouped_mm_tiling(
    offs: Tensor,
    *,
    num_groups: int,
    total_rows: int,
    block_m: int,
    device: torch.device,
) -> Tuple[Tensor, GroupedRowTiling]:
    offsets = _normalize_offsets(
        offs,
        num_groups=num_groups,
        total_rows=total_rows,
        device=device,
    )
    tiling = build_grouped_mm_tiling(offsets, block_m)
    return offsets, tiling


@triton.jit
def _grouped_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    tile_group_ptr,
    tile_row_start_ptr,
    group_rows_ptr,
    group_starts_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    num_row_tiles,
    num_n_tiles,
    num_tiles,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """
    Persistent, cache-aware grouped GEMM.

    • Launch grid = (NUM_PROGRAMS,) where NUM_PROGRAMS ≈ #SMs.
    • Each program loops over tile_ids: for tile_id in range(start_pid, num_tiles, NUM_SMS)
    • Tiles enumerated in column-major order = better L2 reuse of B.
    """
    start_pid = tl.program_id(axis=0)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m = tile_id % num_row_tiles
        tile_n = tile_id // num_row_tiles

        group_idx = tl.load(tile_group_ptr + tile_m)
        tile_row_start = tl.load(tile_row_start_ptr + tile_m)
        group_rows = tl.load(group_rows_ptr + group_idx)
        group_start = tl.load(group_starts_ptr + group_idx)

        row_start = group_start + tile_row_start
        rows_remaining = group_rows - tile_row_start
        rows_in_tile = tl.minimum(rows_remaining, BLOCK_M)
        rows_in_tile = tl.maximum(rows_in_tile, 0)

        offs_m = row_start + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        row_mask = tl.arange(0, BLOCK_M) < rows_in_tile
        col_mask = offs_n < N

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_group_ptr = b_ptr + group_idx * stride_bg
        b_ptrs = b_group_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_idx in range(0, num_k_tiles):
            k_indices = k_idx * BLOCK_K + offs_k
            k_mask = k_indices < K

            a_mask = row_mask[:, None] & k_mask[None, :]
            b_mask = k_mask[:, None] & col_mask[None, :]

            a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a_block, b_block)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(c_ptrs, acc.to(OUTPUT_DTYPE), mask=c_mask)


@triton.jit
def _grouped_mm_grad_weight_kernel(
    a_ptr,
    grad_out_ptr,
    grad_weight_ptr,
    group_rows_ptr,
    group_starts_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gom,
    stride_gon,
    stride_wg,
    stride_wk,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    group_idx = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    group_rows = tl.load(group_rows_ptr + group_idx)
    group_start = tl.load(group_starts_ptr + group_idx)
    tokens_end = group_start + group_rows

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_reduce = tl.cdiv(group_rows, BLOCK_K)
    for k_idx in range(0, num_reduce):
        token_offsets = group_start + k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        token_mask = token_offsets < tokens_end

        a_ptrs = a_ptr + token_offsets[None, :] * stride_am + offs_m[:, None] * stride_ak
        b_ptrs = grad_out_ptr + token_offsets[:, None] * stride_gom + offs_n[None, :] * stride_gon

        a_mask = token_mask[None, :] & (offs_m[:, None] < K)
        b_mask = token_mask[:, None] & (offs_n[None, :] < N)

        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a_block, b_block)

    grad_w_ptrs = (
        grad_weight_ptr
        + group_idx * stride_wg
        + offs_m[:, None] * stride_wk
        + offs_n[None, :] * stride_wn
    )
    mask = (offs_m[:, None] < K) & (offs_n[None, :] < N)
    tl.store(grad_w_ptrs, accumulator.to(OUTPUT_DTYPE), mask=mask)


def _run_grouped_mm_forward(
    a: Tensor,
    b: Tensor,
    *,
    tiling: GroupedRowTiling,
    config: GroupedMMConfig,
    out: Optional[Tensor],
) -> Tensor:
    m_total = a.shape[0]
    n = b.shape[-1]
    k = a.shape[-1]

    if out is None:
        out = torch.empty((m_total, n), device=a.device, dtype=a.dtype)

    if m_total == 0 or n == 0 or k == 0:
        out.zero_()
        return out

    num_row_tiles = tiling.tile_group_index.numel()
    if num_row_tiles == 0:
        out.zero_()
        return out

    num_n_tiles = triton.cdiv(n, config.block_n)
    num_tiles = num_row_tiles * num_n_tiles
    if num_tiles == 0:
        out.zero_()
        return out

    # Persistent launch: one program per SM (or per tile, whichever is smaller)
    props = torch.cuda.get_device_properties(a.device)
    num_sms = props.multi_processor_count
    num_programs = min(num_sms, num_tiles)

    grid = (num_programs,)

    _grouped_mm_kernel[grid](
        a,
        b,
        out,
        tiling.tile_group_index,
        tiling.tile_row_start,
        tiling.group_rows,
        tiling.group_starts,
        m_total,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        num_row_tiles,
        num_n_tiles,
        num_tiles,
        NUM_SMS=num_programs,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        OUTPUT_DTYPE=tl.bfloat16,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out


def _run_grouped_mm_grad_weight(
    a: Tensor,
    grad_out: Tensor,
    *,
    tiling: GroupedRowTiling,
    config: GroupedMMConfig,
    out_dtype: torch.dtype,
) -> Tensor:
    num_groups = tiling.group_rows.shape[0]
    k = a.shape[1]
    n = grad_out.shape[1]
    grad_weight = torch.empty((num_groups, k, n), device=a.device, dtype=out_dtype)

    if k == 0 or n == 0:
        grad_weight.zero_()
        return grad_weight

    grid = (
        num_groups,
        triton.cdiv(k, config.block_m),
        triton.cdiv(n, config.block_n),
    )
    if grid[1] == 0 or grid[2] == 0:
        grad_weight.zero_()
        return grad_weight

    _grouped_mm_grad_weight_kernel[grid](
        a,
        grad_out,
        grad_weight,
        tiling.group_rows,
        tiling.group_starts,
        n,
        k,
        a.stride(0),
        a.stride(1),
        grad_out.stride(0),
        grad_out.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
        grad_weight.stride(2),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        OUTPUT_DTYPE=tl.bfloat16,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return grad_weight


def launch_grouped_mm(
    a: Tensor,
    b: Tensor,
    *,
    offs: Tensor,
    config: GroupedMMConfig = DEFAULT_GROUPED_MM_CONFIG,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Launch the Triton grouped matmul kernel on CUDA tensors."""

    if a.ndim != 2:
        raise ValueError("Input must be 2D (total_tokens, in_features).")
    if b.ndim != 3:
        raise ValueError("Weight must be (num_groups, in_features, out_features).")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Grouped matmul expects CUDA tensors.")
    if a.device != b.device:
        raise ValueError("Input and weight tensors must be on same device.")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("Grouped matmul only supports bfloat16.")

    num_groups, in_features, out_features = b.shape
    total_rows, input_dim = a.shape
    if input_dim != in_features:
        raise ValueError(
            f"Incompatible dims: input features {input_dim} vs weight features {in_features}."
        )

    if out is not None:
        if not out.is_cuda or out.device != a.device:
            raise ValueError("out must be CUDA tensor on same device.")
        if out.shape != (total_rows, out_features):
            raise ValueError(
                f"out must have shape ({total_rows}, {out_features}), got {tuple(out.shape)}."
            )
        if out.dtype != a.dtype:
            raise ValueError("out dtype must match input dtype.")

    _, tiling = prepare_grouped_mm_tiling(
        offs,
        num_groups=num_groups,
        total_rows=total_rows,
        block_m=config.block_m,
        device=a.device,
    )

    out = _run_grouped_mm_forward(a, b, tiling=tiling, config=config, out=out)
    return out


def grouped_mm(
    a: Tensor,
    b: Tensor,
    *,
    offs: Tensor,
    config: GroupedMMConfig = DEFAULT_GROUPED_MM_CONFIG,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Functional alias for :func:`launch_grouped_mm`."""

    return launch_grouped_mm(a, b, offs=offs, config=config, out=out)


__all__ = [
    "launch_grouped_mm",
    "grouped_mm",
    "GroupedRowTiling",
    "build_grouped_mm_tiling",
    "prepare_grouped_mm_tiling",
]

