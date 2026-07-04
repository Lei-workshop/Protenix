# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114
import json
import os
import time
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from protenix.data.constants import STD_RESIDUES_WITH_GAP
from protenix.model.modules.primitives import LinearNoBias, Transition
from protenix.model.modules.transformer import AttentionPairBias
from protenix.model.modules.fused_ops import dropout_add_rowwise
from protenix.model.triangular.layers import DropoutRowwise, LayerNorm, OuterProductMean
from protenix.model.triangular.triangular import (
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from protenix.model.utils import (
    checkpoint_blocks,
    expand_at_dim,
    get_checkpoint_fn,
    pad_at_dim,
    permute_final_dims,
    sample_msa_feature_dict_random_without_replacement,
)
from protenix.utils.distributed import get_inference_parallel_context


def _pairformer_profile_path() -> Optional[str]:
    return os.environ.get("PAIRFORMER_PROFILE_LOG")


def _pairformer_sync_if_needed(enabled: bool, ref: torch.Tensor) -> None:
    if enabled and ref.is_cuda:
        torch.cuda.synchronize(ref.device)


def _pairformer_row_parallel_enabled(
    z: torch.Tensor,
    triangle_multiplicative: str,
    triangle_attention: str,
    training: bool,
) -> bool:
    if os.environ.get("PROTENIX_PAIRFORMER_ROW_PARALLEL", "0") != "1":
        return False
    min_n = int(os.environ.get("PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N", "768"))
    if min_n > 0 and z.shape[-3] < min_n:
        return False
    if training or torch.is_grad_enabled():
        return False
    if triangle_multiplicative != "torch" or triangle_attention != "wmma":
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return z.is_cuda and world_size > 1


def _pairformer_row_parallel_init(z: torch.Tensor) -> tuple[int, int, object]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if z.device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        except TypeError:
            dist.init_process_group("nccl")
    ctx = get_inference_parallel_context()
    return ctx.mp_rank, ctx.mp_world_size, ctx.mp_group


def _pairformer_row_bounds(n: int, world: int, rank: int) -> tuple[int, int, int]:
    rows = (n + world - 1) // world
    start = rank * rows
    end = min(start + rows, n)
    return start, end, rows


def _pairformer_pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[-3] == rows:
        return x.contiguous()
    out = x.new_zeros((*x.shape[:-3], rows, *x.shape[-2:]))
    if x.shape[-3] > 0:
        out[..., : x.shape[-3], :, :].copy_(x)
    return out


def _pairformer_gather_rows(
    local: torch.Tensor,
    n: int,
    world: int,
    group: object,
) -> torch.Tensor:
    flat = local.contiguous().view(-1)
    gathered = torch.empty((world * flat.numel(),), device=local.device, dtype=local.dtype)
    dist.all_gather_into_tensor(gathered, flat, group=group)
    chunks = gathered.view(world, *local.shape)
    return (
        chunks.permute(1, 0, 2, 3, 4)
        .reshape(local.shape[0], world * local.shape[1], *local.shape[2:])
        [:, :n]
        .contiguous()
    )


def _pairformer_trimul_project_a_b(module: nn.Module, z_norm: torch.Tensor, mask: torch.Tensor):
    mask = mask.unsqueeze(-1)
    a = mask * torch.sigmoid(module.linear_a_g(z_norm))
    a = a * module.linear_a_p(z_norm)
    b = mask * torch.sigmoid(module.linear_b_g(z_norm))
    b = b * module.linear_b_p(z_norm)
    return a, b


def _pairformer_trimul_update_rows(
    module: nn.Module,
    z: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    z_norm = module.layer_norm_in(z)
    a, b = _pairformer_trimul_project_a_b(module, z_norm, mask)
    if module._outgoing:
        x = torch.einsum("bikc,bkjc->bijc", a[:, start:end], b)
    else:
        x = torch.einsum("bkic,bkjc->bijc", a[:, :, start:end], b)
    x = module.layer_norm_out(x)
    x = module.linear_z(x)
    g = torch.sigmoid(module.linear_g(z_norm[:, start:end]))
    return x * g


def _pairformer_triangle_attention_shard(
    module: nn.Module,
    local_x: torch.Tensor,
    local_mask: torch.Tensor,
) -> torch.Tensor:
    x = module.layer_norm(local_x)
    mask_bias = (module.inf * (local_mask - 1))[..., :, None, None, :]
    triangle_bias = permute_final_dims(module.linear(x), (2, 0, 1)).unsqueeze(-4)

    # WMMA v9 wrapper currently expects Bias2 as [B, 1, H, S, S]. A row-shard
    # naturally produces [B, 1, H, rows, S], so pack rows at the front.
    seq = local_x.shape[-2]
    if triangle_bias.shape[-2] != seq:
        padded_bias = triangle_bias.new_zeros(
            *triangle_bias.shape[:-2], seq, triangle_bias.shape[-1]
        )
        padded_bias[..., : triangle_bias.shape[-2], :] = triangle_bias
        triangle_bias = padded_bias

    return module.mha(
        q_x=x,
        kv_x=x,
        biases=[mask_bias, triangle_bias],
        triangle_attention="wmma",
    )


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3

    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123

    Args:
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
            Defaults to 128.
        c_hidden_pair_att (int, optional): hidden dim [for TriangleAttention]. Defaults to 32.
        no_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        num_intermediate_factor (int, optional): number of intermediate factor for pair_transition. Defaults to 4.
        dropout (float, optional): dropout ratio [for TriangleUpdate]. Defaults to 0.25.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        if hidden_scale_up:
            no_heads_pair = c_z // c_hidden_pair_att
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)
        self.profile_block_index = -1

    def _forward_row_parallel_experimental(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Experimental 4-GPU row-parallel z-triangle path.

        This path intentionally covers the z-heavy triangle subgraph and the
        row-local pair transition. It all-gathers full z after each
        residual-producing module, then runs optional s updates replicated on
        every rank.
        """
        squeeze_batch = z.dim() == 3
        if squeeze_batch:
            z = z.unsqueeze(0)
            pair_mask = pair_mask.unsqueeze(0) if pair_mask is not None else None

        rank, world, group = _pairformer_row_parallel_init(z)
        start, end, rows = _pairformer_row_bounds(z.shape[-3], world, rank)
        n = z.shape[-3]
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        profile_log = _pairformer_profile_path()
        profile_enabled = profile_log is not None
        timings_ms: dict[str, float] = {}

        _pairformer_sync_if_needed(profile_enabled, z)
        t_last = time.perf_counter()

        def mark_profile(name: str, ref: torch.Tensor = z) -> None:
            nonlocal t_last
            if not profile_enabled:
                return
            _pairformer_sync_if_needed(True, ref)
            now = time.perf_counter()
            timings_ms[name] = timings_ms.get(name, 0.0) + (now - t_last) * 1000.0
            t_last = now

        def gather_profile(local: torch.Tensor, name: str) -> torch.Tensor:
            gathered = _pairformer_gather_rows(local, n, world, group)
            mark_profile(name, gathered)
            return gathered

        def write_profile(ref: torch.Tensor) -> None:
            if not profile_enabled:
                return
            _pairformer_sync_if_needed(True, ref)
            payload = {
                "path": "pairformer_block_row_parallel",
                "block_index": int(self.profile_block_index),
                "rank": int(rank),
                "world": int(world),
                "rows": int(rows),
                "shape_s": None if s is None else [int(dim) for dim in s.shape],
                "shape_z": [int(dim) for dim in z.shape],
                "dtype_s": None if s is None else str(s.dtype),
                "dtype_z": str(z.dtype),
                "device": str(z.device),
                "timings_ms": timings_ms,
                "total_ms": float(sum(timings_ms.values())),
                "triangle_attention": "wmma",
                "triangle_multiplicative": "torch",
            }
            profile_dir = os.path.dirname(profile_log)
            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
            with open(profile_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

        update = _pairformer_trimul_update_rows(
            self.tri_mul_out, z, pair_mask, start, end
        )
        mark_profile("tri_mul_out_compute", update)
        local = _pairformer_pad_rows(z[:, start:end] + update, rows)
        z = gather_profile(local, "tri_mul_out_gather")

        update = _pairformer_trimul_update_rows(
            self.tri_mul_in, z, pair_mask, start, end
        )
        mark_profile("tri_mul_in_compute", update)
        local = _pairformer_pad_rows(z[:, start:end] + update, rows)
        z = gather_profile(local, "tri_mul_in_gather")

        local_z = _pairformer_pad_rows(z[:, start:end], rows)
        local_mask = _pairformer_pad_rows(pair_mask[:, start:end].unsqueeze(-1), rows).squeeze(-1)
        local_update = _pairformer_triangle_attention_shard(
            self.tri_att_start, local_z, local_mask
        )
        mark_profile("tri_att_start_compute", local_update)
        z = gather_profile(local_z + local_update, "tri_att_start_gather")

        z = z.transpose(-2, -3).contiguous()
        mark_profile("transpose_after_start", z)
        mask_t = pair_mask.transpose(-1, -2).contiguous()
        local_z = _pairformer_pad_rows(z[:, start:end], rows)
        local_mask = _pairformer_pad_rows(mask_t[:, start:end].unsqueeze(-1), rows).squeeze(-1)
        local_update = _pairformer_triangle_attention_shard(
            self.tri_att_end, local_z, local_mask
        )
        mark_profile("tri_att_end_compute", local_update)
        z = gather_profile(local_z + local_update, "tri_att_end_gather")
        z = z.transpose(-2, -3).contiguous()
        mark_profile("transpose_after_end", z)

        local_z = z[:, start:end]
        local_update = self.pair_transition(local_z)
        mark_profile("pair_transition_compute", local_update)
        local = _pairformer_pad_rows(local_z + local_update, rows)
        z = gather_profile(local, "pair_transition_gather")

        if squeeze_batch:
            z = z.squeeze(0)
            pair_mask = pair_mask.squeeze(0)

        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z)
            mark_profile("attention_pair_bias_add", s)
            s = s + self.single_transition(s)
            mark_profile("single_transition_add", s)
        write_profile(z)
        return s, z

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": Cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        if _pairformer_row_parallel_enabled(
            z=z,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            training=self.training,
        ):
            return self._forward_row_parallel_experimental(s=s, z=z, pair_mask=pair_mask)

        profile_log = _pairformer_profile_path()
        profile_enabled = profile_log is not None
        timings_ms: dict[str, float] = {}

        _pairformer_sync_if_needed(profile_enabled, z)
        t_last = time.perf_counter()

        def mark_profile(name: str) -> None:
            nonlocal t_last
            if not profile_enabled:
                return
            _pairformer_sync_if_needed(True, z)
            now = time.perf_counter()
            timings_ms[name] = timings_ms.get(name, 0.0) + (now - t_last) * 1000.0
            t_last = now

        def write_profile() -> None:
            if not profile_enabled:
                return
            payload = {
                "path": "pairformer_block",
                "block_index": int(self.profile_block_index),
                "shape_s": None if s is None else [int(dim) for dim in s.shape],
                "shape_z": [int(dim) for dim in z.shape],
                "dtype_s": None if s is None else str(s.dtype),
                "dtype_z": str(z.dtype),
                "device": str(z.device),
                "triangle_multiplicative": triangle_multiplicative,
                "triangle_attention": triangle_attention,
                "inplace_safe": bool(inplace_safe),
                "chunk_size": None if chunk_size is None else int(chunk_size),
                "timings_ms": timings_ms,
                "total_ms": float(sum(timings_ms.values())),
            }
            profile_dir = os.path.dirname(profile_log)
            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
            with open(profile_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

        if inplace_safe:
            z = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            mark_profile("tri_mul_out")
            z = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            mark_profile("tri_mul_in")
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            mark_profile("tri_att_start_add")
            z = z.transpose(-2, -3).contiguous()
            mark_profile("transpose_after_start")
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            mark_profile("tri_att_end_add")
            z = z.transpose(-2, -3).contiguous()
            mark_profile("transpose_after_end")
            z += self.pair_transition(z)
            mark_profile("pair_transition_add")
        else:
            tmu_update = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            mark_profile("tri_mul_out")
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            mark_profile("dropout_add_tri_mul_out")
            del tmu_update
            tmu_update = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            mark_profile("tri_mul_in")
            z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
            mark_profile("dropout_add_tri_mul_in")
            del tmu_update
            tri_att_update = self.tri_att_start(
                z,
                mask=pair_mask,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            mark_profile("tri_att_start")
            z = dropout_add_rowwise(z, tri_att_update, self.p_drop, self.training)
            mark_profile("dropout_add_tri_att_start")
            del tri_att_update
            z = z.transpose(-2, -3).contiguous()
            mark_profile("transpose_after_start")
            tri_att_update = self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            mark_profile("tri_att_end")
            z = dropout_add_rowwise(z, tri_att_update, self.p_drop, self.training)
            mark_profile("dropout_add_tri_att_end")
            del tri_att_update
            z = z.transpose(-2, -3).contiguous()
            mark_profile("transpose_after_end")

            z = z + self.pair_transition(z)
            mark_profile("pair_transition_add")
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
            )
            mark_profile("attention_pair_bias_add")
            s = s + self.single_transition(s)
            mark_profile("single_transition_add")
        write_profile()
        return s, z


class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3

    Args:
        n_blocks (int, optional): number of blocks [for PairformerStack]. Defaults to 48.
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 4.
        dropout (float, optional): dropout ratio. Defaults to 0.25.
        blocks_per_ckpt (int, optional): number of Pairformer blocks in each activation checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for block_index in range(n_blocks):
            block = PairformerBlock(
                n_heads=n_heads,
                c_z=c_z,
                c_s=c_s,
                num_intermediate_factor=num_intermediate_factor,
                dropout=dropout,
                hidden_scale_up=hidden_scale_up,
            )
            block.profile_block_index = block_index
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the update of s and z
                [..., N_token, c_s]
                [..., N_token, N_token, c_z]
        """
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z


class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AF3

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c (int, optional): hidden dim [for MSAPairWeightedAveraging]. Defaults to 32.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        n_heads (int, optional): number of heads [for MSAPairWeightedAveraging]. Defaults to 8.
    """

    def __init__(
        self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8
    ) -> None:
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_m = c_m
        self.c = c
        self.n_heads = n_heads
        self.c_z = c_z
        # Input projections
        self.layernorm_m = LayerNorm(self.c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.n_heads
        )
        self.linear_no_bias_mg = LinearNoBias(
            in_features=self.c_m,
            out_features=self.c * self.n_heads,
            initializer="zeros",
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads,
            out_features=self.c_m,
            initializer="zeros",
        )

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        # Input projections
        m = self.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
        v = self.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
        v = v.reshape(
            *v.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        b = self.linear_no_bias_z(
            self.layernorm_z(z)
        )  # [...,n_token, n_token, n_heads]
        g = torch.sigmoid(
            self.linear_no_bias_mg(m)
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        g = g.reshape(
            *g.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        w = self.softmax_w(b)  # [...,n_token, n_token, n_heads]
        wv = torch.einsum(
            "...ijh,...mjhc->...mihc", w, v
        )  # [...,n_msa_sampled,n_token,n_heads,c]
        o = g * wv
        o = o.reshape(
            *o.shape[:-2], self.n_heads * self.c
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        m = self.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
        if (not self.training) and m.shape[-3] > 5120:
            del v, b, g, w, wv, o
        return m


class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c (int, optional): hidden [for MSAStack] dim. Defaults to 8.
        dropout (float, optional): dropout ratio. Defaults to 0.15.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c: int = 8,
        dropout: float = 0.15,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
    ) -> None:
        super(MSAStack, self).__init__()
        self.c = c
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            c_m=c_m, c=self.c, c_z=c_z
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]

        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        chunk_size = self.msa_chunk_size
        if self.training:
            # Padded m to avoid static graph change in DDP training, which will raise
            # RuntimeError: Your training graph has changed in this iteration,
            # e.g., one parameter is unused in first iteration, but then got used in the second iteration.
            # this is not compatible with static_graph set to True
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            msa_pair_weighted = self.chunk_forward(
                self.msa_pair_weighted_averaging, m_new, z, chunk_size
            )
            m = dropout_add_rowwise(m, msa_pair_weighted[: m.shape[-3], :, :], self.p_drop, self.training)
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            m_transition = self.chunk_forward(
                self.transition_m, m_new, None, chunk_size
            )
            m = m + m_transition[: m.shape[-3], :, :]
            if (not self.training) and (z.shape[-2] > 2000 or m.shape[-3] > 5120):
                del msa_pair_weighted, m_transition
        else:
            m = self.inference_forward(m, z, chunk_size)
        return m

    def chunk_forward(
        self,
        module: nn.Module,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_size (int): size of each chunk for gradient checkpointing

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """

        def fixed_length_chunk(m, chunk_length, dim=0):
            dim_size = m.size(dim)
            chunk_num = (dim_size + chunk_length - 1) // chunk_length
            chunks = []

            for i in range(chunk_num):
                start = i * chunk_length
                end = min(start + chunk_length, dim_size)
                chunk = m.narrow(dim, start, end - start)
                chunks.append(chunk)

            return chunks

        checkpoint_fn = get_checkpoint_fn()
        # Split the tensor `m` into chunks along the first dimension
        # m_chunks = torch.chunk(m, chunk_size, dim=0)
        m_chunks = fixed_length_chunk(m, chunk_size, dim=0)

        # Process each chunk with gradient checkpointing
        if z is not None:
            processed_chunks = [checkpoint_fn(module, chunk, z) for chunk in m_chunks]
        else:
            processed_chunks = [checkpoint_fn(module, chunk) for chunk in m_chunks]
        if (not self.training) and m.shape[-3] > 5120:
            del m_chunks
        # Concatenate the processed chunks back together
        m = torch.cat(processed_chunks, dim=0)
        if (not self.training) and m.shape[-3] > 5120:
            del processed_chunks
        return m

    def inference_forward(
        self, m: torch.Tensor, z: torch.Tensor, chunk_size: int = 2048
    ) -> torch.Tensor:
        """Inplace slice forward for saving memory
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_num (int): size of each chunk for gradient checkpointing

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """
        num_msa = m.shape[-3]
        no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
        for i in range(no_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_msa)
            # Use inplace to save memory
            m[start:end, :, :] += self.msa_pair_weighted_averaging(
                m[start:end, :, :], z
            )
            m[start:end, :, :] += self.transition_m(m[start:end, :, :])
        return m


class MSABlock(nn.Module):
    """
    Base MSA Block, Line6-Line13 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_hidden (int, optional): hidden dim [for MSABlock]. Defaults to 32.
        is_last_block (bool, optional): if this is the last block of MSAModule. Defaults to False.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSABlock, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block
        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=self.c_hidden
        )
        if not self.is_last_block:
            # MSA stack
            self.msa_stack = MSAStack(
                c_m=self.c_m,
                c_z=self.c_z,
                dropout=msa_dropout,
                msa_chunk_size=msa_chunk_size,
                msa_max_size=msa_max_size,
            )
        # Pair stack
        self.pair_stack = PairformerBlock(
            c_z=c_z, c_s=0, dropout=pair_dropout, hidden_scale_up=hidden_scale_up
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: updated m z of MSABlock
                [...,n_msa_sampled, n_token, c_m]
                [...,n_token, n_token, c_z]
        """
        # Communication
        z = z + self.outer_product_mean_msa(
            m, inplace_safe=inplace_safe, chunk_size=chunk_size
        )
        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z)
        # Pair stack
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if not self.is_last_block:
            return m, z
        else:
            return None, z  # to ensure that `m` will not be used.


class MSAModule(nn.Module):
    """
    Implements Algorithm 8 [MSAModule] in AF3

    Args:
        n_blocks (int, optional): number of blocks [for MSAModule]. Defaults to 4.
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional):
            hidden dim for single embedding from InputFeatureEmbedder. Defaults to 449.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        blocks_per_ckpt: number of MSAModule blocks in each activation checkpoint. Defaults to 1.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
        msa_configs (dict, optional): a dictionary containing keys: "enable", "strategy", etc. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = 1,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        msa_configs: Optional[dict[str, Any]] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSAModule, self).__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.blocks_per_ckpt = blocks_per_ckpt
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size
        self.input_feature = {
            "msa": 32,
            "has_deletion": 1,
            "deletion_value": 1,
        }

        self.msa_configs = {
            "enable": msa_configs.get("enable", False),
            "strategy": msa_configs.get("strategy", "random"),
        }
        if "sample_cutoff" in msa_configs:
            self.msa_configs["train_cutoff"] = msa_configs["sample_cutoff"].get(
                "train", 512
            )
            self.msa_configs["test_cutoff"] = msa_configs["sample_cutoff"].get(
                "test", 16384
            )
            # the default msa_max_size is 16384 if not specified
            self.msa_max_size = self.msa_configs["train_cutoff"]
        if "min_size" in msa_configs:
            self.msa_configs["train_lowerb"] = msa_configs["min_size"].get("train", 1)
            self.msa_configs["test_lowerb"] = msa_configs["min_size"].get("test", 1)

        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            block = MSABlock(
                c_m=self.c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                msa_chunk_size=self.msa_chunk_size,
                msa_max_size=self.msa_max_size,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]
        return blocks

    def one_hot_fp32(
        self, tensor: torch.Tensor, num_classes: int, dtype=torch.float32
    ) -> torch.Tensor:
        """like F.one_hot, but output dtype is float32.

        Args:
            tensor (torch.Tensor): the input tensor
            num_classes (int): num_classes
            dtype (torch.float32, optional): the output dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: the one-hot encoded tensor with shape
                [..., n_msa_sampled, N_token, num_classes]
        """
        shape = tensor.shape
        one_hot_tensor = torch.zeros(
            *shape, num_classes, dtype=dtype, device=tensor.device
        )
        one_hot_tensor.scatter_(len(shape), tensor.unsqueeze(-1), 1)
        return one_hot_tensor

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]):
                input meta feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_token, c_s_inputs]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated z
                [..., N_token, N_token, c_z]
        """
        # If n_blocks < 1, return z
        if self.n_blocks < 1:
            return z

        if "msa" not in input_feature_dict:
            return z
        # Check msa shape!
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
        if input_feature_dict["msa"].dim() < 2:
            return z
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature_dict,
            dim_dict={feat_name: -2 for feat_name in self.input_feature},
            cutoff=(
                self.msa_configs["train_cutoff"]
                if self.training
                else self.msa_configs["test_cutoff"]
            ),
            lower_bound=(
                self.msa_configs["train_lowerb"]
                if self.training
                else self.msa_configs["test_lowerb"]
            ),
            strategy=self.msa_configs["strategy"],
        )
        # pylint: disable=E1102
        if not self.training and z.shape[-2] > 2000:
            # msa_feat["msa"] is torch.int64, we convert it
            # to torch.float32 for saving half of the CUDA memory
            msa_feat["msa"] = self.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )

        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [
                msa_feat[name].reshape(*target_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )  # [..., N_msa_sample, N_token, 32 + 1 + 1]
        # Msa_feat is very large, if N_MSA=16384 and N_token=4000,
        # msa_feat["msa"] consumes about 16G CUDA memory, so we
        # need to clear cache to avoid OOM
        if not self.training:
            del msa_feat
        # Line2
        msa_sample = self.linear_no_bias_m(msa_sample)

        # Auto broadcast [...,n_msa_sampled, n_token, c_m]
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        msa_sample, z = checkpoint_blocks(
            blocks,
            args=(msa_sample, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return z


class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3

    Args:
        n_blocks (int, optional): number of blocks for TemplateEmbedder. Defaults to 2.
        c (int, optional): hidden dim of TemplateEmbedder. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 2.
        dropout (float, optional): dropout ratio for PairformerStack. Defaults to 0.25.
            Note this value is missed in Algorithm 16, so we use default ratio for Pairformer
        blocks_per_ckpt (int, optional): number of TemplateEmbedder/Pairformer blocks in each activation
            checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        num_intermediate_factor: int = 2,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(TemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            num_intermediate_factor=num_intermediate_factor,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.relu = nn.ReLU()
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: torch.Tensor = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]): input feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor, optional): pair masking. Default to None.
                [..., N_token, N_token]
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel

        Returns:
            torch.Tensor: the template feature
                [..., N_token, N_token, c_z]
        """
        # Do not use TemplateEmbedder by setting n_blocks=0
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            # Compatible with the Protenix 0.5.0 model series
            return 0
        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)

        num_residues = z.shape[0]
        # determine whether the number of templates is the configured maximum value, otherwise error out
        num_templates = input_feature_dict["template_aatype"].shape[0]
        query_num_channels = z.shape[-1]

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        z = self.layernorm_z(z)
        u = 0
        for template_id in range(num_templates):
            u = u + self.single_template_forward(
                template_id=template_id,
                input_feature_dict=input_feature_dict,
                z=z,
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
                triangle_attention=triangle_attention,
                triangle_multiplicative=triangle_multiplicative,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(self.relu(u))
        assert u.shape == (num_residues, num_residues, query_num_channels)
        return u

    def single_template_forward(
        self,
        template_id: int,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        multichain_mask: Optional[torch.Tensor] = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        to_concat = []

        dgram = input_feature_dict["template_distogram"][
            template_id
        ]  # [N_token, N_token, 39]
        pseudo_beta_mask_2d = input_feature_dict["template_pseudo_beta_mask"][
            template_id
        ]
        dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask_2d * multichain_mask * pair_mask
        )  # [N_token, N_token]
        to_concat.append(dgram)
        to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))

        aatype = input_feature_dict["template_aatype"][template_id]  # [N_token]
        aatype = F.one_hot(aatype, num_classes=len(STD_RESIDUES_WITH_GAP))
        to_concat.append(expand_at_dim(aatype, dim=-3, n=z.shape[0]))
        to_concat.append(expand_at_dim(aatype, dim=-2, n=z.shape[0]))

        unit_vector = input_feature_dict["template_unit_vector"][template_id]
        unit_vector = (
            unit_vector * multichain_mask[..., None] * pair_mask[..., None]
        )  # [N_token, N_token, 3]
        to_concat.append(unit_vector)

        backbone_mask_2d = input_feature_dict["template_backbone_frame_mask"][
            template_id
        ]
        backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
        to_concat.append(backbone_mask_2d.unsqueeze(-1))

        at = torch.concat(to_concat, dim=-1)
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at)
        _, v = self.pairformer_stack(
            s=None,
            z=v,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        v = self.layernorm_v(v)
        return v
