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

import json
import os
import time
from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    broadcast_token_to_local_atom_pair,
    DropPath,
    LinearNoBias,
    rearrange_qk_to_dense_trunk,
)
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    checkpoint_blocks,
    permute_final_dims,
)
from protenix.utils.distributed import get_inference_parallel_context


def _diffusion_transformer_profile_path() -> Optional[str]:
    return os.environ.get("DIFFUSION_TRANSFORMER_PROFILE_LOG")


def _diffusion_transformer_sync_if_needed(enabled: bool, ref: torch.Tensor) -> None:
    if enabled and ref.is_cuda:
        torch.cuda.synchronize(ref.device)


def _diffusion_ulysses_enabled() -> bool:
    return os.environ.get("PROTENIX_DIFFUSION_ULYSSES_SP", "0") == "1"


def _diffusion_ulysses_init_if_needed(ref: torch.Tensor) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ref.device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        except TypeError:
            dist.init_process_group("nccl")
    return True


def _sp_row_bounds(n: int, world: int, rank: int) -> tuple[int, int, int]:
    rows = (n + world - 1) // world
    start = rank * rows
    end = min(start + rows, n)
    return start, end, rows


def _sp_pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    if x.shape[-2] == rows:
        return x.contiguous()
    out = x.new_zeros(*x.shape[:-2], rows, x.shape[-1])
    if x.shape[-2] > 0:
        out[..., : x.shape[-2], :] = x
    return out


def _sp_gather_rows(
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
        chunks.permute(1, 0, 2, 3)
        .reshape(local.shape[0], world * local.shape[1], local.shape[2])
        [:, :n]
        .contiguous()
    )


def _sp_seq2head(x: torch.Tensor, world: int, group: object) -> torch.Tensor:
    bsz, local_seq, heads, head_dim = x.shape
    shard_heads = heads // world
    send = x.reshape(bsz, local_seq, world, shard_heads, head_dim)
    send = send.permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.permute(1, 0, 2, 3, 4).reshape(
        bsz, world * local_seq, shard_heads, head_dim
    ).contiguous()


def _sp_seq2head_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world: int,
    group: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, local_seq, heads, head_dim = q.shape
    shard_heads = heads // world
    packed = torch.stack((q, k, v), dim=0)
    send = packed.reshape(3, bsz, local_seq, world, shard_heads, head_dim)
    send = send.permute(3, 0, 1, 2, 4, 5).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    packed = recv.permute(1, 2, 0, 3, 4, 5).reshape(
        3, bsz, world * local_seq, shard_heads, head_dim
    )
    return tuple(t.contiguous() for t in packed.unbind(dim=0))


def _sp_head2seq(x: torch.Tensor, world: int, group: object) -> torch.Tensor:
    bsz, full_seq, shard_heads, head_dim = x.shape
    local_seq = full_seq // world
    send = x.reshape(bsz, world, local_seq, shard_heads, head_dim)
    send = send.permute(1, 0, 3, 2, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.permute(1, 3, 0, 2, 4).reshape(
        bsz, local_seq, world * shard_heads, head_dim
    ).contiguous()


def _sp_valid_key_mask(n: int, n_pad: int, device: torch.device) -> Optional[torch.Tensor]:
    if n_pad == n:
        return None
    mask = torch.zeros(n_pad, device=device, dtype=torch.bool)
    mask[n:] = True
    return mask.view(1, 1, 1, n_pad)


class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3

    Args:
        has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
        create_offset_ln_z (bool, optional): the value of create_offset for the LayerNorm applied to z. Defaults to False.
        n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        cross_attention_mode (bool, optional): If cross_attention_model = True, the adaptive layernorm will be applied
            to query and key/value seperately. Defaults to False.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = LayerNorm(c_a)

        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
            zero_init=not self.has_s,  # Adaptive zero init
        )
        self.layernorm_z = LayerNorm(c_z, create_offset=self.create_offset_ln_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

        # Line 13
        if self.has_s:
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            q (torch.Tensor): query embedding
                [..., N_atom, c_a]
            kv (torch.Tensor): key/value embedding
                [..., N_atom, c_a]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return q

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            q (torch.Tensor): the query embedding
                [..., N_token, c_a]
            kv (torch.Tensor): the key/value embedding
                [..., N_token, c_a]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        # Multi-head attention bias
        if enable_efficient_fusion:
            weight = (self.linear_nobias_z.weight * self.layernorm_z.weight[None, :])[
                :, :, None, None
            ]
            bias = F.conv2d(z, weight)
        else:
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = permute_final_dims(
                bias, [2, 0, 1]
            )  # [..., n_heads, N_token, N_token]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)

        return q

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = None

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a


class DiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_heads (int): number of heads for DiffusionTransformerBlock.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
        drop_path_rate: float = 0.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.profile_block_index = -1

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out_a: the output of DiffusionTransformerBlock [..., N, c_a]
                - s: the single embedding [..., N, c_s]
                - z: the pair embedding
        """
        profile_log = _diffusion_transformer_profile_path()
        profile_enabled = profile_log is not None
        timings_ms: dict[str, float] = {}

        _diffusion_transformer_sync_if_needed(profile_enabled, a)
        t_last = time.perf_counter()

        def mark_profile(name: str, ref: torch.Tensor = a) -> None:
            nonlocal t_last
            if not profile_enabled:
                return
            _diffusion_transformer_sync_if_needed(True, ref)
            now = time.perf_counter()
            timings_ms[name] = timings_ms.get(name, 0.0) + (now - t_last) * 1000.0
            t_last = now

        def write_profile(ref: torch.Tensor) -> None:
            if not profile_enabled:
                return
            _diffusion_transformer_sync_if_needed(True, ref)
            payload = {
                "path": "diffusion_transformer_block",
                "block_index": int(self.profile_block_index),
                "shape_a": [int(dim) for dim in a.shape],
                "shape_s": [int(dim) for dim in s.shape],
                "shape_z": [int(dim) for dim in z.shape],
                "dtype": str(a.dtype),
                "device": str(a.device),
                "n_heads": int(self.n_heads),
                "c_a": int(self.c_a),
                "c_s": int(self.c_s),
                "c_z": int(self.c_z),
                "local_attention": bool(n_queries and n_keys),
                "chunk_size": None if chunk_size is None else int(chunk_size),
                "enable_efficient_fusion": bool(enable_efficient_fusion),
                "timings_ms": timings_ms,
                "total_ms": float(sum(timings_ms.values())),
            }
            profile_dir = os.path.dirname(profile_log)
            if profile_dir:
                os.makedirs(profile_dir, exist_ok=True)
            with open(profile_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")

        attn_out = self.drop_path(
            self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
            )
        )
        mark_profile("attention_pair_bias", attn_out)
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        mark_profile("attention_residual", attn_out)
        ff_out = self.drop_path(self.conditioned_transition_block(a=attn_out, s=s))
        mark_profile("conditioned_transition_block", ff_out)
        out_a = ff_out + attn_out
        mark_profile("transition_residual", out_a)
        write_profile(out_a)
        # Avoid s/z to be deleted by torch.utils.checkpoint
        return out_a, s, z


class DiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_blocks (int): number of blocks in DiffusionTransformer.
        n_heads (int): number of heads in attention.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
        drop_path_rate (float, optional): drop skip connection path rate. Defaults to 0.0.
        blocks_per_ckpt (int, optional): number of DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        cross_attention_mode: bool = False,
        drop_path_rate: float = 0.0,  # drop skip connection path
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(DiffusionTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt
        self._ulysses_sp_bias_cache: dict[tuple, torch.Tensor] = {}

        self.blocks = nn.ModuleList()
        drop_path_rates = [
            drop_path_value.item()
            for drop_path_value in torch.linspace(0, drop_path_rate, n_blocks)
        ]
        for i in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
                drop_path_rate=drop_path_rates[i],
            )
            block.profile_block_index = i
            self.blocks.append(block)

    def _prep_blocks(
        self,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
    ) -> list[Callable]:
        blocks = [
            partial(
                b,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            for b in self.blocks
        ]
        return blocks

    def _ulysses_sp_cache_key(
        self,
        block_idx: int,
        z: torch.Tensor,
        rows: int,
        world: int,
        rank: int,
    ) -> tuple:
        return (
            block_idx,
            int(z.untyped_storage().data_ptr()),
            tuple(int(dim) for dim in z.shape),
            tuple(int(stride) for stride in z.stride()),
            str(z.dtype),
            str(z.device),
            int(rows),
            int(world),
            int(rank),
        )

    def _ulysses_sp_local_head_bias(
        self,
        block_idx: int,
        z: torch.Tensor,
        rows: int,
        world: int,
        rank: int,
    ) -> torch.Tensor:
        block = self.blocks[block_idx]
        module = block.attention_pair_bias
        n = z.shape[-3]
        n_pad = rows * world
        shard_heads = module.attention.num_heads // world
        head_start = rank * shard_heads
        head_end = head_start + shard_heads

        key = self._ulysses_sp_cache_key(block_idx, z, rows, world, rank)
        cached = self._ulysses_sp_bias_cache.get(key)
        if cached is not None:
            return cached

        if z.dim() == 4 and z.stride(0) == 0:
            z_for_bias = z[:1]
        else:
            z_for_bias = z
        z_norm = module.layernorm_z(z_for_bias)
        weight = module.linear_nobias_z.weight[head_start:head_end]
        bias = F.linear(z_norm, weight)
        bias = permute_final_dims(bias, [2, 0, 1]).contiguous()
        if n_pad != n:
            out = bias.new_zeros(*bias.shape[:-2], n_pad, n_pad)
            out[..., :n, :n] = bias
            bias = out
        if z.dim() == 4 and z.stride(0) == 0 and z.shape[0] != bias.shape[0]:
            bias = bias.expand(z.shape[0], *bias.shape[1:])
        max_entries = int(os.environ.get("PROTENIX_DIFFUSION_ULYSSES_SP_CACHE_MAX", "64"))
        if len(self._ulysses_sp_bias_cache) >= max_entries:
            self._ulysses_sp_bias_cache.clear()
        self._ulysses_sp_bias_cache[key] = bias
        return bias

    def _ulysses_sp_local_head_bias_fused(
        self,
        block_idx: int,
        z: torch.Tensor,
        rows: int,
        world: int,
        rank: int,
    ) -> torch.Tensor:
        block = self.blocks[block_idx]
        module = block.attention_pair_bias
        n = z.shape[-1]
        n_pad = rows * world
        shard_heads = module.attention.num_heads // world
        head_start = rank * shard_heads
        head_end = head_start + shard_heads

        key = self._ulysses_sp_cache_key(block_idx, z, rows, world, rank)
        cached = self._ulysses_sp_bias_cache.get(key)
        if cached is not None:
            return cached

        # Fused diffusion path prepares z as channel-first normalized pair features:
        # [B, C_z, N, N]. Reuse its 1x1 conv formulation but only for local heads.
        weight = (
            module.linear_nobias_z.weight[head_start:head_end]
            * module.layernorm_z.weight[None, :]
        )[:, :, None, None]
        bias = F.conv2d(z, weight).contiguous()
        if n_pad != n:
            out = bias.new_zeros(*bias.shape[:-2], n_pad, n_pad)
            out[..., :n, :n] = bias
            bias = out
        max_entries = int(os.environ.get("PROTENIX_DIFFUSION_ULYSSES_SP_CACHE_MAX", "64"))
        if len(self._ulysses_sp_bias_cache) >= max_entries:
            self._ulysses_sp_bias_cache.clear()
        self._ulysses_sp_bias_cache[key] = bias
        return bias

    def _ulysses_sp_project_qkv(self, module: AttentionPairBias, q_x: torch.Tensor):
        attn = module.attention
        q = attn.linear_q(q_x)
        k = attn.linear_k(q_x)
        v = attn.linear_v(q_x)
        q = q.view(*q.shape[:-1], attn.num_heads, attn.c_hidden)
        k = k.view(*k.shape[:-1], attn.num_heads, attn.c_hidden)
        v = v.view(*v.shape[:-1], attn.num_heads, attn.c_hidden)
        q = q / (attn.c_hidden**0.5)
        return q, k, v

    def _ulysses_sp_attention(
        self,
        block_idx: int,
        local_a: torch.Tensor,
        local_s: torch.Tensor,
        z: torch.Tensor,
        n: int,
        rows: int,
        world: int,
        rank: int,
        enable_efficient_fusion: bool,
    ) -> torch.Tensor:
        module = self.blocks[block_idx].attention_pair_bias
        q_x = module.layernorm_a(a=local_a, s=local_s)
        q_local, k_local, v_local = self._ulysses_sp_project_qkv(module, q_x)

        group = get_inference_parallel_context().mp_group
        q, k, v = _sp_seq2head_qkv(q_local, k_local, v_local, world, group)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        if enable_efficient_fusion:
            bias = self._ulysses_sp_local_head_bias_fused(
                block_idx, z, rows, world, rank
            )
        else:
            bias = self._ulysses_sp_local_head_bias(block_idx, z, rows, world, rank)

        logits = torch.matmul(q, k.transpose(-1, -2)) + bias
        mask = _sp_valid_key_mask(n, rows * world, logits.device)
        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)
        attn_out = torch.matmul(torch.softmax(logits, dim=-1), v)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        local_heads = _sp_head2seq(attn_out, world, group)
        out = module.attention._wrap_up(local_heads, q_x)
        return torch.sigmoid(module.linear_a_last(local_s)) * out

    def _ulysses_sp_forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        enable_efficient_fusion: bool,
    ) -> torch.Tensor:
        inference_parallel = get_inference_parallel_context()
        world = inference_parallel.mp_world_size
        rank = inference_parallel.mp_rank
        if self.n_heads % world != 0:
            raise ValueError(
                f"Diffusion Ulysses SP requires n_heads ({self.n_heads}) divisible by world ({world})"
            )
        n = a.shape[-2]
        start, end, rows = _sp_row_bounds(n, world, rank)
        local_a = _sp_pad_rows(a[:, start:end], rows)
        local_s = _sp_pad_rows(s[:, start:end], rows)

        for block_idx, block in enumerate(self.blocks):
            local_attn = self._ulysses_sp_attention(
                block_idx,
                local_a,
                local_s,
                z,
                n,
                rows,
                world,
                rank,
                enable_efficient_fusion,
            )
            local_a = local_attn + local_a
            local_a = block.conditioned_transition_block(a=local_a, s=local_s) + local_a

        return _sp_gather_rows(local_a, n, world, inference_parallel.mp_group)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """
                Args:
                    a (torch.Tensor): the single feature aggregate per-atom representation
                        [..., N, c_a]
                    s (torch.Tensor): single embedding
                        [..., N, c_s]
                    z (torch.Tensor): pair embedding
                        [..., N, N, c_z]
                    n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
                    n_keys (int, optional): local window size of key tensor. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

                Returns:
                    torch.Tensor: the output of DiffusionTransformer
                        [..., N, c_a]
        """
        if (
            _diffusion_ulysses_enabled()
            and dist.is_available()
            and _diffusion_ulysses_init_if_needed(a)
            and not torch.is_grad_enabled()
            and n_queries is None
            and n_keys is None
        ):
            return self._ulysses_sp_forward(
                a=a, s=s, z=z, enable_efficient_fusion=enable_efficient_fusion
            )

        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            enable_efficient_fusion=enable_efficient_fusion,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )
        del s, z
        return a


class AtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3

    Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings

    Args:
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in attention. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p (torch.Tensor): atompair embedding in dense block shape.
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the output of AtomTransformer
                [..., N_atom, c_atom]
        """
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )


class ConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3

    Args:
        c_a (int): single embedding dim (single feature aggregated atom info).
        c_s (int):  single embedding dim.
        n (int, optional): channel scale factor. Defaults to 2.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super(ConditionedTransitionBlock, self).__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_a2 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class AtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3

    Args:
        has_coords (bool): whether the module input will contains coordinates (r_l).
        c_token (int): token embedding dim.
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atompair embedding dim. Defaults to 16.
        c_s (int, optional):  single embedding dim. Defaults to 384.
        c_z (int, optional): pair embedding dim. Defaults to 128.
        n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in AtomTransformer. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.local_attention_method = "local_cross_attention"

        self.input_feature = {
            # "ref_pos": 3,
            # "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_ref_pos = LinearNoBias(
            in_features=3, out_features=self.c_atom, precision=torch.float32
        )  # use high precision for ref_pos
        self.linear_no_bias_ref_charge = LinearNoBias(
            in_features=1, out_features=self.c_atom
        )
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair, precision=torch.float32
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s, create_offset=False)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line10
            self.layernorm_z = LayerNorm(
                self.c_z, create_offset=False
            )  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom, precision=torch.float32
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="zeros",
            ),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def prepare_cache(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: torch.Tensor,
        r_l: Union[torch.Tensor, bool, None] = None,
        z: torch.Tensor = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            # use arcsinh for ref_charge
            torch.arcsinh(ref_charge).reshape(*batch_shape, N_atom, 1)
        )
        if inplace_safe:
            c_l += self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l *= ref_mask.reshape(*batch_shape, N_atom, 1)
        else:
            c_l = c_l + self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)

        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info[
            "mask_trunked"
        ].unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm += self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        if r_l is not None:
            p_lm = (
                p_lm.unsqueeze(dim=-5)
                + broadcast_token_to_local_atom_pair(
                    z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                    atom_to_token_idx=atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )[0]
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]
        return p_lm, c_l

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: torch.Tensor,
        r_l: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        p_lm: torch.Tensor = None,
        c_l: torch.Tensor = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_to_token_idx (torch.Tensor): atom_to_token_idx
            ref_pos (torch.Tensor): ref_pos
            ref_charge (torch.Tensor): ref_charge
            ref_mask (torch.Tensor): ref_mask
            ref_atom_name_chars (torch.Tensor): ref_atom_name_chars
            ref_element (torch.Tensor): ref_element
            r_l (torch.Tensor, optional): noisy position.
                [..., N_sample, N_atom, 3] if has_coords else None.
            s (torch.Tensor, optional): single embedding.
                [..., N_sample, N_token, c_s] if has_coords else None.
            z (torch.Tensor, optional): pair embedding
                [..., N_sample, N_token, N_token, c_z] if has_coords else None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., (N_sample), N_token, c_token]
            q_l:
                [..., (N_sample), N_atom, c_atom]
            c_l:
                [..., (N_sample), N_atom, c_atom]
            p_lm:
                [..., (N_sample), N_atom, N_atom, c_atompair]

        """

        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element,
                atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm,
                v_lm=v_lm,
                pad_info=pad_info,
                r_l=r_l,
                z=z,
                inplace_safe=inplace_safe,
            )
        else:
            if inplace_safe:
                p_lm_clone = p_lm.clone()
                c_l_clone = c_l.clone()
                p_lm = p_lm_clone
                c_l = c_l_clone

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            # Broadcast the single and pair embedding from the trunk
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_sample, N_atom, c_atom]

            # Add the noisy positions
            # Different from paper!!
            q_l = c_l + self.linear_no_bias_r(r_l)  # [..., N_sample, N_atom, c_atom]
        else:
            q_l = c_l.clone()

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        if inplace_safe:
            p_lm = p_lm + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
                + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            )  # [..., (N_sample), n_blocks, n_queries, n_keys, c_atompair]

            # Run a small MLP on the pair activations
            p_lm = p_lm + self.small_mlp(p_lm)

        # Cross attention transformer
        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )  # [..., (N_sample), N_token, c_token]
        return a, q_l, c_l, p_lm


class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3

    Args:
        n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
        c_token (int, optional): feature channel of token (single a). Defaults to 384.
        c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
        c_atompair (int, optional): embedding dim for atom pair embedding. Defaults to 16.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = LinearNoBias(
            in_features=c_atom, out_features=3, precision=torch.float32
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token_idx (torch.Tensor): the atom to token index
                [..., N_atom]
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated noisy coordinates
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),  # [..., N_token, c_atom]
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        # Map to positions update
        r = self.linear_no_bias_out(self.layernorm_q(q))

        return r
