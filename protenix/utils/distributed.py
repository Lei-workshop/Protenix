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

import os
from dataclasses import dataclass
from typing import Optional

import torch


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def all_gather_object(self, obj, group=None):
        """Function to gather objects from several distributed processes.
        It is now only used by sync metrics in logger due to security reason.
        """
        if self.world_size > 1 and distributed_available():
            with torch.no_grad():
                obj_list = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(obj_list, obj, group=group)
                return obj_list
        else:
            return [obj]


DIST_WRAPPER = DistWrapper()


@dataclass
class InferenceParallelContext:
    mp_size: int
    mp_rank: int
    mp_world_size: int
    mp_group_id: int
    mp_group: Optional[object]
    mp_leader_rank: int
    dp_rank: int
    dp_world_size: int

    @property
    def is_mp_leader(self) -> bool:
        return self.mp_rank == 0


_INFERENCE_PARALLEL_CONTEXT: Optional[InferenceParallelContext] = None


def _cooperative_inference_requested() -> bool:
    return (
        os.environ.get("PROTENIX_PAIRFORMER_ROW_PARALLEL", "0") == "1"
        or os.environ.get("PROTENIX_DIFFUSION_ULYSSES_SP", "0") == "1"
        or os.environ.get("PROTENIX_DISTRIBUTED_DATA_BROADCAST", "0") == "1"
    )


def _inference_mp_size() -> int:
    value = os.environ.get("PROTENIX_INFERENCE_MP_SIZE")
    if value is not None:
        return int(value)
    if DIST_WRAPPER.world_size > 1 and _cooperative_inference_requested():
        return DIST_WRAPPER.world_size
    return 1


def get_inference_parallel_context() -> InferenceParallelContext:
    global _INFERENCE_PARALLEL_CONTEXT
    if _INFERENCE_PARALLEL_CONTEXT is not None:
        return _INFERENCE_PARALLEL_CONTEXT

    world_size = DIST_WRAPPER.world_size
    rank = DIST_WRAPPER.rank
    mp_size = _inference_mp_size()
    if mp_size < 1:
        raise ValueError(f"PROTENIX_INFERENCE_MP_SIZE must be >= 1, got {mp_size}")
    if world_size % mp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by PROTENIX_INFERENCE_MP_SIZE ({mp_size})"
        )

    mp_group = None
    if mp_size > 1 and mp_size < world_size:
        if not distributed_available():
            raise RuntimeError("Distributed process group must be initialized before creating inference MP groups")
        num_groups = world_size // mp_size
        for group_id in range(num_groups):
            ranks = list(range(group_id * mp_size, (group_id + 1) * mp_size))
            group = torch.distributed.new_group(ranks=ranks)
            if rank in ranks:
                mp_group = group

    mp_group_id = rank // mp_size
    mp_rank = rank % mp_size
    dp_world_size = world_size // mp_size
    ctx = InferenceParallelContext(
        mp_size=mp_size,
        mp_rank=mp_rank,
        mp_world_size=mp_size,
        mp_group_id=mp_group_id,
        mp_group=mp_group,
        mp_leader_rank=mp_group_id * mp_size,
        dp_rank=mp_group_id,
        dp_world_size=dp_world_size,
    )
    _INFERENCE_PARALLEL_CONTEXT = ctx
    return ctx


def traverse_and_aggregate(dict_list, aggregation_func=None):
    """Traverse list of dicts and merge into a single dict with leaf values joined to list."""
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]

        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(
                agg_value, aggregation_func=aggregation_func
            )
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value

    return merged_dict


def gather_and_merge(metrics, aggregation_func=None):
    """Gather metrics from ddp workers and aggregate leaf metrics."""
    gathered_metrics = DIST_WRAPPER.all_gather_object(metrics)  # list of metrics
    merged_metrics = traverse_and_aggregate(gathered_metrics, aggregation_func)
    return merged_metrics
