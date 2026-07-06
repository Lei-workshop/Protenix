# Protenix 多卡推理并行技术评估

本文评估 Protenix 在 MetaX C500 8 卡机器上的推理加速方案。依据包括：

- 当前 triangle attention v9 工作；
- triangle multiplicative update 实验；
- diffusion profiling；
- `/root/protenix-triangle-operators-study.md` 中对 Protenix 推理流程的整理。

## 当前结论

- 当前推荐的多卡推理主线是单个 4 卡 island：Pairformer row-parallel + Diffusion Ulysses SP + MP group leader input preprocessing/data broadcast。
- 关闭所有 profile 的默认配置上，4GPU 总 job 为 `218.98s`；最大 7wux case 的 model forward 从单卡 `248.25s` 降到 `130.96s`，约 `1.90x`。
- 2026-07-06 更新：Diffusion Ulysses SP 已补齐 `enable_fusion=True` 的 fused pair-bias 路径；完整默认三 case no-profile A/B 显示 `enable_fusion=False` 略优，4GPU 总 job `157.56s`，当前作为 4GPU SP 性能优先推荐配置。
- 2GPU 在卡时性价比上更均衡，4GPU 更适合大 N 或低延迟目标；8GPU 暂不作为默认目标，因为跨 island gather 收益递减明显。
- 当前实现支持 DP x MP 组合：`PROTENIX_INFERENCE_MP_SIZE` 控制每个协同推理组的卡数；不设置时默认 `mp_size=world_size`，保持纯 MP 行为。
- 默认 `N_sample=5` sample parallel 已验证无明显收益，不再推进；Diffusion 继续走 Ulysses sequence parallel，而不是 sample 切分。
- Pairformer row micro-batch overlap 暂不做：真实 profile 中 gather 只占约 `15.8% - 17.4%`，切得更细很可能增加 collective latency。
- 更大范围 row-sharded state、减少 full-`z` 同步次数是潜在方向，但依赖链复杂，当前只记录，不实施。
- PyTorch-level tri-mul matmul 重排已经验证为负结果，相关实验代码已移除；如果后续继续优化 tri-mul compute，应考虑专用 fused kernel / mctlass-style MMA。

## 当前 Protenix fork 实现状态

当前 fork 支持两种启动模式：

- 两个独立 `torchrun`：每个任务独立拥有自己的 global rank `0..world_size-1`，建议用 `CUDA_VISIBLE_DEVICES` 隔离物理卡，并使用不同 `--rdzv_endpoint` / port。
- 单个 `torchrun` 内做 DP x MP：例如 `world_size=4, PROTENIX_INFERENCE_MP_SIZE=2` 时，rank `[0,1]` 和 `[2,3]` 分别组成两个 2 卡 MP group，对应两个 DP shard。

实现语义：

- `PROTENIX_INFERENCE_MP_SIZE` 只在启用协同推理路径时生效；默认值是 `world_size`，即一个任务内所有 rank 共同处理同一个 sample。
- input preprocessing、dataloader item broadcast、Pairformer collectives、Diffusion Ulysses SP collectives 都在 MP group 内发生。
- dump owner 是 MP group leader，而不是简单的 global rank0。这样 `2DP x 2MP` 时 rank0 负责第一个 DP shard 输出，rank2 负责第二个 DP shard 输出。
- DP x MP dataloader 使用按 DP rank 的非 padding 分片，避免 `DistributedSampler` 在样本数不能整除时重复 padding sample。

精度状态：

- 默认完整性能测试使用 `configs.deterministic=False`，不能用 dumped CIF/summary 做跨运行 bitwise 判断。
- 打开 `configs.deterministic=True` 后，2GPU pure MP 最小 case 连续两次输出 byte-wise 一致。
- `2DP x 2MP` 与等价 pure MP baseline 对齐时，需要保持相同 sample 顺序/RNG 顺序；在该条件下，`7r6r`、`7pzb`、`7wux` 的 CIF 和 summary JSON SHA256 均一致。

## 硬件假设

- 目标机器：8x MetaX C500。
- 当前讨论中的硬件假设：每 4 张卡组成一个强互联 island，带宽特征接近 NVLink 类 intra-island 通信。
- 实际规划默认：
  - 先按单个 4 卡 island 设计；
  - 只有在测清跨 island 带宽和延迟后，再扩展到 8 卡；
  - 优先选择主要发生在 island 内的通信模式。

2026-07-03 更新：C500 collective microbench 结果已经补充在本文后半部分。4 卡 island 的通信结果足够好，Pairformer row-parallel 已经值得做真实 POC。

## 当前 7wux 运行时背景

`WMMA_VERSION=v9`、`trimul=torch` 下，7wux 相关 profile：

| 阶段 | 时间 | 备注 |
|---|---:|---|
| `get_pairformer_output` | ~162s | Pairformer/trunk 主体 |
| `sample_diffusion` | ~79s | 200 diffusion steps，5 samples |
| `confidence_head` | ~9.6s | 较小目标 |

Pairformer block 级别 profile：

| Pairformer 工作 | 时间 |
|---|---:|
| triangle attention start/end | ~88s |
| triangle multiplicative out/in | ~73s |
| pair transition | ~8s |

Diffusion profile：

| Diffusion 工作 | 时间 |
|---|---:|
| token-level `DiffusionTransformer` | ~64.5s |
| atom attention encoder + decoder | ~12.2s |

## 方向 1：Diffusion `N_sample` 并行

最初判断这是最干净的多卡目标，但真实 Protenix 接入测试显示收益很小。

默认推理使用：

```text
N_sample = 5
N_step   = 200
```

trunk 产生以下共享输入后，5 条 diffusion 轨迹相互独立：

```text
s_trunk
z_trunk / pair_z
p_lm / c_l cache
input_feature_dict
```

建议切分方式：

```text
GPU0: sample 0
GPU1: sample 1
GPU2: sample 2
GPU3: sample 3
GPU4: sample 4
```

每张卡运行自己负责的完整 200-step diffusion loop，最后 gather coordinates 给 confidence/ranking。

预期特征：

- 每个 diffusion step 内基本不需要通信；
- 不需要改 triangle attention、triangle multiplicative update 或 diffusion transformer kernel；
- 如果 sample RNG/seed 语义保持一致，推理质量可以不变；
- `N_sample=5` 时天然最多用 5 张卡；剩余卡可跑另一个 seed/job，或在该阶段空闲。

理论收益：

- 只加速 diffusion 和 sample 相关下游工作；
- diffusion 约为 `79s / 253s`，所以理想端到端收益上限约 31%；
- 5-way sample parallel 下，diffusion 时间可接近单条 sample trajectory 的时间加 gather 开销。

当前建议：不作为主线推进，除非后续把 `sample_diffusion_chunk_size` 调小、显存压力迫使 sample 串行，或目标场景的 `N_sample` 明显大于默认 5。

### 最小 POC 结果：2026-07-02

新增 benchmark：

```text
benchmarks/bench_diffusion_sample_parallel.py
```

该 benchmark 保持关键 diffusion tensor shape，将 `N_sample` 按 rank 切分，并 gather 最终 coordinates。它是 synthetic diffusion workload，不是完整 Protenix diffusion，因此测的是并行形态和通信开销，不是精确模型运行时。

真实 shape、缩短循环的测试：

```text
N_sample = 5
N_token  = 1218
N_atom   = 9142
C        = 768
heads    = 16
steps    = 40
blocks   = 2
GPUs     = 4
```

实测结果：

| 指标 | 时间 |
|---|---:|
| 单卡 synthetic compute | 573.935 ms |
| 4 卡 distributed wall time | 240.520 ms |
| 加速比 | 2.39x |
| sample 切分 | `[2, 1, 1, 1]` |

解释：

- 这接近 `N_sample=5` 在 4 卡上的理论上限。由于最慢 rank 拿 2 个 samples，理想加速比约为 2.5x。
- 在这个 synthetic reduced test 中，通信不是 sample parallel 的瓶颈。
- 但这个 benchmark 没有反映真实 Protenix 默认 `sample_diffusion_chunk_size=5` 的 batch 化执行，因此不能作为真实接入收益依据。

### 真实 Protenix 接入验证：2026-07-03

实验性地在 `sample_diffusion` 内部做过 `N_sample` 按 rank 切分，并在末尾 all-gather coordinates。实现要点：

- `PROTENIX_DIFFUSION_SAMPLE_PARALLEL=1` opt-in；
- `N_sample=5, world=4` 时切分为 `[2, 2, 1, 0]`；
- 每个 rank 用 `base_seed + sample_start` 避免不同 rank 生成重复 noise；
- gather 后保持原始 `[N_sample, N_atom, 3]` 输出给 confidence head。

真实 7wux、4 卡、同时启用 Pairformer row-parallel：

| 配置 | no split rank0 forward | sample split rank0 forward | 结论 |
|---|---:|---:|---|
| `cycle=1, step=5, sample=5` | 29.10s | 20.82s | 单次 wall time 有波动，看 profile 不成立 |
| `cycle=1, step=40, sample=5` | 33.32s | 33.60s | 无收益 |

profile 分段更明确：

| 配置 | no split `sample_diffusion` | sample split `sample_diffusion` |
|---|---:|---:|
| `step=5` | 6.489s | 6.433s |
| `step=40` | 19.402s | 19.336s |

结论：

- 默认配置 `sample_diffusion_chunk_size=5`，真实路径本来就把 5 个 samples 作为一个 batch 跑。
- 把 samples 拆到多卡并不会显著降低 diffusion transformer 的 wall time，反而增加 RNG/通信/ordering 复杂度。
- 该实验性代码没有保留到 Protenix 主线；不要继续按“默认 N_sample=5 sample parallel”推进。

## 方向 2：Pairformer 沿 token N 做 row-parallel

triangle attention 和 triangle multiplicative update 都有天然的输出 row sharding 维度。

对 triangle attention：

```text
output[i, :, :] depends on full input z/Q/K/V and full Bias2
```

row-parallel 切分：

```text
GPU r computes i in shard_r
each GPU keeps full input z-derived tensors
all-gather row-sharded output update
apply residual to reconstruct full z
```

对 triangle multiplicative update：

```text
outgoing: x[i,j,c] = sum_k a[i,k,c] * b[k,j,c]
incoming: x[i,j,c] = sum_k a[k,i,c] * b[k,j,c]
```

如果每张卡拥有足够的 `a/b` 数据，也可以按输出 row 切分。最简单 POC 是复制完整 `a/b`，只 shard 输出 rows。

主要挑战：

```text
z size for 7wux = 1218 * 1218 * 128 * 2 bytes ≈ 380 MB
```

Pairformer layer 顺序：

```text
tri_mul_out -> tri_mul_in -> tri_att_start -> transpose -> tri_att_end
-> transpose -> pair_transition
```

每次 residual update 都产生新的 `z`，后续模块需要完整 `z`。简单 row-parallel 实现因此需要在每个 Pairformer block 内多次 full-`z` all-gather。

影响：

- 在强 4 卡 island 内，Pairformer row-parallel 可能可行；
- 扩到 8 卡时，跨 island 通信可能成为问题，需要实测；
- 第一版 POC 应优先从 4 卡 island 开始。

### 最小 kernel POC 结果：2026-07-02

新增 benchmark：

```text
benchmarks/bench_triangle_attention_row_parallel.py
```

该 benchmark 调用真实 triangle attention v9 wrapper，沿 outer row dimension 做 sharding。每个 rank 持有完整输入 tensor，计算自己的 row-sharded output，然后 all-gather 输出。它也支持 microchunked all-gather 和 async overlap。

真实 shape 测试：

```text
N_outer = 512
S       = 1218
H       = 4
D       = 32
GPUs    = 4
```

每 rank 一个 coarse shard 的结果：

| 指标 | 时间 |
|---|---:|
| 单卡完整调用 | 25.208 ms |
| row-parallel compute only | 6.372 ms |
| row-parallel communication only | 0.865 ms |
| row-parallel total | 7.169 ms |
| 相比单卡加速比 | 3.52x |

每 rank 4 个 microchunks 的结果：

| 模式 | Total | Compute | Communication |
|---|---:|---:|---:|
| no overlap | 7.791 ms | 6.647 ms | 1.174 ms |
| async overlap | 7.300 ms | 6.553 ms | 1.175 ms |

解释：

- 单个 triangle attention call 按 row shard 后，在 4 卡上扩展很好。
- 对这个 kernel-level test，输出 all-gather 相比 compute 很小。
- microchunk overlap 能隐藏一部分通信，但单个 kernel call 下没有超过 coarse one-shard 版本，因为额外 collective/launch overhead 抵消了收益。
- 剩余风险不在这个单 kernel all-gather，而在完整 Pairformer：更新后的 `z[N,N,C]` 需要在 triangle multiplicative update、triangle attention、transpose、pair transition 之间反复重建。

### 模块级 triangle attention POC：2026-07-03

新增 benchmark：

```text
benchmarks/bench_pairformer_triatt_row_parallel.py
```

这个 benchmark 比 raw Q/K/V benchmark 更接近真实 Pairformer 路径。它实例化 Protenix 的 `TriangleAttention` 模块，测量：

```text
z += tri_att_start(z)
z = z.transpose(-2, -3).contiguous()
z += tri_att_end(z)
z = z.transpose(-2, -3).contiguous()
```

被测路径包括：

- layernorm；
- triangle bias projection；
- Q/K/V/O projections；
- WMMA triangle attention v9；
- residual add；
- transpose/contiguous；
- 每次 residual update 后的 full-`z` all-gather。

重要接口发现：

- 当前 WMMA wrapper 假设 Bias2 shape 是 `[B, 1, H, S, S]`。
- row-sharded Protenix `TriangleAttention` 自然生成 local Bias2 rows：`[B, 1, H, rows, S]`。
- benchmark 为了不改 production v9 wrapper，把 local Bias2 rows pack 到 padded `[B, 1, H, S, S]` buffer 的前 `rows` 行。
- 正式 row-parallel 实现应该给 wrapper/kernel 增加显式 Bias2 row-offset/row-count 路径，或者做专用 row-shard wrapper，避免不必要的 Bias2 padding。

Correctness：

```text
N = 128
4 GPUs
max diff  = 0.000000
mean diff = 0.000000
```

真实 shape 结果：

```text
N = 1218
C_z = 128
c_hidden = 32
heads = 4
```

| 模式 | 时间 | 相比单卡加速比 |
|---|---:|---:|
| 单卡 module segment | 135.963 ms | 1.00x |
| 4 卡 row-parallel | 40.409 ms | 3.36x |
| 8 卡 row-parallel | 30.447 ms | 4.47x |

解释：

- 模块级结果确认：加入真实 Protenix module overhead 后，row-parallel triangle attention 仍然有效。
- 4 卡已经拿到主要收益，效率较好。
- 8 卡仍然继续变快，但从 4 卡到 8 卡只有约 `1.33x`，和跨 island collective 变慢的实测一致。
- 这个 POC 只覆盖 Pairformer block 内两个 triangle attention 模块，尚未包括 triangle multiplicative update、pair transition、single update 或完整 `PairformerStack` scheduling。

### 模块级 triangle multiplicative update POC：2026-07-03

新增 benchmark：

```text
benchmarks/bench_pairformer_trimul_row_parallel.py
```

这个 benchmark 验证 triangle multiplicative update 是否能按输出 row 做多卡并行。它不是调用完整 module 后再切片，而是复用 Protenix 的真实 module 权重和 layernorm/linear，手写 local-output-row 计算：

```text
outgoing: update[i,j,c] = sum_k a[i,k,c] * b[k,j,c]
incoming: update[i,j,c] = sum_k a[k,i,c] * b[k,j,c]
```

每个 rank 只计算自己负责的 `i` rows，做 residual add 后 all-gather full `z`，再继续后续方向。

Correctness：

```text
N = 128
4 GPUs
direction = both
max diff  = 0.000000
mean diff = 0.000000
```

真实 shape 结果：

```text
N = 1218
C_z = 128
C_hidden = 128
GPUs = 4
```

| 模式 | 单卡时间 | 4 卡 row-parallel | 加速比 |
|---|---:|---:|---:|
| `tri_mul_out` | 54.013 ms | 17.490 ms | 3.09x |
| `tri_mul_in` | 54.108 ms | 26.489 ms | 2.04x |
| `tri_mul_out + tri_mul_in` | 107.960 ms | 44.504 ms | 2.43x |

解释：

- triangle multiplicative update 也可以做 row-parallel，并且 correctness 对齐。
- `tri_mul_out` 的扩展性较好，接近 3x。
- `tri_mul_in` 明显弱一些，主要怀疑点是 incoming 的 row 输出需要取 `a[:, :, rows]` 这种 column shard，内存 layout 和 einsum/matmul 访问不如 outgoing。
- 串联 out+in 后 4 卡约 `2.43x`，仍有价值，但低于 triangle attention 模块级 POC 的 `3.36x`。
- 下一步如果正式接入 PairformerBlock，应优先处理 incoming 的 layout：例如预转置/缓存 `a`，或者把 incoming 通过等价 transpose 路径改写成更接近 outgoing 的连续 row-shard 访问。

### PairformerBlock subset POC：2026-07-03

新增 benchmark：

```text
benchmarks/bench_pairformer_block_subset_row_parallel.py
```

这个 benchmark 把前两个模块级 POC 合成一个更接近真实 PairformerBlock 的子图：

```text
tri_mul_out
-> all_gather full z
-> tri_mul_in
-> all_gather full z
-> tri_att_start
-> all_gather full z
-> transpose
-> tri_att_end
-> all_gather full z
-> transpose
```

它仍然不包含 `pair_transition`、`attention_pair_bias`、`single_transition`，因此不是完整 PairformerBlock。但它已经覆盖当前最大头部：两个 triangle multiplicative update 和两个 triangle attention。

Correctness：

```text
N = 128
4 GPUs
max diff  = 0.000000
mean diff = 0.000000
```

真实 shape 结果：

```text
N = 1218
C_z = 128
trimul_hidden = 128
triatt_hidden = 32
triatt_heads = 4
```

| 模式 | 时间 | 相比单卡加速比 |
|---|---:|---:|
| 单卡 subset | 243.772 ms | 1.00x |
| 4 卡 row-parallel subset | 84.516 ms | 2.88x |
| 8 卡 row-parallel subset | 71.329 ms | 3.42x |

4 卡分段 profile：

| 阶段 | 时间 |
|---|---:|
| `tri_mul_out_compute` | 15.561 ms |
| `tri_mul_out_residual_gather` | 2.315 ms |
| `tri_mul_in_compute` | 24.949 ms |
| `tri_mul_in_residual_gather` | 2.285 ms |
| `tri_att_start_compute_residual` | 17.624 ms |
| `tri_att_start_gather` | 1.839 ms |
| `transpose_after_start` | 1.332 ms |
| `tri_att_end_compute_residual` | 17.535 ms |
| `tri_att_end_gather` | 1.862 ms |
| `transpose_after_end` | 1.429 ms |
| `total_profiled` | 86.732 ms |

8 卡分段 profile：

| 阶段 | 时间 |
|---|---:|
| `tri_mul_out_compute` | 12.845 ms |
| `tri_mul_out_residual_gather` | 5.521 ms |
| `tri_mul_in_compute` | 18.156 ms |
| `tri_mul_in_residual_gather` | 5.688 ms |
| `tri_att_start_compute_residual` | 9.212 ms |
| `tri_att_start_gather` | 5.486 ms |
| `transpose_after_start` | 1.381 ms |
| `tri_att_end_compute_residual` | 9.106 ms |
| `tri_att_end_gather` | 5.504 ms |
| `transpose_after_end` | 1.397 ms |
| `total_profiled` | 74.296 ms |

解释：

- 组合后 4 卡仍有 `2.88x`，说明多次 all-gather、transpose 和模块边界没有吃掉整体收益。
- 4 卡下每次 full-`z` gather 约 `1.8-2.3 ms`，不是主瓶颈。
- `tri_mul_in_compute` 是 4 卡下最重的一段，仍然是后续优化重点。
- 8 卡比 4 卡继续变快，但只从 `84.516 ms` 降到 `71.329 ms`，额外收益较小。
- 8 卡下每次 gather 上升到约 `5.5 ms`，跨 island 通信已经明显吃掉一部分计算收益。
- 因此正式接入仍建议先做单个 4 卡 island；8 卡更适合作为后续扩展或跑两个独立 job/seed/sample group。

### PairformerBlock 实验性集成：2026-07-03

新增/修改：

```text
/root/Protenix/protenix/model/modules/pairformer.py
benchmarks/bench_pairformer_block_integrated_row_parallel.py
```

接入方式：

- 默认路径不变。
- 只有显式设置下面环境变量时才启用：

```text
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
```

- 需要用 `torchrun --nproc_per_node=4` 启动。
- 触发条件较严格：
  - 非 training；
  - `torch.is_grad_enabled() == False`；
  - `triangle_multiplicative == "torch"`；
  - `triangle_attention == "wmma"`；
  - CUDA tensor；
  - `WORLD_SIZE > 1`。

当前集成边界：

- 在真实 `PairformerBlock.forward` 开头分支；
- triangle 子图按 row-parallel 执行：

```text
tri_mul_out -> all_gather
tri_mul_in -> all_gather
tri_att_start -> all_gather
transpose
tri_att_end -> all_gather
transpose
```

- `pair_transition` 仍然在每张卡上复制执行；
- 如果 `c_s > 0`，`attention_pair_bias` 和 `single_transition` 也仍然在每张卡上复制执行；
- 每个 rank 最终都有完整 `z` 和完整 `s`，方便后续模块继续按原逻辑运行。

Correctness smoke：

```text
N = 128
4 GPUs
c_s = 0
z max diff  = 0.000000
z mean diff = 0.000000
```

在 autocast 下覆盖 `s` 更新：

```text
N = 128
4 GPUs
c_s = 384
z max diff  = 0.000000
z mean diff = 0.000000
s max diff  = 0.000000
s mean diff = 0.000000
```

真实 `PairformerBlock.forward` 集成结果：

| Shape | 单卡 | 4 卡 row-parallel | 加速比 | 备注 |
|---|---:|---:|---:|---|
| `N=512, c_s=0` | 22.527 ms | 12.901 ms | 1.75x | 含 `pair_transition` |
| `N=512, c_s=384` | 23.807 ms | 13.980 ms | 1.70x | autocast，z/s correctness 通过 |
| `N=1218, c_s=0` | 257.526 ms | 97.706 ms | 2.64x | 含 `pair_transition` |
| `N=1218, c_s=384` | 262.415 ms | 103.622 ms | 2.53x | autocast，含 s update |

解释：

- 这已经不再是独立 benchmark 复制逻辑，而是真实 `PairformerBlock.forward` 的实验分支。
- 与 subset POC 的 `2.88x` 相比，真实 block 集成降到 `2.53-2.64x`，主要因为 `pair_transition`、`attention_pair_bias`、`single_transition` 仍然复制执行。
- 对 `N=1218` 这种长序列，4 卡收益仍然明确。
- 下一步应进入真实 `PairformerStack`/端到端推理 POC，而不是继续扩大单 block microbench。
- 后续优化点：
  - 优化 `tri_mul_in` layout；
  - 评估 `pair_transition` 是否值得 row-shard；
  - 只让 rank0 继续输出/写结果，或定义多 rank 后处理策略，避免端到端重复工作。

### PairformerStack 实验性集成：2026-07-03

新增 benchmark：

```text
benchmarks/bench_pairformer_stack_integrated_row_parallel.py
```

这个 benchmark 直接实例化 Protenix 的 `PairformerStack`，连续执行多个真实 `PairformerBlock.forward`，并通过 `PROTENIX_PAIRFORMER_ROW_PARALLEL=1` 触发实验性 row-parallel 分支。

Correctness smoke：

```text
N = 128
blocks = 2
4 GPUs
autocast
z max diff  = 0.000000
z mean diff = 0.000000
s max diff  = 0.000000
s mean diff = 0.000000
```

中等规模：

```text
N = 512
blocks = 4
c_s = 384
```

| 模式 | 时间 | 加速比 |
|---|---:|---:|
| 单卡 PairformerStack | 95.185 ms | 1.00x |
| 4 卡 row-parallel | 55.980 ms | 1.70x |

长序列 4 blocks：

```text
N = 1218
blocks = 4
c_s = 384
```

| 模式 | 时间 | 加速比 |
|---|---:|---:|
| 单卡 PairformerStack | 1047.413 ms | 1.00x |
| 4 卡 row-parallel | 431.070 ms | 2.43x |

长序列 48 blocks synthetic full-stack：

```text
N = 1218
blocks = 48
c_s = 384
warmup = 0
iters = 1
```

| 模式 | 时间 | 加速比 |
|---|---:|---:|
| 单卡 PairformerStack | 13063.442 ms | 1.00x |
| 4 卡 row-parallel | 5536.433 ms | 2.36x |

解释：

- 48-block stack 级别仍有 `2.36x`，说明 repeated full-`z` all-gather 没有随 block 数累计成新的主瓶颈。
- 真实 stack 加速比低于 triangle-only subset，因为 `pair_transition`、`attention_pair_bias`、`single_transition` 目前仍复制执行。
- `N=512` 只有 `1.70x`，说明短序列时复制工作和通信/调度开销占比更高。
- `N=1218` 长序列收益明显，适合 7wux 这种大 N 场景。
- 下一步端到端接入要解决 rank 输出策略：所有 rank 都需要参与 Pairformer collectives，但不应该让所有 rank 写同一份预测输出。

### Batch Inference 4 卡 Smoke：2026-07-03

新增/修改：

```text
/root/Protenix/protenix/data/inference/infer_dataloader.py
/root/Protenix/runner/inference.py
```

改动：

- `PROTENIX_PAIRFORMER_ROW_PARALLEL=1` 且 `WORLD_SIZE>1` 时，inference dataloader 在纯 MP 模式下不再使用 `DistributedSampler` 分片。
- 同一个 MP group 内的 rank 按相同顺序处理相同 sample，确保 Pairformer collectives 中每个 rank 都在同一个样本上。
- row-parallel 模式下只允许 MP group leader 执行 `runner.dumper.dump(...)`，避免同组 rank 写同一份结果；DP x MP 模式下每个 MP group leader 负责各自 DP shard 的输出。
- 默认只在 `N_token >= 512` 时启用 row-parallel；可用 `PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N` 覆盖，设为 `0` 表示所有 shape 都尝试启用。

4 卡 smoke 命令核心：

```text
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
WMMA_VERSION=v9
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29605 \
  runner/batch_inference.py \
  --input=examples/example.json \
  --out_dir=/tmp/protenix_row_parallel_smoke \
  --triatt_kernel=wmma \
  --trimul_kernel=torch \
  --seeds=101 \
  --cycle=1 \
  --step=1 \
  --sample=1
```

同配置单卡对照：

```text
python runner/batch_inference.py \
  --input=examples/example.json \
  --out_dir=/tmp/protenix_single_smoke \
  --triatt_kernel=wmma \
  --trimul_kernel=torch \
  --seeds=101 \
  --cycle=1 \
  --step=1 \
  --sample=1
```

结果：

| 样本 | 单卡 model forward | 4 卡 rank0 model forward | 备注 |
|---|---:|---:|---|
| 7r6r | 5.94s | 7.59s | 小 N，4 卡 overhead 占优 |
| 7wux | 19.22s | 8.75s | 长序列，收益明确 |
| 7pzb | 3.94s | 2.67s | 中等 N，有收益 |
| total job | 40.82s | 31.37s | 三组总流程 |

输出检查：

```text
/tmp/protenix_row_parallel_smoke/7r6r/seed_101/predictions/...
/tmp/protenix_row_parallel_smoke/7wux/seed_101/predictions/...
/tmp/protenix_row_parallel_smoke/7pzb/seed_101/predictions/...
```

每个样本只生成一份 rank0 输出，没有发现 4 rank 重复写冲突。

解释：

- 这是第一个真实 `runner/batch_inference.py` 4 卡合作式 row-parallel smoke。
- `cycle=1, step=1, sample=1` 是功能 smoke，不是最终性能配置。
- 长序列 7wux 在真实 batch 流程中已经从 `19.22s` 降到 `8.75s`。
- 小样本 7r6r 会被 4 卡通信、进程调度、复制模块开销反超；当前接入已经加默认 `N_token >= 512` 阈值，避免短序列误走 row-parallel，同时覆盖 7pzb 这类中等 N case。
- 更接近真实性能的 7wux-only reduced run 已完成，配置为 `cycle=10, step=1, sample=1`。完整 `step=200` 仍会被 diffusion replicated work 稀释；但默认 `N_sample=5` sample parallel 已验证无明显收益，后续应看 diffusion transformer 内部并行。

7wux-only reduced 结果：

| 模式 | model forward | seed/job total | 加速比 |
|---|---:|---:|---:|
| 单卡 | 172.13s | 181.06s | 1.00x |
| 4 卡 row-parallel | 75.04s | 83.89s | 2.29x / 2.16x |

阈值 sanity：

| Shape | 结果 |
|---|---|
| `N=128, blocks=2` | 默认阈值下回退普通路径，z/s max diff 均为 0 |
| `N=1218, blocks=1` | 仍启用 row-parallel，z/s max diff 均为 0 |

## 方向 3：DiffusionTransformer token parallelism

token-level diffusion transformer 是 diffusion 中最重的子模块：

```text
shape: [N_sample=5, N_token=1218, C=768]
24 blocks * 200 steps
attention_pair_bias ≈ 58.7s
```

潜在 token parallelism：

- shard query/token rows 到多卡；
- replicate 或 shard pair bias `z`；
- 在 block 边界 all-gather 或 all-reduce token activations。

通信风险：

```text
a tensor size ≈ 5 * 1218 * 768 * 4 bytes ≈ 18.7 MB
block boundaries = 24 * 200 = 4800
```

虽然每次同步的 tensor 远小于 Pairformer `z`，但是同步次数非常高。latency 和 launch overhead 可能主导。

建议：

- 在默认 `N_sample=5` sample parallel 已验证无收益后，这条路线成为 diffusion 多卡方向的主要候选；
- 只有在 `~20 MB` tensor 反复 all-gather 的 latency/bandwidth 测清后再重看。

### LightX2V Ulysses SP 并行推理参考：2026-07-03

已 clone 参考项目：

```text
/root/LightX2V
origin: https://github.com/ModelTC/LightX2V.git
commit: 92aa3b36 update infinitetalk a800 configs (#1220)
```

LightX2V 的并行推理配置集中在 `parallel` 字段：

```json
{
  "parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 2
  }
}
```

核心机制：

- `seq_p_size`：DiT token/sequence 维度并行。这里是 sequence parallel / SP，不是 tensor parallel / TP。
- `seq_p_attn_type="ulysses"`：先按 sequence 切输入，再通过 all-to-all 把 `[seq/world, heads, dim]` 转成 `[seq, heads/world, dim]`，本地做完整 attention，最后 all-to-all 回 sequence shard。
- `seq_p_attn_type="ring"`：每个 rank 保留本地 query shard，K/V shard 在 ring 上流动；每轮做局部 attention，并用 log-sum-exp 合并 softmax 结果。
- `cfg_p_size`：把 classifier-free guidance 的 cond/uncond 两支分到不同 rank，最后 all-gather 两个 noise prediction。
- 配置要求 `cfg_p_size * seq_p_size == world_size`，通过 `device_mesh` 建 `cfg_p` 和 `seq_p` 两个进程组。

LightX2V 的模型接入方式：

- transformer 入口前先按 token 维切 `x`，不足 world size 的部分 pad；
- transformer block 内 self-attention 使用 parallel attention；
- MLP/transition/norm 等 token-local 模块直接在本地 shard 上运行；
- transformer 结束后 all-gather token shards，恢复完整输出。

对应到 Protenix：

- Protenix `DiffusionTransformer` 的 `a_token/s_single` 形状是 `[N_sample, N_token, C]`，token-local 的 `ConditionedTransitionBlock` 可以自然 shard。
- 难点在 `AttentionPairBias.standard_multihead_attention`：它需要 `z_pair -> bias [N_sample, heads, N_token, N_token]`。如果只切 query rows，每个 rank 需要本地 rows 的 bias；如果做 Ring，还要在每轮 K/V shard 对应的 key range 上切 bias block。
- Ulysses 要求 head 数能被 `seq_p_size` 整除。Protenix token diffusion `n_heads=16`，4 卡可行；但 C500 上 all-to-all 成本和 layout 转换需要实测。
- Ring 不要求 head 切分，通信是 K/V shard 环传，理论上更适合先做 POC；但需要实现带 pair bias block 的在线 softmax 合并。

建议路线：

1. 先做 synthetic `DiffusionTransformerBlock` token-parallel benchmark，不直接改完整 Protenix pipeline。
2. 第一版优先验证 Ring-style attention，因为它对 head/layout 假设少，且能直接支持 query-row shard 输出。
3. 如果 Ring POC 性能不够，再评估 Ulysses all-to-all 路径。
4. CFG parallel 不优先，因为 Protenix 没有视频 CFG cond/uncond 双分支结构。

### DiffusionTransformer Ring POC：2026-07-03

新增 standalone benchmark：

```text
benchmarks/bench_diffusion_transformer_ring_parallel.py
```

实现范围：

- 实例化真实 `DiffusionTransformerBlock`，复用真实权重结构；
- 不修改 Protenix pipeline；
- rank0 广播 block 参数、buffer 和输入，确保 correctness 只检验并行算法；
- 沿 token row 维度切 `a/s`；
- 每个 rank 保留本地 query rows，K/V shard 通过 ring P2P 轮转；
- 每轮只投影当前 query rows 与当前 key block 的 pair bias；
- 使用 log-sum-exp 合并分块 softmax；
- transition block 在本地 row shard 上运行；
- block 末尾 all-gather 恢复完整 token activation。

重要修正：

- Protenix `Attention._prep_qkv` 会先把 `q` 除以 `sqrt(c_hidden)`，随后 SDPA 使用 `scale=1.0`；POC 需要保持这个语义。
- Q/K/V tensor layout 是 `[B, N, H, C]`，padding 必须沿 token 维 `dim=1`，不能按倒数第二维补齐。
- 多卡 correctness 不能依赖每张卡本地 seed 后初始化完全一致，必须从 rank0 广播参数和输入。

4GPU C500 实测：

| Shape | Full manual block | Ring parallel block | 加速比 | Correctness |
|---|---:|---:|---:|---|
| `N=128, N_sample=1` | 495.996 ms | 108.261 ms | 4.58x | max diff `3.0e-5`, mean diff `3.0e-6` |
| `N=512, N_sample=1` | 1.520 ms | 3.664 ms | 0.41x | max diff `3.0e-5`, mean diff `2.0e-6` |
| `N=1218, N_sample=1` | 5.233 ms | 4.577 ms | 1.14x | max diff `3.1e-5`, mean diff `2.0e-6` |
| `N=1218, N_sample=5` | 27.387 ms | 15.675 ms | 1.75x | max diff `3.5e-5`, mean diff `2.0e-6` |
| `N=1218, N_sample=5`, stable `iters=10` | 27.342 ms | 21.235 ms | 1.29x | 未重复 check |

当前结论：

- Ring-style token parallel 的数学路径已经跑通，分块 softmax + pair bias block correctness 正常。
- 在真实默认 `N=1218, N_sample=5` 下有正收益，但第一版收益只有约 `1.3x` 到 `1.8x`，不接近 4 卡理想扩展。
- naive POC 每轮 ring 都同步等待 K/V P2P，且每个 key block 都重新投影 pair bias；这些开销会吃掉大量收益。
- 该结果支持继续研究 diffusion transformer 内部并行，但还不支持直接接入 Protenix 作为默认路径。

后续优化方向：

- 做 K/V ring 通信与当前 block matmul/softmax 的 overlap；
- 缓存或预投影 `z -> bias heads`，评估用显存减少 block 内重复 pair bias projection；
- 评估 Ulysses all-to-all 路径，利用 `n_heads=16` 可被 4 卡整除的条件，比较 all-to-all 与 ring P2P；
- 把 benchmark 拆出 attention-only 和 transition-only timing，确认收益主要来自哪一段；
- 若接入 Protenix，先加显式 env gate，不能影响默认单卡路径。

### DiffusionTransformer Ulysses SP POC：2026-07-03

新增 standalone benchmark：

```text
benchmarks/bench_diffusion_transformer_ulysses_sp.py
```

命名说明：

- 该方向应称为 Ulysses sequence parallel / SP，不是 TP。
- 权重没有按 tensor dimension 切分；切的是 token/sequence activation。
- attention 前后通过 all-to-all 在两种布局间转换：

```text
[B, N/world, H, C]  --seq2head all-to-all-->  [B, N, H/world, C]
[B, N, H/world, C]  --head2seq all-to-all-->  [B, N/world, H, C]
```

实现范围：

- 实例化真实 `DiffusionTransformerBlock`，复用真实权重结构；
- 不修改 Protenix pipeline；
- rank0 广播 block 参数、buffer 和输入；
- `N` 不能被 world 整除时补齐到 `ceil(N/world) * world`，attention 对 padded key 加 mask，最后裁掉 padding；
- 本地 token shard 做 Q/K/V projection；
- all-to-all 后每张卡拥有 full padded sequence、`H/world` 个 heads；
- pair bias 只计算当前 rank 的 head shard；
- attention 后 all-to-all 回本地 token shard、full heads，再跑 gate/output projection 和 transition。

4GPU C500 实测：

| Shape | Actual Protenix block | Full manual block | Ulysses SP block | 加速比 vs actual | Correctness |
|---|---:|---:|---:|---:|---|
| `N=128, N_sample=1` | 首次调用 521.192 ms | 2.131 ms | 9.700 ms | 不参考 | max diff `0` |
| `N=512, N_sample=1` | 未测 | 1.479 ms | 1.628 ms | 0.91x vs manual | max diff `0` |
| `N=1218, N_sample=1` | 未测 | 5.401 ms | 4.668 ms | 1.16x vs manual | max diff `2.6e-5` |
| `N=1218, N_sample=5` | 未测 | 27.492 ms | 18.272 ms | 1.50x vs manual | max diff `2.8e-5` |
| `N=1218, N_sample=5`, stable `iters=10` | 27.178 ms | 27.230 ms | 18.594 ms | 1.46x | max diff `2.8e-5` |

当前结论：

- Ulysses SP POC correctness 已通过，数值比 Ring POC 更干净，因为没有分块 softmax 合并误差。
- 对真实默认 `N=1218, N_sample=5`，Ulysses SP 单 block 约 `1.46x` 加速，优于 naive Ring 稳定版的约 `1.29x`。
- actual Protenix block 和 manual full block 时间基本一致，说明 benchmark baseline 可信。
- 但这个收益仍明显低于 LightX2V 视频 DiT 中常见的高扩展比，原因很可能是 Protenix diffusion attention 带有 dense pair bias：
  - 每个 block 都有 `z -> bias [B,H,N,N]`；
  - 当前 POC 虽只算 `H/world` heads，但仍对完整 `z[B,N,N,Cz]` 做 layernorm；
  - all-to-all 后每卡的 attention head 数只有 4，单卡 matmul 粒度变小，kernel efficiency 可能下降。

分段 isolated profile：

```text
N=1218, N_sample=5, world=4, warmup=2, iters=10
ulysses_sp_block_ms = 18.121 ms
```

| Stage | Time | Share |
|---|---:|---:|
| `bias_local_heads` | 14.931 ms | 82.45% |
| `attention_core` | 1.421 ms | 7.84% |
| `conditioned_transition` | 0.472 ms | 2.61% |
| `local_adaln_qkv` | 0.336 ms | 1.86% |
| `seq2head_qkv_all_to_all` | 0.323 ms | 1.79% |
| `final_all_gather` | 0.289 ms | 1.60% |
| `output_gate_projection` | 0.186 ms | 1.03% |
| `head2seq_all_to_all` | 0.151 ms | 0.84% |

解释：

- Ulysses SP 的瓶颈不是 all-to-all，也不是 attention matmul/softmax，而是 dense pair bias 生成。
- `bias_local_heads` 包含 `layernorm_z(z)` 和只投影本 rank 负责的 `H/world` 个 heads。
- 这说明 Protenix diffusion 与视频 DiT 的主要差别在于 dense pair bias；如果每一步都重算 `z -> bias`，Ulysses SP 的扩展比会被明显限制。

Bias cache 上限实验：

```text
N=1218, N_sample=5, world=4, warmup=2, iters=10, --cache-bias
actual_block_ms        = 27.222 ms
single_manual_block_ms = 27.223 ms
ulysses_sp_block_ms    =  3.210 ms
check_max_diff         =  2.9e-5
```

isolated profile:

| Stage | Time | Share |
|---|---:|---:|
| `attention_core` | 1.409 ms | 44.66% |
| `conditioned_transition` | 0.467 ms | 14.78% |
| `local_adaln_qkv` | 0.334 ms | 10.58% |
| `seq2head_qkv_all_to_all` | 0.328 ms | 10.39% |
| `final_all_gather` | 0.291 ms | 9.24% |
| `output_gate_projection` | 0.194 ms | 6.14% |
| `head2seq_all_to_all` | 0.133 ms | 4.21% |
| `bias_local_heads` | 0.000 ms | 0.00% |

结论：

- 如果 `z -> local head bias` 能跨 diffusion steps 复用，单 block 上限从 `18.1 ms` 降到 `3.2 ms`，相对原始 block 约 `8.5x`。
- 对 200 diffusion steps，这个 cache 很有价值：每个 block 的 bias 预计算一次，随后 200 次 step 复用。
- 以当前 synthetic shape 估算，单 rank 每个 block 的 fp32 local bias 大约：

```text
N_sample * ceil(N/4)*4 * ceil(N/4)*4 * (H/4) * 4 bytes
= 5 * 1220 * 1220 * 4 * 4 bytes
≈ 119 MB / rank / block
```

- 24 个 blocks 全部缓存为 fp32 约 `2.9 GB/rank`，bf16 约 `1.4 GB/rank`。这在 4 卡上可能可接受，但需要结合真实 Protenix 显存余量确认。
- 该 benchmark 的 `z` 是 synthetic `[N_sample,N,N,Cz]`。真实 Protenix 里 `z` 通常来自 trunk，不随 sample 和 diffusion step 改变；接入前需要确认它是否物理复制到 `N_sample`，还是 stride-0 expand。如果真实 `z` 跨 sample 共享，bias cache 的内存和预计算量还可以进一步下降。

### Protenix DiffusionTransformer 集成 POC：2026-07-03

新增 env-gated 实验路径：

```text
PROTENIX_DIFFUSION_ULYSSES_SP=1
```

改动位置：

```text
/root/Protenix/protenix/model/modules/transformer.py
```

验证 benchmark：

```text
benchmarks/bench_protenix_diffusion_transformer_ulysses_integration.py
```

实现方式：

- 默认路径完全不变；
- 仅当 `PROTENIX_DIFFUSION_ULYSSES_SP=1`、`torch.distributed` 已初始化、`world_size>1`、inference/no-grad、非 local attention、非 `enable_efficient_fusion` 时启用；
- `DiffusionTransformer.forward` 开始时按 token row 切 `a/s`；
- 在 24 个 transformer blocks 内保持 local token shard；
- 每个 block 内执行 Ulysses SP attention：

```text
[B, N/world, H, C] -> all-to-all -> [B, Npad, H/world, C]
```

- 每个 block/rank 缓存 local head bias；
- 最后一个 block 后 all-gather token rows，恢复完整 `a_token` 给 atom decoder；
- cache key 使用 block index、`z` storage ptr、shape、stride、dtype、device、rank/world/rows；
- cache entries 默认上限 `PROTENIX_DIFFUSION_ULYSSES_SP_CACHE_MAX=64`，超过后清空，避免跨输入无限增长。

真实 Protenix 代码确认：

- `DiffusionConditioning.prepare_cache()` 已经在 step 外缓存 `pair_z`；
- `DiffusionModule.forward()` 中 `z_pair = expand_at_dim(z_pair, dim=-4, n=1)` 使用 `torch.expand`；
- 因此 `z_pair` 的 sample 维是 stride-0，不是物理复制；
- 这使得 per-block local head bias cache 的实际内存接近：

```text
1 * Npad * Npad * (H/world) * 4 bytes
≈ 1 * 1220 * 1220 * 4 * 4
≈ 23.8 MB / rank / block
```

24 blocks fp32 cache 约 `571 MB/rank`；bf16 cache 约 `286 MB/rank`。

4GPU module-level 实测：

| Shape | Blocks | Baseline | Ulysses SP cold | Ulysses SP warm | Speedup warm | Correctness |
|---|---:|---:|---:|---:|---:|---|
| `N=128, N_sample=2` | 2 | 2.155 ms | 未记录 | 2.743 ms | 0.79x | max diff `3.9e-5` |
| `N=1218, N_sample=5` | 2 | 62.317 ms | 未记录 | 6.381 ms | 9.77x | max diff `4.3e-5` |
| `N=1218, N_sample=5` | 24 | 754.942 ms | 未记录 | 73.484 ms | 10.27x | max diff `2.17e-4` |
| `N=1218, N_sample=5` | 24 | 748.634 ms | 181.252 ms | 75.336 ms | 9.94x | max diff `2.13e-4` |

解释：

- 小 `N=128` 因通信和分布式开销反超，应在真实接入时加 token 数阈值。
- `N=1218` 的完整 24-block transformer 获得约 `10x` module-level 加速。
- cold cache 一次性约 `181 ms`，但真实 `N_step=200` 中可以跨 diffusion steps 复用，摊销后约 `0.9 ms/step`。
- correctness 误差量级在 fp32 手写 attention/all-to-all 路径可接受范围内，但仍需要端到端结构输出验证。

当前风险：

- 这还是 module-level benchmark，没有跑完整 `sample_diffusion` 和完整 Protenix endpoint；
- cache 当前是 fp32，显存可接受但仍需实测完整 7wux；
- 还没实现 bf16 cache；
- 如果输入切换或 shape 改变，cache 会通过 key miss 重建；默认 max entries 64 足够 24 blocks，但端到端多 target/job 需要注意释放策略。

### sample_diffusion reduced 接入结果：2026-07-03

为避免 4 个 ranks 各自 dataloader/featurization 导致 profile 污染，当时新增两个实验 env：

```text
PROTENIX_DISTRIBUTED_FORWARD_BARRIER=1
PROTENIX_DISTRIBUTED_DATA_BROADCAST=1
```

当前实现语义：

- `PROTENIX_DISTRIBUTED_FORWARD_BARRIER`：在 `runner.predict(data)` 前 barrier 对齐，并重置 model forward 计时；
- `PROTENIX_DISTRIBUTED_DATA_BROADCAST`：rank0 创建 dataloader、featurize batch，再通过 `dist.broadcast_object_list` 把 batch 发给 rank1/2/3；
- 2026-07-06 后，若 `PROTENIX_PAIRFORMER_ROW_PARALLEL=1` 或 `PROTENIX_DIFFUSION_ULYSSES_SP=1` 且 `mp_size>1`，上述两项默认自动启用；可显式设为 `0` 关闭；
- checkpoint 仍然每个 rank 自己 load，这符合多卡推理需求；
- 当前还有一个上游问题：旧 MSA 格式转换发生在 `infer_predict` 之前，仍然会被 4 个 ranks 重复执行；这主要影响 job time，不影响 forward barrier 后的 model forward profile。

测试配置：

```text
input=examples/example_7wux.json
cycle=1
sample=5
triatt_kernel=wmma
trimul_kernel=torch
enable_fusion=False
PROTENIX_DIFFUSION_ULYSSES_SP=1
```

step=5 对比：

| Path | Model forward | sample_diffusion | diffusion_transformer |
|---|---:|---:|---:|
| single-rank baseline | 30.73s | 2.311s | 1.911s |
| 4GPU Ulysses SP + bias cache + barrier+broadcast | 29.51s rank0 | 0.868s avg / 0.898s max | 0.101s avg per call |

解释：

- `diffusion_transformer` 大幅下降，说明 module-level 优化已经转化到真实 `sample_diffusion`；
- step=5 下 diffusion 本身占比太小，整体 model forward 只小幅下降；
- `sample_diffusion` 从 `2.31s` 到约 `0.87s`，约 `2.7x`。

step=40 结果：

历史 single-rank baseline：

```text
sample_diffusion ≈ 19.402s
```

4GPU Ulysses SP + bias cache + barrier+broadcast：

| Metric | Time |
|---|---:|
| model forward rank0 | 35.15s |
| sample_diffusion avg / max | 6.553s / 6.689s |
| diffusion_transformer avg per call | 0.095s |
| atom_attention_encoder avg per call | 0.032s |
| atom_attention_decoder avg per call | 0.029s |

解释：

- `sample_diffusion` 从约 `19.4s` 降到约 `6.6s`，约 `3.0x`；
- `diffusion_transformer` 已经不是主要瓶颈，后续 diffusion 内剩余时间更多在 atom encoder/decoder、sampler loop 和同步开销；
- 如果跑完整 `step=200`，bias cache 的 cold build 会进一步摊薄，但 atom encoder/decoder 和 sampler loop 会成为扩展上限。

后续优化方向：

- 将旧 MSA 格式转换也改成 rank0-only，避免 4 ranks 重复写 `example-update-msa.json`；
- 继续跑 `step=200` 或更完整 reduced，确认 `sample_diffusion` 约 `3x` 是否稳定；
- 加 `N_token` 阈值，避免小 shape 误走 Ulysses SP；
- 优先尝试 bf16 bias cache，评估 correctness 和显存；
- 组合 PairformerStack row-parallel，评估 Pairformer + diffusion 两条多卡路径的端到端收益；
- 对比 `enable_efficient_fusion=True` 的原始 Protenix block，以及 Ulysses 下的局部 head bias fusion；
- 若继续接入 Protenix，先以 env gate 做单 block/单 diffusion transformer POC，不直接改默认路径。

## 通信 microbench

Pairformer row-parallel 或 diffusion token-parallel 之前，需要测：

1. 4 卡 island 内 `380 MB` bf16-equivalent payload all-gather；
2. 8 卡跨 island `380 MB` all-gather；
3. 4 卡 island 内 `20 MB` payload all-gather/all-reduce，重复大量次数；
4. 8 卡跨 island `20 MB` 版本；
5. 从 GPU0 broadcast read-mostly trunk tensors 到其他 GPU；
6. full Pairformer-like row-parallel loop，用 synthetic compute 加重复 `z` all-gather，测同步频率，而不是只测单次 triangle attention call。

判断标准：

- 如果 4 卡 island 内 `380 MB` all-gather 明显低于被节省的每模块 compute，Pairformer row-parallel 值得正式 POC。
- 如果跨 island 带宽弱很多，Pairformer 保持在单个 4 卡 island 内，第二个 island 用于另一个 seed/job/sample batch。

### Collective microbench 结果：2026-07-03

新增 benchmark：

```text
benchmarks/bench_c500_collectives.py
```

测量说明：

- `all_gather --total-mb 380` 表示重建后的完整 tensor 是 `380 MB`；每个 rank 贡献 `380 / world_size MB`。这匹配 row-sharded Pairformer `z` reconstruction。
- dtype: bf16。
- backend：C500 容器内 torch distributed NCCL 路径。
- 该容器里不能用 `torchrun --standalone`；需要显式使用 `--master_addr=127.0.0.1 --master_port=...`。

4 卡结果：

| Operation | 逻辑 payload | 平均 max-rank 时间 | 近似每 rank traffic | 近似带宽 |
|---|---:|---:|---:|---:|
| all_gather | 380 MB | 2.534 ms | 570 MB | 219.63 GB/s |
| all_gather | 20 MB | 0.383 ms | 30 MB | 76.58 GB/s |
| broadcast | 380 MB | 2.953 ms | 380 MB | 125.67 GB/s |
| broadcast | 20 MB | 0.279 ms | 20 MB | 69.93 GB/s |
| all_reduce | 380 MB | 3.481 ms | 570 MB | 159.93 GB/s |
| all_reduce | 20 MB | 0.265 ms | 30 MB | 110.65 GB/s |

8 卡 all-gather 结果：

| Operation | 逻辑 payload | 平均 max-rank 时间 | 近似每 rank traffic | 近似带宽 |
|---|---:|---:|---:|---:|
| all_gather | 380 MB | 6.513 ms | 665 MB | 99.71 GB/s |
| all_gather | 20 MB | 0.848 ms | 35 MB | 40.29 GB/s |

重复 full-`z` synchronization：

| GPUs | Operation | Iterations | 平均 max-rank 时间 | Total wall time |
|---:|---|---:|---:|---:|
| 4 | all_gather 380 MB | 240 | 2.510 ms | 0.66s |
| 8 | all_gather 380 MB | 240 | 6.436 ms | 1.65s |

解释：

- 4 卡 island 内通信足够快，单次 full-`z` reconstruction 不是阻塞点。
- `380 MB` 场景下，8 卡 all-gather 比 4 卡慢约 2.5x，这确认了 4 卡 island-first 仍是更稳妥的默认策略。
- 即使用粗略模型 `48 Pairformer blocks * 5 full-z syncs/block = 240 syncs` 估算，4 卡 raw all-gather 也只有约 `0.66s`，8 卡约 `1.65s`。相比实测 `~162s` Pairformer/trunk runtime 很小。
- 因此 Pairformer row-parallel 的剩余风险不是 raw collective bandwidth，而是集成开销：module-boundary synchronization、tensor layout conversions、residual update placement、duplicated linear/projection work、Python/torch dispatch overhead。

## 当前推荐路线

### P0：保持单卡 kernel 稳定

- triangle attention 继续使用 v9。
- triangle multiplicative update 继续使用 torch/mcBLAS；PyTorch-level matmul 重排已验证无稳定收益。
- diffusion 质量配置保持默认：`N_step=200`、`N_sample=5`。

### P1：4GPU Pairformer row-parallel + Diffusion Ulysses SP

- 这是当前主线，使用单个 4 卡 island。
- Pairformer 沿 token row 维切分 z-heavy 子图，`PROTENIX_PAIRFORMER_ROW_PARALLEL=1` opt-in，默认只在 `N_token >= 512` 时启用。
- DiffusionTransformer 使用 Ulysses sequence parallel，`PROTENIX_DIFFUSION_ULYSSES_SP=1` opt-in，配合 rank-local pair-bias cache。
- 协同推理 `mp_size>1` 时，rank0/MP leader 默认执行旧 MSA 转换、input preprocessing 和 dataloader featurization，再广播 batch 给其他 rank；forward 前 barrier 也默认启用。两者可分别用 `PROTENIX_DISTRIBUTED_DATA_BROADCAST=0` / `PROTENIX_DISTRIBUTED_FORWARD_BARRIER=0` 显式关闭。
- 当前 4GPU 性能优先配置使用 `--enable_fusion=False`；完整三组默认配置 no-profile run：总 job `157.56s`，最大 7wux model forward `85.79s`。

### P2：2GPU/4GPU 使用建议

- 2GPU：开启 Diffusion SP 后，7wux-only forward `125.09s`；完整三 case total job 仍需重跑。
- 4GPU：总 job `157.56s`，7wux forward `85.79s`；适合大 N 或低延迟目标。
- 8GPU：暂不作为默认路线。8 卡 benchmark 仍有收益，但跨 island gather 从 4 卡约 `2.5ms` 增到约 `6.5ms`，增量收益明显变小。

### 暂停或废弃方向

- 默认 `N_sample=5` sample parallel：真实 Protenix 已 batch 化执行 samples，step=40 下 `sample_diffusion` 基本不变，代码已撤回。
- Pairformer row micro-batch overlap：真实 profile 中 gather 约 `15.8% - 17.4%`，稳态 gather 多数为 `2-4ms`，切更细可能增加 collective latency。
- 更大范围 row-sharded state / 减少 full-`z` 同步次数：理论上可能减少通信和内存流量，但 `tri_mul_in`、`tri_att_end`、下一个 block 都依赖更新后的 full `z`，需要重构 Pairformer 数据流，当前只记录方向，不实施。
- PyTorch-level tri-mul matmul POC：数学正确但没有降低 compute，实验代码已移除；后续若继续优化 tri-mul compute，应转向专用 fused kernel / mctlass-style MMA。

2026-07-03 进一步更新：PairformerBlock subset 已经测得 4 卡 `2.88x`。这说明正式接入 4 卡 Pairformer row-parallel 的优先级可以上升；8 卡虽然还能继续变快，但跨 island gather 成本明显更高，不应作为第一版目标。

2026-07-03 再更新：真实 `PairformerBlock.forward` 实验分支已经完成 opt-in 接入，`PairformerStack` 48-block synthetic full-stack 已验证，`runner/batch_inference.py` 4 卡 smoke 也已跑通。7wux-only reduced 性能 run 和默认 N 阈值启用策略也已完成。

2026-07-03 diffusion sample parallel 更新：真实 Protenix 默认 `sample_diffusion_chunk_size=5` 下，`N_sample=5` 已经 batch 化执行。实验性 sample split 在 `step=40` 下没有带来收益，代码已撤回；下一步不建议继续这个方向。

2026-07-03 DiffusionTransformer Ulysses SP 更新：参考 LightX2V 的 Ulysses sequence parallel 思路，已完成实验性 opt-in 接入，开关为 `PROTENIX_DIFFUSION_ULYSSES_SP=1`。实现不是 TP，而是沿 token/sequence 维切分 attention 的 Q token，每个 rank 保留全量 K/V 上下文，attention 输出按 token row shard 计算后 all-gather 回完整 token activations。`z -> pair bias` 的 local heads 计算加入 rank-local cache，避免在 diffusion 每步重复重算。模块级 `N=1218, N_sample=5, blocks=24` 下，单卡 baseline 约 `748.6 ms`，4 卡 Ulysses SP warm-cache 约 `75.3 ms`，max diff 约 `2.13e-4`。

2026-07-03 端到端多卡更新：`runner/batch_inference.py` 已增加实验性 rank0-only input preprocessing，复用 `PROTENIX_DISTRIBUTED_DATA_BROADCAST=1` 作为开关。多 rank 下只有 rank0 执行旧 MSA format 转换、`preprocess_input()` 和 dataloader featurization，随后广播 updated JSON 路径与 batch object 给其他 rank。`examples/example_7wux.json` 日志确认旧 MSA 转换只出现一次，rank1/2/3 只接收 `/root/Protenix/examples/example_7wux-update-msa.json`。

2026-07-06 更新：data broadcast 和 forward barrier 在 Pairformer row-parallel 或 Diffusion Ulysses SP 且 `mp_size>1` 时默认自动开启；下面历史命令中的两个显式 env 现在可以省略。若需要调试普通 dataloader 或 barrier 卡住问题，可显式设置 `PROTENIX_DISTRIBUTED_DATA_BROADCAST=0` 或 `PROTENIX_DISTRIBUTED_FORWARD_BARRIER=0`。

同一轮 4GPU reduced end-to-end 使用：

```bash
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512
PROTENIX_DIFFUSION_ULYSSES_SP=1
WMMA_VERSION=v9
--input=examples/example_7wux.json
--triatt_kernel=wmma
--trimul_kernel=torch
--cycle=1
--sample=5
--enable_fusion=False
```

`step=5` 结果：

| Metric | 4GPU rank 平均 / rank0 |
|---|---:|
| rank0 model forward | `14.47s` |
| rank0 job | `31.16s` |
| get_pairformer_output | `7.47s` avg |
| sample_diffusion | `0.93s` avg |
| confidence_head | `3.98s` avg |

`step=40` 结果：

| Metric | 4GPU rank 平均 / rank0 |
|---|---:|
| rank0 model forward | `19.54s` |
| rank0 job | `37.54s` |
| get_pairformer_output | `7.44s` avg |
| sample_diffusion | `6.04s` avg |
| confidence_head | `3.96s` avg |
| diffusion_transformer | `83.2 ms/step/rank` avg |

对比上一版只有 Diffusion Ulysses SP、没有 Pairformer row-parallel 的 `step=40` reduced run，rank0 model forward 从约 `35.15s` 降到 `19.54s`。对比单卡 reduced baseline，`sample_diffusion` 从约 `19.40s` 降到 `6.04s`，约 `3.2x`；`get_pairformer_output` 从约 `16.7s` 降到 `7.44s`，约 `2.25x`。当前 reduced end-to-end 的主要剩余大头变成 confidence/head 相关路径和固定的 CPU/模型加载/featurization wall time。

2026-07-06 完整三组端到端 no-profile 更新：`examples/example.json` 三组经典 case 已在 2GPU、4GPU 完整默认配置下跑通；1GPU 在关闭 profile 后 7r6r/7wux 已回到历史水平，但完整三 case 到 7pzb 时仍偶发卡住，判断为 GPU0/连续运行状态问题，不作为 production wall time。配置为 `cycle=10, step=200, sample=5, triatt=wmma, trimul=torch, enable_cache=True, enable_fusion=True`。2GPU/4GPU 开启 Pairformer row-parallel、Diffusion Ulysses SP、rank0 input preprocessing/dataloader broadcast。测试前重启测试容器并 warm reset GPU0-3，避免 GPU0/1 残留上下文影响结果。

注意：`PROTENIX_PROFILE_LOG` 会在主模型阶段插入多次 `torch.cuda.synchronize()`，只能用于阶段归因，不能作为 production wall time。正式性能结论使用 no-profile run；profile run 的分阶段数字保留为解释 Pairformer/diffusion 占比。

4GPU no-profile rank0 结果：

| Case | N_token | rank0 model forward |
|---|---:|---:|---|
| 7r6r | 245 | `22.96s` |
| 7wux | 1218 | `130.96s` |
| 7pzb | 600 | `41.98s` |

整体 rank0 job time 为 `218.98s`。作为对照，1GPU no-profile 在同一代码下 7r6r/7wux 分别为 `22.99s` / `248.25s`，与历史 `22.82s` / `248.80s` 对齐；单独 7r6r no-profile 为 `22.69s`。因此当前支持多卡后的 world_size=1 路径没有暴露系统性性能退化。Profile run 显示 Pairformer 随卡数下降明显，但 `sample_diffusion` 在完整默认路径里基本维持在 `~74s`。这组数据对应旧代码：当时 Diffusion Ulysses SP 没有覆盖默认 `enable_fusion=True` 的 fused pair-bias 路径。

2026-07-06 Diffusion fused Ulysses SP 更新：当前已补齐 `enable_fusion=True` 的 local-head pair-bias 生成。默认 fused path 中的 `z` 是 channel-first normalized pair feature：`[B, C_z, N, N]`；每个 rank 只取本 rank 负责的 attention heads，对 `linear_nobias_z.weight[head_start:head_end] * layernorm_z.weight` 做 1x1 conv2d，生成 local-head bias。后续仍按 token row 切分 `a/s`，用 packed QKV all-to-all 在 seq/head 间转换，attention 后 all-gather 回完整 token activations。

module-level 2GPU correctness：

```text
DiffusionTransformer(c_a=64,c_s=32,c_z=16,n_blocks=2,n_heads=8)
N=37, B=2, enable_fusion=True
fused baseline vs fused SP: max_diff=6.80e-06, mean_diff=2.98e-08
```

4GPU 7wux reduced 对比：

```text
cycle=1
step=40
sample=5
triatt_kernel=wmma
trimul_kernel=torch
enable_fusion=True
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512
PROTENIX_DISTRIBUTED_FORWARD_BARRIER=1
PROTENIX_DISTRIBUTED_DATA_BROADCAST=1
```

| Diffusion SP | rank0 model forward | rank0 job | `get_pairformer_output` avg | `sample_diffusion` avg | profile total avg |
|---|---:|---:|---:|---:|---:|
| off | `30.50s` | `48.47s` | `5.86s` | `19.27s` | `29.23s` |
| on | `21.53s` | `37.62s` | `5.80s` | `10.34s` | `20.28s` |

结论：fused SP 对 `sample_diffusion` 的 reduced 收益约 `1.86x`，并且 Pairformer 时间基本不变，归因清楚。这个收益小于早期 `enable_fusion=False` reduced 路径的约 `3x`，说明默认 fused path 中仍有未并行/固定开销，例如 atom encoder/decoder、sampler loop、confidence 及框架调度。

随后补跑完整默认三 case no-profile A/B，配置均为：

```text
cycle=10
step=200
sample=5
triatt_kernel=wmma
trimul_kernel=torch
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512
PROTENIX_DIFFUSION_ULYSSES_SP=1
all profile env disabled
```

| `enable_fusion` | Total job | 7r6r forward | 7wux forward | 7pzb forward |
|---|---:|---:|---:|---:|
| `True` | `160.75s` | `21.08s` | `86.02s` | `30.61s` |
| `False` | `157.56s` | `17.94s` | `85.79s` | `30.58s` |

判断：

- `enable_fusion=False` 在三 case 上没有回退，总 job 比 `True` 快 `3.19s`，约 `2.0%`。
- 差距主要来自 7r6r；7wux 和 7pzb 基本持平。
- 因收益较小，暂不做 `world_size>1 && diffusion_sp=1` 时的代码自动切换，避免隐式改变 Protenix 默认语义。
- 当前 4GPU SP 性能优先命令建议显式使用 `--enable_fusion=False`。

2026-07-06 auto broadcast/barrier 更新：当 `PROTENIX_PAIRFORMER_ROW_PARALLEL=1` 或 `PROTENIX_DIFFUSION_ULYSSES_SP=1` 且 `mp_size>1` 时，`PROTENIX_DISTRIBUTED_DATA_BROADCAST` 和 `PROTENIX_DISTRIBUTED_FORWARD_BARRIER` 默认自动开启；推荐命令不再需要显式设置这两个 env。若需要调试，可分别设置为 `0` 显式关闭。

验证配置：

```text
input=single-case json for 7r6r / 7wux / 7pzb
cycle=10
step=200
sample=5
triatt_kernel=wmma
trimul_kernel=torch
enable_fusion=False
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512
PROTENIX_DIFFUSION_ULYSSES_SP=1
PROTENIX_DISTRIBUTED_DATA_BROADCAST unset
PROTENIX_DISTRIBUTED_FORWARD_BARRIER unset
all profile env disabled
```

日志确认 non-leader ranks 收到 preprocessed JSON 和 MP-group metadata，说明自动 data broadcast 生效。

single-case no-profile 结果：

| Case | GPUs | rank0 model forward | rank0 job |
|---|---:|---:|---:|
| 7r6r | 2 | `17.33s` | `24.01s` |
| 7r6r | 4 | `16.38s` | `22.70s` |
| 7wux | 2 | `125.09s` | `141.83s` |
| 7wux | 4 | `86.66s` | `103.04s` |
| 7pzb | 2 | `39.50s` | `49.84s` |
| 7pzb | 4 | `30.66s` | `40.51s` |

判断：

- 4GPU 7wux 与前序 `enable_fusion=False` 7wux `85-87s` 区间一致，证明 auto broadcast/barrier 没有引入性能回退。
- 2GPU 开启 Diffusion SP 后，7wux single-case forward 从旧三 case数据中的 `162.50s` 降到 `125.09s`，说明 diffusion SP 对 2GPU 也有效。
- 7r6r 是短序列，2GPU/4GPU 差距很小；4GPU 性能略好但卡时不划算。
- 7pzb 从 2GPU 到 4GPU 仍有收益，forward `39.50s -> 30.66s`。

2026-07-06 1/2/4 卡性价比更新：2GPU/4GPU 配置与 1GPU 相同，只增加协同推理环境变量并把 `torchrun --nproc_per_node` 分别设置为 `2` / `4`。2GPU/4GPU 日志确认 rank0-only MSA/preprocess 和 dataloader broadcast 生效。所有 profile env 均关闭，包括 `PROTENIX_PROFILE_LOG`、`TRIATT_PROFILE_LOG`、`PAIRFORMER_PROFILE_LOG`、`PAIRFORMER_TRIMUL_PROFILE_LOG`、`DIFFUSION_PROFILE_LOG`、`DIFFUSION_TRANSFORMER_PROFILE_LOG`。

| GPUs | Total job | 7r6r forward | 7wux forward | 7pzb forward | Total speedup vs 1GPU | 7wux speedup vs 1GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | n/a | `22.99s` | `248.25s` | n/a | n/a | `1.00x` |
| 2 | `252.75s` | `22.66s` | `162.50s` | `46.22s` | n/a | `1.53x` |
| 4 | `218.98s` | `22.96s` | `130.96s` | `41.98s` | n/a | `1.90x` |

1GPU full no-profile run 的 7pzb 在连续三 case 中再次卡在 forward 入口后，GPU0 保持显存占用但无完成日志；相同代码下单独 7pzb no-profile 可完成，forward `61.73s`。因此当前只用 1GPU 的 7r6r/7wux 作为单卡退化判断，用 2GPU/4GPU 完整 job 作为多卡 production 对比。

保留一组带 top-level profile 的阶段归因数据：

关键 profile 对比：

| Case | GPUs | Pairformer / `get_pairformer_output` | `sample_diffusion` | 说明 |
|---|---:|---:|---:|---|
| 7wux | 1 | `163.56s` | `74.24s` | 单卡 baseline |
| 7wux | 2 | `~82.9s` | `~74.0s` | Pairformer 约 `1.97x`，diffusion 基本不降 |
| 7wux | 4 | `~51.9s` | `~74.0s` | Pairformer 约 `3.15x`，diffusion 仍是主要剩余大头 |
| 7pzb | 1 | `28.04s` | `26.16s` | 单卡 baseline |
| 7pzb | 2 | `~18.54s` | `~26.0s` | 中等 N 下 Pairformer 有收益，diffusion 不降 |
| 7pzb | 4 | `~14.13s` | `~26.0s` | 4GPU 仍有收益，但被 diffusion 固定耗时限制 |

性价比判断：

- 2GPU 相比 1GPU 有稳定收益，但主要来自长序列 7wux；小 N 的 7r6r 几乎没有收益。
- 4GPU 相比 2GPU 仍有收益，尤其 7wux 从 `162.50s` 降到 `130.96s`，但增量收益已经变小。
- 如果目标是“单 job 最短延迟”，4GPU 是当前最好选择。
- 如果目标是“吞吐/卡时性价比”，2GPU 更均衡；4GPU 的完整 job wall time 最短，但卡数增长后的边际收益递减，适合大 N 或延迟敏感场景。

输出一致性检查：

- seed 均为 `101`，比较对象为完整的 5 个 dumped samples：summary confidence JSON 和 CIF atom coordinates。
- 直接按 `sample_0..4` 对齐，1/2/4 卡输出都不是 bitwise 或结构一致。
- 对每个 case 内 5 个 samples 做最优匹配后，2GPU vs 4GPU 的 summary 差异小于对单卡差异，但 CIF 坐标仍有明显差异。例如 2GPU vs 4GPU：
  - 7r6r：summary mean diff `0.365`，coordinate mean diff `9.99 A`
  - 7wux：summary mean diff `0.737`，coordinate mean diff `7.96 A`
  - 7pzb：summary mean diff `1.179`，coordinate mean diff `12.33 A`
- 因此当前端到端 dumped structure 不能作为“跨并行度严格一致”的证据。更合理的解释是 diffusion/sample dumping 路径跨运行不具备 bitwise deterministic trajectory；已有 module-level correctness 仍然成立，但端到端结构一致性需要额外设计 deterministic 对齐测试。

2026-07-03 deterministic 精度复查：

- 默认 `configs.deterministic=False`，之前完整性能测试没有打开 PyTorch deterministic algorithms。
- 单卡最小 case 打开 `configs.deterministic=True` 后，`7r6r, seed=101, cycle=1, step=5, sample=5` 连续两次运行输出完全一致：`diff -qr` 无差异，5 个 CIF 和 5 个 summary JSON 的 SHA256 完全相同。
- 进一步用更小的计算精度检查配置 `cycle=1, step=1, sample=5` 对比 1GPU 与 2GPU：
  - 1GPU forward `6.84s`；
  - 2GPU rank0 forward `6.53s`，rank1 forward `6.29s`；
  - 2GPU 同配置重复两次输出 byte-wise 一致，说明当前多卡路径自身在 deterministic 打开后是可复现的；
  - 1GPU vs 2GPU 不 byte-wise 一致，summary 核心分数差异较小：`ranking_score` 约 `2e-4` 到 `1.9e-3`，`ptm` 约 `2e-4` 到 `1.6e-3`，`iptm` 约 `2e-4` 到 `2.3e-3`，`has_clash` 一致为 `False`。
- `step=1` 下 CIF 坐标不是好的端到端结构 oracle：扩散远未收敛，坐标本身可到千级绝对值，按 sample index 直接比较会放大轨迹差异。因此当前判断是：多卡路径没有暴露随机漂移；1GPU 与 2GPU 因 Ulysses SP / row-parallel 改变计算拆分和规约顺序，不应期待 dumped structure bitwise 等价。后续若要判断结构等价，应使用更合理的 `step=40` 或 `step=200`，并比较 summary/结构指标，而不是用 `step=1` raw CIF 坐标下结论。

2026-07-04 DP x MP deterministic 复查：

- 新增 `PROTENIX_INFERENCE_MP_SIZE` 后，验证了 `world_size=4, mp_size=2` 的 `2DP x 2MP` 路径。
- 对照方式不是把 2DP x 2MP 直接和单个 standalone sample 比，而是和等价 pure MP baseline 对齐 sample/RNG 顺序：
  - `7r6r`、`7pzb` 使用同一个 2GPU pure MP 顺序 baseline；
  - `7wux` 使用独立 2GPU pure MP baseline。
- 打开 `configs.deterministic=True` 后，`2DP x 2MP` 输出目录中的 `7r6r`、`7pzb`、`7wux` 与上述 baseline 的 CIF 和 summary JSON SHA256 均一致。
- 结论：在 deterministic 打开、并且 sample 顺序/RNG 顺序匹配时，当前 DP x MP 实现没有引入额外精度偏差；默认非 deterministic 完整性能跑仍不承诺 dumped structure 跨并行度 bitwise 一致。

2026-07-04 多卡剩余效率 POC：

- 针对剩余 `68s Pairformer / 74s diffusion`，先做了两项低风险实现：
  - Pairformer row-parallel 路径中，`pair_transition` 从每卡 full `z` replicated 计算改为本 rank row shard 计算，再 all-gather 回 full `z`。
  - Diffusion Ulysses SP 中，Q/K/V 的 3 次 `_sp_seq2head` all-to-all 合并为一次 packed QKV all-to-all。
- Correctness / smoke：
  - 2GPU deterministic 最小 case `7r6r, cycle=1, step=1, sample=5` 跑通。
  - 修改后 2GPU 输出与修改前 `/tmp/protenix_det_2gpu_7r6r_s1` byte-wise 一致，说明这两项改动没有改变当前 deterministic 多卡输出。
- 性能验证：
  - 公平 reduced 配置：`7wux, cycle=1, step=40, sample=5, enable_fusion=False`，开启 Pairformer row-parallel、Diffusion Ulysses SP、rank0 dataloader broadcast。
  - 修改后 rank0 model forward `19.61s`，top-level profile：
    - `get_pairformer_output`: `7.45-7.52s`
    - `sample_diffusion`: `6.07-6.08s`
    - `confidence_head`: `3.93-3.94s`
  - 这与上一轮 `19.54s / get_pairformer_output 7.44s / sample_diffusion 6.04s` 基本一致，没有可见端到端收益。
  - `cycle=10, step=40` 下 rank0 model forward `79.77s`，其中 `get_pairformer_output ~67.5s`、`sample_diffusion ~6.08s`。这和 `cycle=1` 的 Pairformer 时间近似按 recycle 数放大，符合预期。
- 结论：
  - `pair_transition` replicated compute 和 QKV 三次 all-to-all 不是当前 reduced 端到端的主要瓶颈，至少这两项简单优化收益被噪声/其他开销淹没。
  - 打开 `TRIATT_PROFILE_LOG` 或 `DIFFUSION_TRANSFORMER_PROFILE_LOG` 会显著改变 wall time，不能拿带 block-level/profile sync 的结果和真实 reduced 性能直接对比。
  - 后续若继续做 overlap，应先补 Pairformer row-parallel 的 compute/gather 细粒度 profile，再决定是否做 row micro-batch overlap；目前不建议直接把复杂 async overlap 接入真实推理路径。

2026-07-04 Pairformer row-parallel compute/gather profile：

- 注意：宿主 `/root/Protenix` 与容器 `b62d01a65c5d:/root/Protenix` 是两份文件，不是同一 bind mount。宿主侧修改后，C500 运行前需要显式 `docker cp` 同步到容器。本轮已同步：
  - `protenix/model/modules/pairformer.py`
  - `protenix/model/modules/transformer.py`
- 独立 2GPU `PairformerBlock(c_s=0)` smoke 已验证 `PAIRFORMER_PROFILE_LOG` 能输出 row-parallel profile，字段包括 `*_compute`、`*_gather`、transpose 和 s 分支。
- 真实 7wux profile 配置：
  - `cycle=1, step=1, sample=1`
  - 4GPU，`PROTENIX_PAIRFORMER_ROW_PARALLEL=1`
  - 只开 `PAIRFORMER_PROFILE_LOG` 和 top-level `PROTENIX_PROFILE_LOG`，不开 tri-attn/diffusion-transformer 细粒度 profile。
- top-level profile：
  - rank0 model forward `9.19s`
  - rank0 `get_pairformer_output 7.50s`
  - `sample_diffusion 0.19s`
- row-parallel Pairformer 聚合，256 records：
  - total profiled Pairformer block time：`28.02s` across all ranks，约 `7.0s/rank`
  - compute-like stages：`22.56s` across ranks
  - gather stages：`4.88s` across ranks，`17.4%`
  - transpose：`0.58s` across ranks
  - per-rank gather 占比约 `13.1% - 19.6%`
- gather 细节：
  - `tri_mul_out_gather` 有首个 block 初始化型 outlier，最大约 `553ms`；去掉最大 outlier 后，整体 gather 占比约 `15.8%`。
  - 稳态 `tri_att_start/end_gather` 和 `pair_transition_gather` 多数是 `2-4ms` 级。
- 判断：
  - Pairformer 通信确实可见，但不是主导项；理论最大 overlap 收益约为每 rank `~1.0-1.3s`，且还不可能完全隐藏。
  - row micro-batch 会增加 all-gather 次数，可能把 2-4ms 的小 gather latency 放大；在当前数据下不建议直接实现复杂 async micro-batch overlap。
  - 后续更值得看的方向是减少 compute 本身或减少 full-z 同步次数，而不是把现有 gather 切得更细。
- 同步新代码后的轻量 reduced 性能：
  - `7wux, cycle=1, step=40, sample=5, enable_fusion=False`
  - rank0 model forward `19.12s`
  - top-level profile：`get_pairformer_output 6.91-6.96s`，`sample_diffusion 5.97s`，`confidence_head 4.09s`
  - 相比上一轮 `19.54s / get_pairformer_output 7.44s / sample_diffusion 6.04s` 略好，但属于小幅收益/波动范围；当前不能把它归因成显著优化。

2026-07-04 后续方向收敛：

- 不建议继续做 Pairformer row micro-batch overlap：
  - 真实 7wux Pairformer row-parallel profile 中，gather 约占 `15.8% - 17.4%`，不是主导项。
  - 稳态 gather 多数是 `2-4ms` 级小通信，切成更多 micro-batch 会增加 collective 次数，容易把 latency 放大。
  - 即使理想隐藏全部通信，理论收益也只有每 rank `~1.0-1.3s`，实际无法完全隐藏。
- 更可能有效的两个方向：
  - 减少 Pairformer 计算本身，尤其 `tri_mul_out/in_compute` 和 triangle attention compute；这部分在 7wux profile 里仍是主耗时。
  - 减少 full-`z` 同步次数或同步数据量，但必须沿依赖图谨慎做：`tri_mul_in` 依赖 `tri_mul_out` 后的 full `z`，`tri_att_end` 依赖 `tri_att_start` 后转置的 full `z`，`pair_transition` 后下一 block 也需要 full `z`。因此这不是简单删一个 all-gather，需要更大范围的 row-sharded state 设计。

2026-07-04 Pairformer tri-mul matmul POC：

- 曾实现过一个实验开关 `PROTENIX_PAIRFORMER_TRIMUL_MATMUL=1`，只影响 row-parallel 路径内的 triangle multiplicative contraction。
- POC 把 row shard 的 contraction 从 `torch.einsum` 改写成按 channel 分组的 `torch.matmul`，数学等价；float32/bf16 小张量 outgoing/incoming correctness 均为 `max diff 0.0`。
- 因性能无稳定收益，相关实验代码已移除，当前代码不再保留该开关。
- 4GPU `7wux, cycle=1, step=40, sample=5, enable_fusion=False`：
  - baseline 同步后 rank0 model forward 约 `19.12s`，`get_pairformer_output 6.91-6.96s`，`sample_diffusion 5.97s`。
  - matmul POC rank0 model forward `19.02s`，`get_pairformer_output 6.92-7.43s`，`sample_diffusion 5.94s`。
  - 端到端只有 `~0.1s` 级变化，属于噪声范围。
- 4GPU `7wux, cycle=1, step=1, sample=1` block-level profile：

| Path | Pairformer all-rank total | tri_mul_compute | tri_mul_gather |
|---|---:|---:|---:|
| baseline einsum | `28.02s` | `11.54s` | `3.24s` |
| matmul POC | `27.40s` | `11.92s` | `2.20s` |

- 判断：
  - matmul POC 没有降低 compute，`tri_mul_compute` 反而增加约 `0.39s` across ranks。
  - 总时间减少主要来自 gather 抖动/首轮 outlier 差异，不是稳定的计算优化。
  - 该方向作为负结果记录，不进入代码主线；后续如果要优化 tri-mul compute，应转向专用 fused kernel / mctlass-style MMA，而不是 PyTorch-level matmul 重排。

2026-07-04 cleanup 回归：

- Protenix 代码中已移除 `PROTENIX_PAIRFORMER_TRIMUL_MATMUL` 和相关 helper，容器内 `grep` 确认无残留。
- `py_compile` 覆盖当前改动文件通过。
- 2GPU 7wux 最小回归通过：
  - `cycle=1, step=1, sample=1, enable_fusion=False`
  - rank0 model forward `12.50s`，rank1 model forward `12.33s`
- 4GPU 7wux reduced 性能回归通过：
  - `cycle=1, step=40, sample=5, enable_fusion=False`
  - rank0 model forward `19.02s`
  - top-level profile：`get_pairformer_output 6.93-6.97s`，`sample_diffusion 6.16s`，`confidence_head 3.58s`
- 结论：删除 matmul POC 后，正式多卡路径保持健康，没有观察到性能回退。

2026-07-04 tri_mul_in layout 优化：

- 保持多 GPU 维度仍按 output row `i` 切分，只改 row-parallel `tri_mul_in` 内部 layout。
- 原 incoming 公式 `out[i,j,c] = sum_k a[k,i,c] * b[k,j,c]` 在 row-shard 下需要取 `a[:, :, i_rows]` column slice，layout 不如 outgoing 的 `a[:, i_rows, :]`。
- 当前改为先构造 `a_t = a.transpose(1, 2).contiguous()`，再用 outgoing 形态计算：`einsum("bikc,bkjc->bijc", a_t[:, i_rows], b)`。数学不变，但 local shard 访问变成连续 row slice。
- 4GPU `PairformerBlock(c_s=0), N=1218` benchmark：
  - single block `259.10ms`
  - row-parallel block `84.35ms`
  - 约 `3.07x`，比历史真实 block `~97.7ms / 2.64x` 明显改善。
- 稳态 profile 中 `tri_mul_in_compute` 约 `16.6-17.0ms`；历史 subset profile 中该段约 `24.9ms`。
- Correctness：
  - `N=128, c_s=0`：`z max diff 0`
  - `N=128, c_s=384, autocast`：`z/s max diff 0`
- 4GPU 7wux reduced 端到端：
  - 配置：`cycle=1, step=40, sample=5, enable_fusion=False`
  - rank0 model forward `17.56s`
  - top-level profile：`get_pairformer_output 6.22-6.27s`，`sample_diffusion ~6.20s`，`confidence_head ~2.97s`
  - 对比上一轮 `get_pairformer_output 6.91-6.97s`，收益主要落在 Pairformer。

2026-07-04 默认配置回归与 row-parallel 阈值更新：

- 4GPU 7wux 默认配置，`cycle=10, step=200, sample=5, enable_fusion=True`：
  - rank0 model forward `140.87s`
  - top-level profile：`get_pairformer_output 55.7-56.1s`，`sample_diffusion 78.3-79.0s`
  - 对比上一轮 4GPU 默认 7wux `149.05s / Pairformer 68.36s`，收益主要来自 Pairformer。
- 4GPU 三组默认配置，`PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=768` 时：
  - rank0 job `235.60s`
  - 7r6r forward `22.22s`
  - 7wux forward `135.54s`
  - 7pzb forward `56.42s`
  - 7pzb 因 `N_token=600 < 768` 未启用 row-parallel，Pairformer 约 `26.5s`，明显偏慢。
- 单独测试 7pzb，`PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512`：
  - rank0 model forward `47.72s`
  - top-level profile：`get_pairformer_output 14.6-14.7s`，`sample_diffusion 29.8-30.9s`
  - 说明 7pzb 这类中等 N case 应启用 row-parallel。
- 单独测试 7r6r，默认阈值 512：
  - rank0 model forward `22.31s`
  - Pairformer `5.0-5.3s`
  - `N_token=245` 仍低于阈值，没有观察到短序列回退。
- 因此默认 `PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N` 从 `768` 下调到 `512`。

2026-07-04 row-parallel tri-mul 内部分段 profile：

- 新增 gated profile 环境变量：`PAIRFORMER_TRIMUL_PROFILE_LOG=/tmp/trimul.jsonl`。
- 该 profile 只覆盖 row-parallel `tri_mul_out/in` helper，默认不开启；开启后每个方向、每个 rank 写一条 JSONL，包含 projection/gate、transpose、einsum、output linear/gate 等分段。
- Correctness smoke：
  - `N=128, c_s=384, autocast`
  - 强制 `PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=0`
  - `z/s max diff 0`
- 4GPU `PairformerBlock(c_s=0), N=1218` warm profile：
  - no-profile block time 恢复到正常量级：single `257.65ms`，row-parallel `85.86ms`
  - profile 结果只用于阶段占比，不能和 no-profile wall time 直接比较。

稳态分段聚合：

| Direction | Per Record | `einsum_contraction` | `incoming_a_transpose` | 主要非 matmul 开销 |
|---|---:|---:|---:|---|
| outgoing | `16.58ms` | `36.8%` | n/a | layernorm/projection/gate/output 合计约 `63%` |
| incoming | `18.82ms` | `32.8%` | `11.9%` | layernorm/projection/gate/output 合计约 `55%` |

判断：

- contraction/einsum 仍是最大单项，但不是绝对主导；单独替换 matmul kernel 不一定划算。
- incoming 的显式 `a.transpose(1, 2).contiguous()` 是明确可见成本，约 `12%`，后续可看是否在 projection 阶段直接产出 incoming-friendly layout。
- projection/gate/norm/output linear 分散但合计很大，后续如果做 kernel，应该考虑更大范围 fusion，而不是只替换 `einsum_contraction`。

2026-07-05 local-`a` projection 优化：

- 保持 `einsum` / torch-mcBLAS contraction 不变，只减少 contraction 前的 `a` 侧 projection 和 layout round trip。
- 旧 row-parallel helper 会先对 full `z_norm[N,N,C]` 生成 full `a[N,N,C_hidden]`，然后：
  - outgoing 只使用 `a[:, i_rows]`
  - incoming 再做 full `a.transpose(1, 2).contiguous()` 后使用 `a_t[:, i_rows]`
- 新实现只生成本 rank 需要的 local `a`：
  - outgoing: `a_input = z_norm[:, i_rows]`
  - incoming: `a_input = z_norm[:, :, i_rows].transpose(1, 2).contiguous()`
  - `b` 仍保持 full projection，matmul/einsum 不变。
- Correctness：
  - `N=128, c_s=384, autocast`
  - 强制 `PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=0`
  - `z/s max diff 0`
- 4GPU `PairformerBlock(c_s=0), N=1218` no-profile benchmark：
  - single block `257.72ms`
  - row-parallel block `76.07ms`
  - 对比上一版 row-parallel `84-86ms`，继续下降约 `10-12%`。
- warm internal profile：

| Direction | Per Record | `einsum_contraction` | local layout cost | `linear_a_*` cost |
|---|---:|---:|---:|---:|
| outgoing | `14.17ms` | `42.7%` | n/a | `~3.2%` |
| incoming | `14.42ms` | `42.0%` | `incoming_a_input/mask_transpose ~1.8%` | `~3.0%` |

- 4GPU 7wux reduced 端到端：
  - 配置：`cycle=1, step=40, sample=5, enable_fusion=False`
  - rank0 model forward `16.88s`
  - top-level profile：`get_pairformer_output 5.79-5.85s`，`sample_diffusion ~6.06s`
  - 对比上一版 `17.56s / get_pairformer_output 6.22-6.27s`，收益继续落在 Pairformer。

2026-07-05 b 侧和 post 侧融合收益上限评估：

- 基于 `PAIRFORMER_TRIMUL_PROFILE_LOG` 的 4GPU `PairformerBlock(c_s=0), N=1218` warm profile。
- 当前 local-`a` 版本里，contraction/einsum 仍保持 torch/mcBLAS 路径，不作为当前优化对象。

local-`a` 后稳态占比：

| Direction | Per Record | Contraction | b linear | b elementwise | post total | b side total |
|---|---:|---:|---:|---:|---:|---:|
| outgoing | `14.17ms` | `42.7%` | `8.9%` | `17.1%` | `6.8%` | `26.0%` |
| incoming | `14.42ms` | `42.0%` | `8.8%` | `17.4%` | `6.7%` | `26.2%` |

融合上限判断：

- 只融合 b 侧 elementwise 和 post elementwise 的理论上限约为每 record `~3.0ms`，占 `~21%`。真实 Triton/CUDA POC 不可能完全拿满，因为还会有 kernel launch、读写和编译开销。
- post 侧总量只有 `~6-7%`，单独优化优先级不高。
- b 侧总量约 `26%`，是下一步更值得看的对象；但其中 `linear_b_g/linear_b_p` 仍是 linear/GEMM，不应直接替换 mcBLAS。更合理的方向是融合 b projection 后的 sigmoid/mask/mul，或探索减少 b 中间 tensor materialize。
- 因此下一步如果写 Triton/CUDA POC，应优先做 b 侧 pre-matmul fusion，上限评估清楚后再决定是否推广到 post 侧。

2026-07-05 PyTorch 原地 elementwise POC：

- 尝试把 `_pairformer_trimul_project_a/b` 中的 `sigmoid + mask multiply + projection multiply` 改成原地形式：
  - `linear_*_g`
  - `sigmoid_()`
  - `mul_(mask)`
  - `mul_(linear_*_p(...))`
- 不改变 `linear_*` 和 `einsum` / torch-mcBLAS contraction。
- Correctness：
  - `N=128, c_s=384, autocast`
  - 强制 `PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=0`
  - `z/s max diff 0`
- 4GPU `PairformerBlock(c_s=0), N=1218` no-profile benchmark：
  - single block `257.48ms`
  - row-parallel block `75.82ms`
  - 对比 local-`a` 版本 `76.07ms`，只有极小改善。
- internal profile 中 b elementwise 仍约 `17%`，post elementwise 仍约 `3.7%`；PyTorch 原地写法没有真正融合 kernel，因此不能显著降低 elementwise launch/读写成本。
- 结论：该 POC 效果不明显，代码不保留；如果继续优化 b/post elementwise，需要 Triton/CUDA fused elementwise kernel，而不是只改 PyTorch 表达式。
