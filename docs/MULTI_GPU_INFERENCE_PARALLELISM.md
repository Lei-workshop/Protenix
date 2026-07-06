# Protenix 多卡推理并行技术评估

本文记录 Protenix 在 MetaX C500 8 卡机器上的推理并行方案、当前推荐配置、性能数据、正确性边界和后续方向。

更早的 POC、负结果和中间 benchmark 细节见 [MULTI_GPU_INFERENCE_PARALLELISM_HISTORY.md](MULTI_GPU_INFERENCE_PARALLELISM_HISTORY.md)。主文档只保留当前仍会影响实现和使用方式的结论。

## 当前结论

- 当前主线是单个 4 卡 island 上的协同推理：Pairformer row-parallel + Diffusion Ulysses sequence parallel。
- 当前 4GPU 性能优先配置建议显式使用 `--enable_fusion=False`。完整三 case no-profile 结果为 total job `157.56s`。
- triangle attention 继续使用 WMMA v9；triangle multiplicative update 继续使用 torch/mcBLAS。
- 默认 `N_sample=5` sample parallel 已验证无明显收益，不再推进。
- PyTorch-level tri-mul matmul 重排已验证为负结果，实验代码已移除。
- Pairformer row micro-batch overlap 暂不做：真实 profile 中 gather 只占约 `15.8% - 17.4%`，切更细很可能增加 collective latency。
- 更大范围 row-sharded state、减少 full-`z` 同步次数是潜在长期方向，但需要重构 Pairformer 数据流，当前只记录，不实施。

## 推荐运行配置

当前推荐配置：

```bash
WMMA_VERSION=v9
PROTENIX_PAIRFORMER_ROW_PARALLEL=1
PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512
PROTENIX_DIFFUSION_ULYSSES_SP=1
```

Protenix 参数：

```bash
--triatt_kernel=wmma
--trimul_kernel=torch
--enable_fusion=False
```

协同推理时，以下行为已经默认自动开启：

- MP group leader 执行 input preprocessing、旧 MSA 转换和 dataloader featurization。
- batch object 广播给同一 MP group 内的其他 rank。
- model forward 前做 barrier，用于对齐计时和避免 rank 间启动偏移。

因此推荐命令不需要显式设置 `PROTENIX_DISTRIBUTED_DATA_BROADCAST` 或 `PROTENIX_DISTRIBUTED_FORWARD_BARRIER`。

只有调试时才需要显式关闭：

```bash
PROTENIX_DISTRIBUTED_DATA_BROADCAST=0
PROTENIX_DISTRIBUTED_FORWARD_BARRIER=0
```

## 当前 fork 行为

当前 fork 支持两种启动模式。

第一种是多个独立 `torchrun`。每个任务独立拥有自己的 global rank `0..world_size-1`，建议用 `CUDA_VISIBLE_DEVICES` 隔离物理卡，并使用不同 `--rdzv_endpoint` / port。

第二种是单个 `torchrun` 内做 DP x MP。例如 `world_size=4, PROTENIX_INFERENCE_MP_SIZE=2` 时，rank `[0,1]` 和 `[2,3]` 分别组成两个 2 卡 MP group，对应两个 DP shard。

实现语义：

- `PROTENIX_INFERENCE_MP_SIZE` 只在启用协同推理路径时生效。
- 不设置 `PROTENIX_INFERENCE_MP_SIZE` 时，默认 `mp_size=world_size`，即一个任务内所有 rank 共同处理同一个 sample。
- input preprocessing、dataloader item broadcast、Pairformer collectives、Diffusion Ulysses SP collectives 都在 MP group 内发生。
- dump owner 是 MP group leader，不是简单的 global rank0。这样 `2DP x 2MP` 时 rank0 负责第一个 DP shard 输出，rank2 负责第二个 DP shard 输出。
- DP x MP dataloader 使用按 DP rank 的非 padding 分片，避免 `DistributedSampler` 在样本数不能整除时重复 padding sample。

## 硬件假设

- 目标机器：8x MetaX C500。
- 当前默认按 4 卡 island 设计。机器内部每 4 张卡有较强互联，适合优先做 intra-island 协同推理。
- 8 卡暂不作为默认目标。8 卡 benchmark 仍有收益，但跨 island gather 从 4 卡约 `2.5ms` 增到约 `6.5ms`，增量收益明显变小。

collective microbench 的核心结论：

| GPUs | Operation | Iterations | 平均 max-rank 时间 | Total wall time |
|---:|---|---:|---:|---:|
| 4 | all_gather 380 MB | 240 | `2.510 ms` | `0.66s` |
| 8 | all_gather 380 MB | 240 | `6.436 ms` | `1.65s` |

这说明 4 卡 island 内 raw collective bandwidth 不是主要瓶颈。Pairformer row-parallel 的剩余开销更多来自 module-boundary synchronization、layout conversion、residual update、duplicated projection work 和 torch/Python dispatch。

## 最新性能数据

正式性能结论使用 no-profile run。`PROTENIX_PROFILE_LOG`、`TRIATT_PROFILE_LOG`、`PAIRFORMER_PROFILE_LOG`、`PAIRFORMER_TRIMUL_PROFILE_LOG`、`DIFFUSION_PROFILE_LOG`、`DIFFUSION_TRANSFORMER_PROFILE_LOG` 会插入同步或额外统计逻辑，只用于归因，不作为 production wall time。

### 4GPU `enable_fusion` A/B

配置：

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

### Single-case 2GPU / 4GPU

配置：

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

| Case | GPUs | rank0 model forward | rank0 job |
|---|---:|---:|---:|
| 7r6r | 2 | `17.33s` | `24.01s` |
| 7r6r | 4 | `16.38s` | `22.70s` |
| 7wux | 2 | `125.09s` | `141.83s` |
| 7wux | 4 | `86.66s` | `103.04s` |
| 7pzb | 2 | `39.50s` | `49.84s` |
| 7pzb | 4 | `30.66s` | `40.51s` |

判断：

- 7wux 是当前最值得用 4GPU 的 case，4GPU forward 稳定在 `85-87s` 区间。
- 7pzb 从 2GPU 到 4GPU 仍有收益，forward `39.50s -> 30.66s`。
- 7r6r 是短序列，2GPU/4GPU 差距很小；4GPU 性能略好但卡时不划算。
- 2GPU 开启 Diffusion SP 后，7wux single-case forward 从旧三 case 数据中的 `162.50s` 降到 `125.09s`，说明 Diffusion SP 对 2GPU 也有效。

### 旧 4GPU 数据的定位

早期 4GPU full no-profile 数据为 total job `218.98s`，7wux forward `130.96s`。这组数据对应旧代码：当时 Diffusion Ulysses SP 没有覆盖默认 `enable_fusion=True` 的 fused pair-bias 路径，因此只能作为历史对照，不是当前最佳结果。

## 并行策略

### Pairformer row-parallel

Pairformer 中最重的是 z-heavy 子图，核心 tensor 近似为 `[B, N, N, C]`。当前方案沿 output row `i` 切分：

```text
rank r owns i rows: z[:, i_start:i_end, :, :]
```

主要收益来自：

- triangle attention start/end 的 row 维切分；
- triangle multiplicative update out/in 的 row 维切分；
- 对中大 N case 避免每张卡重复完整 z-heavy compute。

当前仍会在阶段边界 all-gather full `z`，因为 `tri_mul_in`、`tri_att_end`、pair transition、下一个 block 都依赖更新后的完整 pair representation。raw all-gather 本身不大，但 full-`z` 同步会引入同步点、layout 转换和调度开销。

`PROTENIX_PAIRFORMER_ROW_PARALLEL_MIN_N=512` 用于避免短序列误走 row-parallel。7r6r 这类小 N case 不是该优化的主要收益对象。

### Diffusion Ulysses SP

当前 DiffusionTransformer 走 Ulysses sequence parallel：

- token/sequence 维切分 query rows；
- QKV 做 packed all-to-all，在 seq/head layout 间转换；
- 每个 rank 只计算本 rank 负责的 local heads pair bias；
- attention 后 all-gather 回完整 token activations。

这条路线不是 TP。它利用 Protenix diffusion attention 的 `n_heads=16` 可被 2/4 卡整除的条件，在 attention 内做 sequence parallel。

`enable_fusion=True` 和 `enable_fusion=False` 都已支持 SP。当前完整三 case no-profile 下 `enable_fusion=False` 略快，因此作为性能优先推荐。

### DP x MP

当前实现支持单任务内的 DP x MP：

```text
world_size=4
PROTENIX_INFERENCE_MP_SIZE=2
```

此时 rank `[0,1]` 是一个 2 卡 MP group，rank `[2,3]` 是另一个 2 卡 MP group。每个 group 内执行 Pairformer row-parallel 和 Diffusion SP；不同 group 处理不同 DP shard。

注意事项：

- dump 判断必须用 MP group leader，不能只看 global rank0。
- input preprocessing 和 dataloader broadcast 也必须限制在 MP group 内。
- 如果用两个独立 `torchrun` 分别跑两组 4 卡任务，则两个 job 的 global rank 都各自从 0 开始；这是最简单、最推荐的多任务使用方式。

## Correctness 状态

默认完整性能测试使用 `configs.deterministic=False`，不能用 dumped CIF/summary 做跨运行 bitwise 判断。

已经验证的正确性边界：

- Diffusion fused SP module-level correctness 通过：

```text
DiffusionTransformer(c_a=64,c_s=32,c_z=16,n_blocks=2,n_heads=8)
N=37, B=2, enable_fusion=True
fused baseline vs fused SP: max_diff=6.80e-06, mean_diff=2.98e-08
```

- 打开 `configs.deterministic=True` 后，2GPU pure MP 最小 case 连续两次输出 byte-wise 一致。
- `2DP x 2MP` 与等价 pure MP baseline 对齐时，需要保持相同 sample 顺序/RNG 顺序；在该条件下，`7r6r`、`7pzb`、`7wux` 的 CIF 和 summary JSON SHA256 均一致。

不承诺的部分：

- 默认非 deterministic 完整性能跑不承诺 dumped structure 跨并行度 bitwise 一致。
- 1GPU 与 2/4GPU 因 Ulysses SP、row-parallel 改变计算拆分和规约顺序，不应期待 raw CIF byte-wise 完全相同。

## 已暂停或废弃方向

### `N_sample=5` sample parallel

真实 Protenix 默认 `sample_diffusion_chunk_size=5` 下，`N_sample=5` 已经 batch 化执行。实验性 sample split 在 `step=40` 下没有带来收益，代码已撤回。

结论：默认 `N_sample=5` sample parallel 不作为当前方向。除非后续显存压力迫使 sample 串行，或目标场景的 `N_sample` 明显大于 5，否则不建议继续。

### Ring-style Diffusion token parallel

Ring POC 的数学路径跑通，分块 softmax + pair bias block correctness 正常。但真实默认 `N=1218, N_sample=5` 下稳定收益只有约 `1.3x`，低于 Ulysses SP，且实现复杂度更高。

结论：保留为技术参考，不作为主线。

### PyTorch-level tri-mul matmul 重排

尝试过把 `tri_mul_in` 的 `a[k,i]` 访问改成更连续的 local `a_t[i,k]` 形态，也尝试过 PyTorch-level matmul 重排。结果数学正确，但端到端收益不稳定或不明显。

结论：正式代码不保留该 POC。后续若继续优化 tri-mul compute，应转向专用 fused kernel / mctlass-style MMA，而不是 PyTorch 表达式层面的重排。

### Pairformer row micro-batch overlap

profile 显示 gather 只占约 `15.8% - 17.4%`，稳态 gather 多数为 `2-4ms`。把 row shard 继续切成 micro-batch 会增加 all-gather 次数，可能放大 collective latency。

结论：当前不建议实现复杂 async micro-batch overlap。

### 更大范围 row-sharded state

理论上，如果更多 Pairformer 状态长期保持 row-sharded，可以减少 full-`z` all-gather 的次数或数据量。但当前 Pairformer 数据流里多个阶段都依赖 full `z`，改动会跨越 triangle attention、triangle multiplicative update、pair transition 和 block 间 residual。

结论：这是长期设计方向，当前不实施。

## 后续工作

短期：

- 保持当前 4GPU 推荐配置和文档命令稳定。
- 继续用 no-profile run 作为性能结论来源，profile 只做阶段归因。
- 如果后续发现 7pzb/7wux 性能漂移，优先确认 GPU reset、profile env、`enable_fusion` 和 distributed env 状态。

中期：

- 如果继续优化 Pairformer，优先看 tri-mul 前后 elementwise / projection / gate / norm / linear 的 fused kernel，保留 torch/mcBLAS contraction。
- 如果继续优化 Diffusion，重点看 `enable_fusion=False` 路径为什么在 4GPU SP 下略优，以及 fused pair-bias 路径是否还有固定开销。
- 如果要评估 8GPU，先单独测跨 island collective 和真实 end-to-end，不默认推广。
