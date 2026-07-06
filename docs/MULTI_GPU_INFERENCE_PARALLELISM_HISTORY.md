# Protenix 多卡推理历史实验摘要

本文归档 `MULTI_GPU_INFERENCE_PARALLELISM.md` 中被压缩掉的历史 POC 和中间结论。主文档只保留当前可执行结论；本文用于说明为什么一些方向被暂停或废弃。

## 背景 profile

`WMMA_VERSION=v9`、`trimul=torch` 下，7wux 的早期 profile 显示：

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

## Diffusion `N_sample` 并行

最初判断 sample parallel 是最干净的多卡目标，因为 trunk 输出后 5 条 diffusion 轨迹相互独立。

synthetic POC 使用真实 shape、缩短循环：

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

结果：

| 指标 | 时间 |
|---|---:|
| 单卡 synthetic compute | 573.935 ms |
| 4 卡 distributed wall time | 240.520 ms |
| 加速比 | 2.39x |
| sample 切分 | `[2, 1, 1, 1]` |

这个结果说明通信不是 sample parallel 的瓶颈。但真实 Protenix 默认 `sample_diffusion_chunk_size=5` 已经把 5 个 samples batch 化执行，真实接入后收益消失。

真实 7wux、4 卡、同时启用 Pairformer row-parallel：

| 配置 | no split rank0 forward | sample split rank0 forward | 结论 |
|---|---:|---:|---|
| `cycle=1, step=5, sample=5` | 29.10s | 20.82s | 单次 wall time 有波动，看 profile 不成立 |
| `cycle=1, step=40, sample=5` | 33.32s | 33.60s | 无收益 |

结论：实验代码已撤回，默认 `N_sample=5` sample parallel 不再推进。

## Pairformer row-parallel POC

Pairformer row-parallel 先后做过：

- 最小 kernel POC；
- 模块级 triangle attention POC；
- 模块级 triangle multiplicative update POC；
- PairformerBlock subset POC；
- PairformerStack synthetic full-stack POC；
- `runner/batch_inference.py` 4 卡 smoke；
- 真实 7wux reduced / full run。

核心发现：

- triangle attention row-parallel 可接近 `3.36x`，通信不是主要瓶颈。
- triangle multiplicative update out 比 in 更容易加速；in 的 layout/访问更差。
- full `z` all-gather 本身很快，但阶段边界同步、layout conversion、residual update 和 torch dispatch 会稀释收益。
- 7r6r 这类短序列不适合作为 row-parallel 收益目标，因此加入了 `N_token >= 512` 阈值。

通信 microbench 说明 4 卡 island 具备足够 bandwidth：

| GPUs | Operation | Iterations | 平均 max-rank 时间 | Total wall time |
|---:|---|---:|---:|---:|
| 4 | all_gather 380 MB | 240 | 2.510 ms | 0.66s |
| 8 | all_gather 380 MB | 240 | 6.436 ms | 1.65s |

结论：Pairformer row-parallel 成为主线，但默认优先 4 卡 island，不默认扩展到 8 卡。

## Diffusion token parallel

在默认 `N_sample=5` sample parallel 负结果后，diffusion 方向转向 transformer 内部 token parallel。

### Ring POC

Ring-style POC 做 query-row shard，K/V shard 环传，并实现带 pair bias block 的在线 softmax 合并。

稳定结果里，真实默认 `N=1218, N_sample=5` 收益约 `1.29x`，不接近 4 卡理想扩展。

结论：数学正确，但复杂度高、收益低，不作为主线。

### Ulysses SP POC

Ulysses sequence parallel 利用 `n_heads=16` 可被 2/4 卡整除的条件，在 sequence/head layout 间做 all-to-all。

早期 module-level 结果：

| Shape | Blocks | Baseline | Ulysses SP warm | Speedup warm | Correctness |
|---|---:|---:|---:|---:|---|
| `N=1218, N_sample=5` | 2 | 62.317 ms | 6.381 ms | 9.77x | max diff `4.3e-5` |
| `N=1218, N_sample=5` | 24 | 748.634 ms | 75.336 ms | 9.94x | max diff `2.13e-4` |

这个高收益依赖 rank-local pair-bias cache，避免 diffusion 每步重复重算 `z -> bias`。

后续接入真实 `sample_diffusion` 后发现：

- `enable_fusion=False` reduced 路径收益较大；
- `enable_fusion=True` 默认 fused path 最初没有进入 SP，导致完整默认路径 diffusion 时间基本不降；
- 后续补齐 fused local-head pair-bias 后，`enable_fusion=True` 也支持 SP。

4GPU 7wux reduced、`enable_fusion=True`：

| Diffusion SP | rank0 model forward | rank0 job | `get_pairformer_output` avg | `sample_diffusion` avg |
|---|---:|---:|---:|---:|
| off | 30.50s | 48.47s | 5.86s | 19.27s |
| on | 21.53s | 37.62s | 5.80s | 10.34s |

结论：Ulysses SP 是当前 diffusion 主线。完整三 case no-profile 下 `enable_fusion=False` 略快，因此推荐性能跑显式关闭 fusion。

## DP x MP 和 deterministic 检查

为支持单个 `torchrun` 内 `2DP x 2MP`，实现从 global rank 语义调整为 MP group leader 语义：

- dump owner 使用 MP group leader；
- preprocessing 和 dataloader broadcast 限定在 MP group；
- dataloader 按 DP rank 做非 padding 分片。

deterministic 检查显示：

- 2GPU pure MP 最小 case 连续两次输出 byte-wise 一致；
- `2DP x 2MP` 与等价 pure MP baseline 在相同 sample 顺序/RNG 顺序下，`7r6r`、`7pzb`、`7wux` 的 CIF 和 summary JSON SHA256 一致。

默认性能跑仍使用 `configs.deterministic=False`，不承诺跨并行度 dumped structure bitwise 一致。

## 负结果和撤回项

### PyTorch-level tri-mul matmul 重排

尝试过把 incoming 的 `a[k,i]` 变成 local `a_t[i,k]`，以及更大范围的 PyTorch-level matmul 重排。虽然局部 profile 有时改善，但端到端收益不稳定，正式代码不保留该 POC。

如果未来继续优化 tri-mul compute，应考虑专用 fused kernel / mctlass-style MMA。

### b/post PyTorch 表达式融合

尝试过 b 侧和 post 侧的 PyTorch 表达式重排。效果不明显，代码不保留。

如果继续优化 b/post elementwise，需要 Triton/CUDA fused elementwise kernel，而不是只改 PyTorch 表达式。

### Pairformer row micro-batch overlap

真实 profile 中 gather 占比不够高，切分 micro-batch 可能增加 collective latency。当前不实现复杂 async overlap。

## 旧完整结果定位

早期 4GPU 完整三 case no-profile：

| Case | N_token | rank0 model forward |
|---|---:|---:|
| 7r6r | 245 | 22.96s |
| 7wux | 1218 | 130.96s |
| 7pzb | 600 | 41.98s |

整体 rank0 job time 为 `218.98s`。

这组数据对应旧代码：当时 Diffusion Ulysses SP 没有覆盖默认 `enable_fusion=True` 的 fused pair-bias 路径。因此它是历史对照，不是当前最佳性能。
