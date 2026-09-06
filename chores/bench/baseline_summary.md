# 基线测试结果

测试日期：2026-08-20
计算节点：cn23033（鲲鹏 920F，ARMv9 SVE2）
运行方式：`OMP_NUM_THREADS=38`（无 numactl，见下方说明）

## 计算节点环境

```
CPU:        Kunpeng 920F, 608 核, 32 NUMA × 38 核, 2.0 GHz
Features:   sve sve2 sme (ARMv9)
作业 vCPU:  16（每 NUMA 1 核, CPU 37,75,113,...,607）
```

作业请求 cpu=40 或 cpu=128，调度器均只分配 16 vCPU（已验证 -x job 独占模式也如此）。

## numactl -N 1 问题

竞赛文档规定 `OMP_NUM_THREADS=38 numactl -N 1`。`-N` 是 `--cpunodebind`（绑 CPU）。

OMP=1 对照实验证明 `numactl -N 1` 时 38 线程与单线程同速（零并行加速）。
根因：深超算自定义内核"亲和域"机制（`cpu.affinity_*` 文件，ccs_agent 管理）强制执行
CPU 分配——线程 affinity 指向分配集之外时全部合流到单核，`sched_setaffinity` 无法覆盖。
详见 [docs/02-pitfalls.md](../docs/02-pitfalls.md#13)。

run.sh 保留 `numactl -N 1`（竞赛规定）。基线测试去掉 numactl 以获得真实并行性能。
详见 [docs/02-pitfalls.md](../docs/02-pitfalls.md#13)。

## 基线 GFLOPS（全部 PASS）

### CONV

| 用例 | 图像尺寸 | 卷积核 | 时间(ms) | GFLOPS | 误差 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4096×6144 | 39×39 | 5509.68 | 13.68 | 0 |
| 2 | 6144×4096 | 41×41 | 5756.69 | 14.46 | 0 |
| 3 | 4256×6390 | 55×55 | 9765.79 | 16.49 | 0 |
| 4 | 6390×4256 | 81×81 | 21294.60 | 16.24 | 0 |

### ZGEMM

| 用例 | M×N×K | 时间(ms) | GFLOPS | 误差 |
| --- | --- | --- | --- | --- |
| 1 | 7427×7427×256 | 34890.15 | 3.24 | 0 |
| 2 | 14848×14848×256 | 78204.17 | 5.77 | 0 |
| 3 | 37360×8192×512 | — | — | 基线太慢未完成 |

### TRSM

| 用例 | M×N | 时间(ms) | GFLOPS | 误差 |
| --- | --- | --- | --- | --- |
| 1 | 512×19968 | 2224.12 | 2.35 | 1.67e-16 |
| 2 | 2432×17024 | 32307.80 | 3.12 | 2.22e-16 |
| 3 | 17024×512 | 77409.63 | 1.92 | 1.11e-16 |

## 注意事项

- 赛题官方注明：**评测资源为单一 NUMA、CPU 核心数不超过 38**（计算节点一个 NUMA 正好 38 核）。
  评测机上 `numactl -N 1` 应为正常操作（绑定到分到的 NUMA）；零并行为深超算开发容器特有现象。
- 赛题参考编译命令基于华为 HPKit 环境（`gcc/kml25.2.0/kblas/multi`），深超算对应替代为 kplblas/920FSVE。
  评测可能在华为方算力执行，本环境绝对 GFLOPS 仅参考，优化决策看相对提升。
- ZGEMM case3 计算量 ~1.25 TFLOP，基线代码在 16 vCPU 下超 10 分钟

原始日志：`compute_node_baseline.txt`, `conv_numactl_study.txt`, `baseline_v2.log`（集群）
