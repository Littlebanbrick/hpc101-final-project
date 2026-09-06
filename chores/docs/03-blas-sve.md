# BLAS 库选型与 SVE 环境分析

TRSM 编译链接问题的背景分析。做 TRSM/ZGEMM 优化前应了解的环境约束。

## 登录节点 vs 计算节点

| | 登录节点 | 计算节点 |
| --- | --- | --- |
| CPU | Kunpeng 920F (`0xd01`)，128 核 4 NUMA | 608 核 32 NUMA |
| 向量化 | NEON (asimd/asimddp)，**无 SVE** | **armv9 + SVE2 + SME** |
| kplblas | `Illegal instruction`（SVE 指令） | 可运行 ✅ |
| 来源证据 | `/proc/cpuinfo` Features 无 `sve` | 库名 `libkplblas_armv9p_v2409.so` + cpuinfo 有 `sve` |

kplblas 文件名含 `armv9p_v2409`，表明为 ARMv9 计算节点编译，依赖 SVE 指令集。登录节点是 ARMv8.2，无 SVE，执行即崩。

## ⚠️ SVE 向量长度是 512-bit（2026-09-02 实测）

计算节点 `svcntd() = 8`——**每个 z 寄存器 8 个 double**，不是常见的 256-bit (4 个 double)。
影响所有手写 SVE 代码：

- 硬编码向量长假设的内核会错位崩掉（TRSM v4 实测 err 0.5）
- 寄存器分块微内核的收益比 256-bit 机器翻倍
- 写法：运行时 `svcntd()` 自适应，或 `whilelt` 谓词控制（auto-vec 自动正确）
- 单核 dgemm 实测 43.5 GFLOPS ≈ 2×512-bit FMA 管线 @ ~2.6GHz 的理论值

## 集群 BLAS 库对比

| 库 | 路径 | 编译器 | SVE | `cblas_domatcopy` | 登录节点 | 竞赛用 |
| --- | --- | --- | --- | --- | --- | --- |
| **kplblas/920FSVE** | `/work_ssd/software/soft/tool/kplblas/920FSVE/lib` | — | ✅ | ✅ | ❌ | **是** |
| shenchao kblas | `/home/share/shenchao_common/kblas` | BiSheng/LLVM | ✅ | ✅ | ❌ | 否（同源旧版） |
| armpl/24.10 | `/work_ssd/software/soft/lib/armpl/24.10-gcc10.3.1/...` | gcc 10 | NEON+SVE 双版 | ❌ | ✅ | 否 |
| openblas | `/work_ssd/software/soft/lib/openblas/...` | gcc/bisheng | ? | ? | ✅ | 否 |

### 为什么 TRSM 必须用 kplblas

`bench_trsm.c` 硬编码 `#include "kblas.h"`，且调用 `cblas_dgemm` + `cblas_domatcopy`。同时满足「头文件 + 两个 CBLAS 符号」的只有 kplblas：

- armpl 有 `cblas_dgemm` 但**无 `cblas_domatcopy`**（只有 Fortran 接口 `domatcopy_`）。
- kplblas 符号齐全：`cblas_dgemm`、`cblas_domatcopy`、`cblas_zgemm`、`cblas_zgemm3m`。
- **完整 BLAS API 都在**：`cblas_dtrsm`、`cblas_dtrsm_pack`、`cblas_ztrsm` 等全部导出（nm 确认）。此外有非标准线程控制：`BlasSetNumThreads` / `BlasGetNumThreads(Local)`——自研 OpenMP 编排单线程 BLAS 调用的关键开关（已验证有效，且必须先调用，否则 libomp 嵌套警告 + 性能灾难）。

### kblas.h 来源

kplblas 模块不带头文件。`kblas.h` 从 `/home/share/shenchao_common/kblas/kblas.h` 取（60KB，完整声明）。

## 对 ZGEMM 优化的启示

`kblas.h` 中除标准 `cblas_zgemm` 外，还声明了：

- `cblas_zgemm3m` — 3M 法（复数 GEMM 拆为 3 次实 GEMM + 组合，降运算量）
- `cblas_zgemm_pack` / `cblas_zgemm_compute` — 打包式 GEMM（先 pack A/B 再计算）

这些接口的**设计思路**对 ZGEMM 优化有参考价值（3M 法、分块打包），但竞赛只能改 `zgemm.c` 自行实现，不能调库。

## 编译器选择

| 编译器 | 版本 | 路径 | 备注 |
| --- | --- | --- | --- |
| gcc | 10.3.1 | `/usr/bin/gcc` | 默认，`-fopenmp` 链 libgomp |
| BiSheng (clang) | 5.1.0 (19.1.7) | `/home/share/eapp_common/chenmiao/BiShengCompiler-5.1.0-aarch64-linux/bin/clang` | 基于 LLVM，`-fopenmp` 链 libomp |

kplblas 依赖 `libomp.so`（LLVM 版），但用 gcc 编译 trsm.c 本身没问题——gcc 的 `-fopenmp` 给 trsm.c 自己的 OpenMP 用 libgomp，kplblas 内部用 libomp，两者在运行时共存可行（已验证编译和运行均成功）。
