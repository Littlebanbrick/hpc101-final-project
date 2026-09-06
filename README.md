# HPC101 大作业 — 鲲鹏高性能计算全球挑战赛（S2 赛季）算子优化

本仓库为 HPC101 课程大作业：参加**鲲鹏高性能计算全球挑战赛 S2 赛季**，
在**鲲鹏 920F**（ARMv9 SVE2）平台上完成三道算子优化赛题，
在保证数值正确性的前提下最大化性能（GFLOPS）。

- 最终成绩：**CONV 58–68× · TRSM 48–115×（三用例总计 94.5×）· ZGEMM 2.5–24×**
- 完整技术报告见根目录 [`report.pdf`](report.pdf)（16 页），答辩幻灯片见 [`slides.pdf`](slides.pdf)（27 页）
- 大作业文档：<https://hpc101.zjusct.io/lab/Final-Project/>

## 成绩总览

| 题目 | 说明 | 基线 GFLOPS | 最终 GFLOPS | 加速比 |
| --- | --- | --- | --- | --- |
| CONV | 二维卷积（float） | 12.95 | 748–886 | 58–68× |
| TRSM | 三角方程组求解 `LX=B`（double） | 1.7–2.7 | 102–213 | 48–115× |
| ZGEMM | 复数矩阵乘法 `C=αAB+βC` | 5.7–49 | 160–207 | 2.5–24× |

> 数值与[实验报告](report.pdf)总览一致：最终 GFLOPS 为登录节点复测；
> ZGEMM 加速比按开发节点单 NUMA 同环境对比计算（详见报告第三章说明）。
> TRSM 三用例总耗时 126.78 s → 1.34 s（**94.5×**），且执行环境自适应机制
> 保证在集群异常调度下性能有界（见下文与报告第二章）。

## 优化方法摘要

### CONV — 748–886 GFLOPS（58–68×）

四行融合处理、垂直权重打包与 `ik` 四步滑动窗口三项访存/向量化优化，
OpenMP 38 线程并行。提供编译期 A/B 回退宏（`-DCONV_DISABLE_ROW4_SLIDE4`
等）便于对照实验。

### TRSM — 102–213 GFLOPS（48–115×）

分块右视前代法（块大小 512）：计算主体更新段转化为**形状自适应**的
`cblas_dgemm`（按两操作数大小选择行/列切分，广播小操作数），
对角段按列条带并行调用单线程 `cblas_dtrsm`。

针对开发集群的**合流坍缩**现象（超过 8 线程在特定掩码下被强制锁到同一核，
并留下粘性降级标记，详见报告第二章），设计**双信号一次性探针决策**：
环境健康则 8 线程并行，检出异常则全程串行兜底（23–35 GFLOPS，
仍为基线 11–20×）——任何环境下性能有界、无灾难。

### ZGEMM — 160–207 GFLOPS（2.5–24×）

复矩阵乘法拆解为实数分块运算：实/虚部打包（`a_re_pack`/`a_im_pack`）
预处理 + 分块参数调优，将复数 GEMM 转化为对 SVE 向量化友好的实数运算组合。

## 环境与复现

- 平台：鲲鹏 920F 集群（登录节点 ARMv8.2 NEON，计算节点 ARMv9 SVE2）
- 各题目录下 `bash run.sh` 一键编译运行（CONV/ZGEMM 无外部依赖；
  TRSM 需 kblas（HPKit）或 kplblas（NSCCSZ），脚本自动探测）
- 数值容差：CONV 1e-5 · TRSM 1e-12 · ZGEMM 1e-10（全部用例 PASS）
- 每题仅可修改对应源文件（`conv2d.c` / `trsm.c` / `zgemm.c`），
  评测框架 `bench_*.c` 不可修改
- x86/Windows 本机可编译验证正确性，真实性能基准须在鲲鹏 920F 集群上测得

```bash
cd CONV  && bash run.sh
cd TRSM  && bash run.sh
cd ZGEMM && bash run.sh
```

报告/PPT 重编译（typst）：

```bash
typst compile chores/final_report/report.typ report.pdf
typst compile chores/final_report/slides.typ slides.pdf
```

## 仓库结构

```
hpc101-final-project/
├── README.md            # 本文件
├── report.pdf           # 实验报告终稿（16 页）
├── slides.pdf           # 答辩 PPT 终稿（27 页，16:9）
├── LICENSE              # MIT 许可证
├── CONV/                # 赛题一：conv2d.c、run.sh、README、运行截图
├── TRSM/                # 赛题二：trsm.c、run.sh、README、运行截图
├── ZGEMM/               # 赛题三：zgemm.c、run.sh、README、运行截图
└── chores/              # 过程文档与实验记录
```

## 过程文档导览（chores/）

- `docs/02-pitfalls.md`：23 个实战踩坑记录，含 TRSM 合流坍缩的完整定位过程
- `bench/trsm_kblas_study.md` / `bench/trsm_final_data.md`：
  TRSM 机理研究与最终性能数据
- `final_report/`：报告与幻灯片的 typst 源码及图片素材
- `CLUSTER.md`、`docs/01-onboarding.md`：集群使用指南与协作者上手文档

## 许可

[MIT](LICENSE)
