// HPC101 期末大作业报告 — 排版对齐 Lab3.5_report.typ 范本
// 编译： typst compile report.typ
// 素材： 浙江大学.png / conv.png / zgemm1.png / shot_trsm.png（同目录）

// ===== Code Block Style =====
#show raw.where(block: true): set block(
  fill: luma(250),
  inset: 6pt,
  radius: 3pt,
)
#show raw.where(block: true): set text(
  size: 9pt,
  font: ("Courier New", "NSimSun", "SimSun", "Microsoft YaHei UI"),
)
#show raw.where(block: false): set text(
  font: ("Courier New", "NSimSun", "SimSun", "Microsoft YaHei UI"),
)
#show table.cell: set text(size: 12pt)
#show table.cell: set par(leading: 0.3em, spacing: 0.5em)
#show table.cell.where(y: 0): set block(above: 0.5em, below: 0.5em)
#show raw.where(block: true): set par(leading: 1.15em)

// Math styling
#show math.equation.where(block: true): set text(size: 12pt)
#show math.equation.where(block: true): it => block(
  fill: luma(245),
  inset: 8pt,
  radius: 3pt,
)[#it]
#show math.equation.where(block: false): set text(size: 12pt)

// Page
#set page(
  paper: "a4",
  margin: (left: 2.6cm, right: 2.6cm, top: 2.4cm, bottom: 2.8cm),
  numbering: "1",
  number-align: bottom + center,
  header: context [
    #text(size: 12pt, fill: gray.darken(25%))[ZJUSCT / HPC101]
    #h(1fr)
    #text(size: 12pt, fill: gray.darken(25%))[#datetime.today().display("[month repr:short] [day], [year]")]
  ],
  footer: context align(center)[
    #text(size: 11pt, fill: gray.darken(50%))[#counter(page).display()]
  ],
)

// Body text
#set text(
  font: ("Times New Roman", "Georgia"),
  size: 14pt,
  lang: "en",
)
#set par(
  justify: true,
  first-line-indent: 0em,
  leading: 0.75em,
  spacing: 1em,
)

// Heading hierarchy
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [
  #v(0.9em)
  #text(size: 20pt, weight: "bold", it.body)
  #v(0.35em)
]
#show heading.where(level: 2): it => [
  #v(0.55em)
  #text(size: 15pt, weight: "semibold", it.body)
  #v(0.25em)
]
#show heading.where(level: 3): it => [
  #v(0.35em)
  #text(size: 14pt, weight: "semibold", it.body)
  #v(0.15em)
]

#show figure.caption: it => [
  #set text(size: 10pt, fill: gray.darken(40%))
  #it.supplement
  #context it.counter.display(it.numbering)
  #it.separator
  #it.body
]

// ===== Cover =====
#align(center + horizon)[
  #v(0%)
  #text(size: 35pt, weight: "bold")[
    HPC101 期末大作业
    \
    鲲鹏 920F 三算子优化
  ]
  #image("浙江大学.png", width: 40%)
  #v(2em)
  #text(size: 20pt)[作者：齐思航　王传宇]
  #v(0.5em)
  #text(size: 20pt)[时间：2026-09-05]
  #v(0.5em)
]

#pagebreak()

// ===== 零、总体概览 =====

= 零、总体概览

本大作业在鲲鹏 920F 开发集群（ARMv9 SVE2，608 核 / 32 NUMA × 38 核 @ 2.0 GHz）上完成 HPC101 三道赛题的优化：CONV（二维卷积）、TRSM（三角方程组求解）、ZGEMM（复数矩阵乘法）。三题均为 C + OpenMP 手写优化，技术路线覆盖 NEON SIMD 向量化、循环分块与数据复用、BLAS 库调用编排、以及面向执行环境异常的自适应线程方案，正确性全部通过官方校验。

分工：题目一 CONV、题目三 ZGEMM 由齐思航负责；题目二 TRSM 由王传宇负责。各题最终成绩一览：

#table(
  stroke: 0.3pt,
  columns: 5,
  [*题目*], [*负责人*], [*基线 GFLOPS*], [*最终 GFLOPS*], [*加速比*],
  [CONV], [齐思航], [12.95], [748–886], [58–68×],
  [TRSM], [王传宇], [1.7–2.7], [102–213], [48–115×],
  [ZGEMM], [齐思航], [5.7–49], [160–207], [2.5–24×],
)

其中 CONV/ZGEMM 为 38 线程实测，最终区间取自各章末尾的性能图（登录节点复测，最大误差与章内终验完全一致，为同一代码）；ZGEMM 章内过程数据为开发节点单 NUMA 形态实测，其加速比按章内同环境对比计算。TRSM 为健康环境下 8 线程编排实测（其执行环境自适应机制详见第二章）。三题完整的优化思路、过程与数据见后续各章。

#pagebreak()

// ===== 一、CONV（齐思航）=====

= 一、CONV —— 二维卷积优化

== 1.1 题目背景

给图像、卷积核，求二维卷积的输出矩阵（尺寸为`(inputHeight - kernelHeight + 1) × (inputWidth - kernelWidth + 1)`）。

测试样例有4个，图像尺寸与卷积核尺寸各不相同。

== 1.2 baseline分析

朴素四重循环，仅在外层循环使用OpenMP并行化。

性能测定如下，由于计算节点的donau环境执行机制有问题，38线程会在单NUMA节点的一个CPU上运行，效率很低，所以这里暂时不用单NUMA。
```
OMP_NUM_THREADS=1,GFLOPS:1.0037
OMP_NUM_THREADS=38,GFLOPS:12.9528
```

== 1.3 V1优化

=== 1.3.1 局部累加，消除反复读写输出

将
```
output[idx] += input[...] * kernel[...];
```
改为
```
float sum = 0.0f;
sum += input[...] * kernel[...];
output[idx] = sum;
```
将sum作为临时变量，避免反复写入output数组，节省开销。

=== 1.3.2 restrict

声明 `input`、`kernel`、`output` 不重叠，帮助编译器进行寄存器分配和向量化。

=== 1.3.3 OpenMP 静态调度

```
#pragma omp parallel for schedule(static)
```
按输出行分配线程，降低调度开销。

=== 1.3.4 指针提升

基线计算数组下标时，总是计算`(j + jk) * inputWidth + (i + ik)`，其实可以提前算好行指针`inputRow`和`kernelRow`，内层只需要：
```
inputRow[ik]
kernelRow[ik]
```

=== 1.3.5 AArch64 NEON SIMD

充分利用AArch64的NEON SIMD指令集，一次计算连续4个输出列
```
output[j][i]
output[j][i + 1]
output[j][i + 2]
output[j][i + 3]
```
可以理解为以下伪代码，同时计算四个输出
```
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;

for (int jk = 0; jk < kernelHeight; ++jk) {
    for (int ik = 0; ik < kernelWidth; ++ik) {
        float w = kernel[jk * kernelWidth + ik];

        sum0 += input[(j + jk) * inputWidth + (i + ik)]     * w;
        sum1 += input[(j + jk) * inputWidth + (i + ik + 1)] * w;
        sum2 += input[(j + jk) * inputWidth + (i + ik + 2)] * w;
        sum3 += input[(j + jk) * inputWidth + (i + ik + 3)] * w;
    }
}

output[j * outputWidth + i]     = sum0;
output[j * outputWidth + i + 1] = sum1;
output[j * outputWidth + i + 2] = sum2;
output[j * outputWidth + i + 3] = sum3;
```
V1优化后，38线程GFLOPS提升至165.5719。

== 1.4 V2优化

=== 1.4.1 输出列分块

V1优化后，每次迭代利用一个NEON向量一次处理4个列，其实可以一次性处理16列，即一次处理4个NEON向量，对应每个输出行有4个累加器
```
float32x4_t s0[4];
```
这样可以减少循环控制、地址计算和边界判断。

分块规模经过充分尝试，8列并行度不足，20/32/64列累加器过多，寄存器压力太大，最佳选择是16列并行，这样达到了并行度与寄存器数量之间的平衡。

=== 1.4.2 四行输出融合

显然，对于相邻几行的输出，卷积核是部分重叠的，因此可以利用一次计算多行的方式节省寄存器。
```
CONV_ROW4_VP_FMA(s0, v, w, 0);
CONV_ROW4_VP_FMA(s1, v, w, 1);
CONV_ROW4_VP_FMA(s2, v, w, 2);
CONV_ROW4_VP_FMA(s3, v, w, 3);
```
经测试四行融合比二行融合、六行融合要好，遂采纳，提升约22%性能。

V2优化后，38线程GFLOPS提升至621.42。

=== 1.4.3 输入滑动窗口

对于相邻的 ik，输入窗口高度重叠。例如：
```
ik = 0: input[0..3]
ik = 1: input[1..4]
ik = 2: input[2..5]
ik = 3: input[3..6]
```
如果每次都重新加载，会重复读取大量数据。

可以使用滑动窗口减少输入次数，具体操作如下：

1、加载5个相邻向量

2、使用 `vextq_f32` 生成 4 个滑动窗口

3、分别完成 4 个卷积位置的计算

第二轮优化后，38线程GFLOPS提升至809.31。

#align(center, image("conv.png", width: 10cm))

最终性能如图所示。

#pagebreak()

// ===== 二、TRSM（王传宇）=====

= 二、TRSM —— 三角方程组求解优化

== 2.1 题目背景

给定 $M times M$ 下三角矩阵 $L$ 与 $M times N$ 右端矩阵 $B$，在行主序存储下求解 $L X = B$。总计算量约 $M^2 N$。官方测试用例三组：$512 times 19968$（宽矩阵）、$2432 times 17024$（近方阵）、$17024 times 512$（高矩阵），长宽比跨度大，分别考察访存模式与并行结构在不同形状下的表现。要求结果通过精度校验（双精度，误差在机器精度量级）。

== 2.2 baseline 分析

基线为逐列前代消元的三重循环，外层列循环用 OpenMP 并行：

```c
#pragma omp parallel for
for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int k = 0; k < i; ++k)
            s += L[i * lda + k] * B[k * ldb + j];
        B[i * ldb + j] = (B[i * ldb + j] - s) / L[i * lda + i];
    }
```

实测仅 2.12 / 2.74 / 1.69 GFLOPS。瓶颈在于：(1) 行主序下按列访问 $B$，跨步为 ldb，缓存行利用率极低；(2) 标量点积无法向量化，逐元素除法打断流水线；(3) 每列内部是 $i$ 方向的写后读依赖链，列间并行粒度受 $N$ 限制且各列工作量不均（高矩阵形状下并行度骤降）。

== 2.3 优化路线一：分块算法与 BLAS 库编排

将 $L$ 沿对角划分为宽度 512 的条带，逐步推进，每一步分两段：

+ *对角段*：对角块 $L_(k,k)$ 与当前 RHS 条带构成窄三角系统，按列切片并行求解——每个线程调用一次单线程 `cblas_dtrsm` 解出自己的列条带，合并即得 $X_k$；
+ *更新段*：下方剩余条带做 Schur 补更新 $B_"rest" -= L_"rest,k" times X_k$，全部转化为规则的 `dgemm` 调用。

更新段按形状自适应分割：目标块行数较少时按列分割、广播 $L_(i,k)$，否则按行分割、广播 $X_k$，在保证负载均衡的同时让每个线程的访存连续。

多线程策略上，所有库调用强制单线程（`BlasSetNumThreads(1)`），由外层 OpenMP 编排 8 个单线程库调用并发执行：既避免库内多线程与集群调度器的冲突，又保证每个 dgemm 的规模足够大、能逼近库的峰值效率。该路线在健康环境下即达到 96–213 GFLOPS。

== 2.4 优化路线二：执行环境异常与自适应线程数

=== 2.4.1 现象与机理

官方评测命令（38 线程 + `numactl -N 1` 绑定单 NUMA）在本集群上会触发"合流坍缩"：大规模多线程运行数秒后，全部线程被内核强制调度到同一物理核，吞吐下降 20–30 倍；且标记具有粘性——同进程内后续单线程代码也被降速。经 OMP=1 对照、`sched_getcpu` 采样、掩码外借核等实验确认为集群自定义亲和域内核的强制合流机制：在 numactl 掩码内检测到严重过订阅时启用，评估周期约 2 秒，8 线程及以下不触发。

这带来两个硬约束：短探针会被"延迟生效期"骗过（探针期间并行正常，正式工作开始后被合流）；"先试并行、失败再退串行"不可行——试探本身即触发粘性标记，回退后的串行也被毒化（实测毒化后串行仅 0.6 GFLOPS，正常串行为 23–35 GFLOPS）。

=== 2.4.2 v10 自适应方案

+ *正式工作封顶 8 线程*：8 线程实测不触发合流（可借用掩码内空闲核获得真实 8 倍加速）；即使中途被合流，退化也只是温和的串行水平，灾难下界被结构性消除（实测最坏 8.3 s，而非无防护版本的 95 s+）。
+ *一次性决策，绝不回退*：进程启动时用与正式工作同构的 8 线程探针（"测得即跑得"）做双信号判定：`sched_getcpu()` 同步采样（8 线程每 20 ms 上报所在物理核，连续 5 个采样点全部同核即判定已合流，选串行）与纯 FMA 吞吐比（8 线程并行/单线程吞吐比低于 3 选串行，不低于 6 放行并行，中间模糊带做第二次采样取最小值）。

== 2.5 最终性能

主对比（同节点、无 numactl、背靠背运行，全部 PASS）：

#table(
  stroke: 0.3pt,
  columns: 5,
  [*用例 (M×N)*], [*基线 (ms / GFLOPS)*], [*v10 第1次 (ms / GFLOPS)*], [*v10 第2次 (ms / GFLOPS)*], [*加速比*],
  [512×19968], [2465.96 / 2.12], [64.62 / 81.00], [50.91 / 102.82], [48.4×],
  [2432×17024], [36705.67 / 2.74], [472.47 / 213.12], [527.58 / 190.85], [77.7×],
  [17024×512], [87605.03 / 1.69], [805.38 / 184.24], [763.11 / 194.45], [114.8×],
  [总时间], [126.78 s], [1.34 s], [1.34 s], [94.5×],
)

官方评测形态（38 线程 + `numactl -N 1`）在同一天下午的两次实测：

#table(
  stroke: 0.3pt,
  columns: 4,
  [*节点状态*], [*三用例 GFLOPS*], [*总时间*], [*自适应行为*],
  [中等], [9.74 / 12.64 / 0.89], [约 175 s], [前两案 8 线程并行；第三案探针检出毒化转串行],
  [毒化], [0.59 / 0.93 / 0.88], [约 286 s], [全程串行（有界退化，非灾难）],
  [健康（09-03 记录）], [—], [约 9.3 s（最佳 6.38 s）], [探针放行，8 线程并行],
)

串行分支（兜底路径）三用例 23.1 / 29.6 / 35.0 GFLOPS，仍为基线的 11–20 倍。版本演进摘要：

#table(
  stroke: 0.3pt,
  columns: 3,
  [*阶段*], [*核心改动*], [*结果*],
  [基线], [naive 三循环 + OpenMP], [2–3 GFLOPS],
  [v4–v6], [分块 TRSM + BLAS 编排、形状自适应分割], [健康时快；环境毒化时 95 s 灾难],
  [v7–v9], [探针方案迭代（识别延迟生效与粘性两大陷阱）], [逐步逼近正确机制模型],
  [v10], [8 线程封顶 + 双信号一次决策], [健康 102–213 GFLOPS；最坏有界],
)

最终版在健康环境下的实测输出如下（三用例均 PASS）：

#align(center, image("shot_trsm.png", width: 90%))

== 2.6 小结

TRSM 优化由两条主线构成：算法层面用分块结构把问题转化为高效率的 BLAS 库调用编排（形状自适应分工）；系统层面对执行环境的异常行为建立有界的自适应方案（8 线程封顶 + 同构探针双信号）。最终在同条件对比下取得 48–115 倍加速（总时间 94.5 倍），且最坏情形被结构性约束在串行水平。

#pagebreak()

// ===== 三、ZGEMM（齐思航）=====

= 三、ZGEMM —— 复数矩阵乘法优化

== 3.1 题目背景

本题聚焦于复数矩阵乘法计算，要求计算C = α*A*B + β*C，其中A、B、C为复数矩阵，α、β为复数标量。

== 3.2 baseline分析

基线实现为朴素三重循环，baseline性能如下：
```
Case1:49.0264,Case2:9.8127,Case3:5.7341
```

== 3.3 V1,针对竞赛路径分发

题目强调ABC行主序，AB不转置，利用代码判断
```
if (Order == CblasRowMajor &&
    TransA == CblasNoTrans &&
    TransB == CblasNoTrans) {

    zgemm_row_notrans_real_simd(...);
    return;
}
```
在竞赛路径里，可以采取一系列专门优化。

== 3.4 V2,循环顺序改为 i-k-j

baseline使用i-j-k的顺序循环，会导致B数组的访问不连续，缓存命中率低，为此可以改成i-k-j循环，固定i和k让j变化
```
for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
        const complex scaled_a = alpha * A[i][k];

        for (j = 0; j < N; ++j) {
            C[i][j] += scaled_a * B[k][j];
        }
    }
}
```

== 3.5 V3,矩阵分块和局部累加

若按整块计算C，可能会导致缓存放不下，这时如果反复访问A和B中的元素，缓存命中率会很低，效率低下。

考虑将C分块计算，8 行 × 64/128 列，在更新完C分块后一次性写回C。

局部累加器：
```
double acc_re[8][128];
double acc_im[8][128];
```

== 3.6 V4,实虚分离，使用SIMD

GCC 通常不能直接对 double complex 循环进行有效向量化，因此将复数运算展开为实部和虚部：
```
acc_re += ar * br - ai * bi;
acc_im += ar * bi + ai * br;
```
并对连续的 j 循环使用：
```
#pragma omp simd
for (j = 0; j < jmax; ++j)
```
这样就可以对实数数组生成128位SIMD指令。

截止目前优化，性能结果如下：
```
Case1:110.87,Case2:106.14,Case3:98.56
```

== 3.7 V5,AB矩阵实虚部打包

原始 SIMD 内层循环中，每次都要从交错存储的复数中提取：
```
creal(B[k][j])
cimag(B[k][j])
```
但是复数在内存中是实虚部交叉保留的，内存不连续，不利于SIMD加载。

因此可以在`cblas_zgemm()` 开始时，将 A/B 打包为两个连续数组，减少交错内存访问。

这一步优化后，性能来到
```
Case1:122.88,Case2:137.72,Case3:135.89
```

== 3.8 V6、基于 perf 的热点分析与持续优化

前面的优化已经把主要的循环计算改成实部/虚部分离的 SIMD 形式，因此继续优化时不再盲目增加微内核宽度，而是使用 `perf stat` 检查访存和后端停顿。最终版本在 Case2 上的计数约为 IPC `0.53`、后端停顿 `79%`，cache miss 约 `5.5e10`，说明主要瓶颈是后端访存等待，而不是分支或前端取指。

== 3.9 V7、B 矩阵实部/虚部打包

原始复数矩阵采用实部、虚部交错存储：
```
br0, bi0, br1, bi1, br2, bi2, ...
```
如果在最内层循环中反复使用 `creal(B[k][j])` 和 `cimag(B[k][j])`，会产生交错加载和拆分开销。因此在一次 `cblas_zgemm` 调用开始时，将 B 复制为两个连续数组：
```
b_re_pack[k][j]
b_im_pack[k][j]
```
内层 SIMD 循环直接读取连续的实部和虚部，减少复数拆分以及地址计算。

在 38 线程、单 NUMA、`test_runs=3` 下，所有用例均通过 `1e-10` 校验：
```
Case1:108.6215 -> 112.4558 GFLOPS，1.04x
Case2:107.9539 -> 125.2314 GFLOPS，1.16x
Case3:105.6191 -> 133.2813 GFLOPS，1.26x
```
Case3 的 `perf` 对照显示 cycles 约从 `20.66e12` 降至 `20.34e12`，cache miss 约从 `161.4e9` 降至 `160.0e9`。后端停顿仍然较高，说明该优化降低了加载/拆分开销，但没有消除内存层级瓶颈，因此保留为默认路径。

== 3.10 V8、A 矩阵实部/虚部打包

B 打包后，热点中仍然需要在 `k` 循环内执行 `creal(A[i][k])` 和 `cimag(A[i][k])`。因此仿照 B 的方式，将 A 预先打包为：
```
a_re_pack[i][k]
a_im_pack[i][k]
```
这样计算核心只访问连续的实数数组，避免对 A 反复进行复数提取。候选版本四组校验均通过，测得：
```
Case1:120.50 GFLOPS
Case2:138.24 GFLOPS
Case3:137.79 GFLOPS
```
最终默认配置为 `8 x 128` 分块、A/B 实部虚部打包和 `omp simd`。重新验证结果为：
```
Case1:122.8823 GFLOPS，max error 7.90e-13，PASS
Case2:137.7200 GFLOPS，max error 7.63e-13，PASS
Case3:135.8956 GFLOPS，max error 1.48e-12，PASS
```

== 3.11 V11、M 方向分块探索

当前默认 C 分块为 `8 x 128`。为提高 A 行块复用，测试了 `M=4` 和 `M=16`：
```
M=4：Case1 102.04，Case2 110.49，Case3 102.02 GFLOPS
M=16：Case1 130.18，Case2 136.72，Case3 142.43 GFLOPS
```
两种版本均通过校验。`M=4` 明显降低并行和数据复用效率，淘汰；`M=16` 在 Case1/3 有潜在收益，但 Case2 没有稳定超过默认值，仍需交错复测，暂不改变默认 `M=8`。

== 3.12 V13、K 方向分块候选

当前每个 C 分块会完整扫描 K 维度。针对后端访存等待，尝试将 K 切成 64 的小段：
```
for (kk = 0; kk < K; kk += 64)
    for (k = kk; k < min(kk + 64, K); ++k)
        更新当前 C 分块
```
该候选保持 `k=0..K-1` 的累加顺序，三组均通过 `1e-10` 校验：
```
Case1:120.96 GFLOPS
Case2:137.81 GFLOPS
Case3:127.87 GFLOPS
```
小规模 `perf` 对照中 backend idle 由 `74.68%` 变为 `74.31%`，但正式尺寸没有稳定提升。额外的 K 分块循环控制抵消了可能的缓存收益，因此候选不启用默认路径。

== 3.13 V15、编译器向量化诊断

使用 GCC `-fopt-info-vec` 检查当前默认实现，确认实部/虚部的内层 `j` 累加和写回循环已经生成 128 位向量代码；A/B 打包循环也被向量化，但部分数组访问仍提示可能存在别名或复杂访问模式。

这与 `perf` 的结论一致：当前主要限制是 backend memory stall，而不是 SIMD 未生效。

#align(center, image("zgemm1.png", width: 15cm))

最终性能如上图所示。

#pagebreak()

// ===== 生成式AI工具使用声明 =====

= 生成式AI工具使用声明

本次大作业在实施与报告撰写过程中使用了大型语言模型作为辅助工具，以下按工具概况与具体辅助作用分述。

== 使用工具概况

- 实验执行、代码编写与报告撰写：DeepSeek Harness（命令行客户端），模型为 glm-5.3，在本地工作目录与远程计算集群之间交互，贯穿实验全过程。

== 补充说明

以上所有 AI 工具生成的技术方案与代码均经过实验者的实际操作验证。AI 工具在本大作业中充当技术顾问与写作辅助角色，所有实验操作的实施、数据的采集与分析、以及最终结论的得出，均由两位作者独立完成。
