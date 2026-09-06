// HPC101 期末答辩幻灯片 — typst 手写模板（16:9）
// 编译： typst compile slides.typ
// 与报告同字体体系：Times New Roman/Georgia + 宋体 fallback

#let accent = rgb("#1F4E8C")

#set page(
  paper: "presentation-16-9",
  margin: (left: 2.4cm, right: 2.4cm, top: 1.5cm, bottom: 1.3cm),
  footer: context [
    #if counter(page).get().first() > 1 [
      #text(size: 11pt, fill: gray.darken(45%))[ZJUSCT / HPC101 · 期末大作业答辩]
      #h(1fr)
      #text(size: 11pt, fill: gray.darken(45%))[#counter(page).display()]
    ]
  ],
)

#set text(
  font: ("Times New Roman", "Georgia"),
  size: 19pt,
  lang: "en",
)
#set par(justify: false, leading: 0.7em, spacing: 0.85em)
#show raw.where(block: true): set text(size: 13pt, font: ("Courier New", "NSimSun", "SimSun", "Microsoft YaHei UI"))
#show table.cell: set text(size: 15pt)
#show table.cell: set par(leading: 0.3em, spacing: 0.35em)

// 幻灯片标题组件
#let stitle(title) = [
  #v(0.05em)
  #text(size: 26pt, weight: "bold", fill: accent)[#title]
  #v(-0.15em)
  #line(length: 100%, stroke: 1pt + accent.lighten(55%))
  #v(0.45em)
]

// 章节分隔页
#let divider(big, small) = [
  #align(center + horizon)[
    #v(1fr)
    #text(size: 44pt, weight: "bold", fill: accent)[#big]
    #v(0.35em)
    #text(size: 22pt)[#small]
    #v(1fr)
  ]
  #pagebreak()
]

// ========= S1 封面 =========
#align(center + horizon)[
  #v(1fr)
  #text(size: 40pt, weight: "bold")[HPC101 期末大作业答辩]
  #v(0.4em)
  #text(size: 24pt)[鲲鹏 920F 三算子优化：CONV · TRSM · ZGEMM]
  #v(1.2em)
  #image("浙江大学.png", width: 24%)
  #v(1.2em)
  #text(size: 22pt)[齐思航　王传宇]
  #v(0.3em)
  #text(size: 18pt, fill: gray.darken(30%))[2026-09-06]
  #v(1fr)
]
#pagebreak()

// ========= S2 总览 =========
#stitle[总览]
- 平台：鲲鹏 920F 开发集群（ARMv9 SVE2，608 核 / 32 NUMA × 38 核 @ 2.0 GHz）
- 技术栈：C + OpenMP，NEON SIMD / 循环分块 / BLAS 库编排 / 环境自适应线程
- 分工：CONV、ZGEMM —— 齐思航；TRSM —— 王传宇

#v(0.4em)
#table(
  stroke: 0.4pt,
  columns: 5,
  [*题目*], [*负责人*], [*基线 GFLOPS*], [*最终 GFLOPS*], [*加速比*],
  [CONV], [齐思航], [12.95], [748–886], [58–68×],
  [TRSM], [王传宇], [1.7–2.7], [102–213], [48–115×],
  [ZGEMM], [齐思航], [5.7–49], [160–207], [2.5–24×],
)
#v(0.3em)
#text(size: 15pt, fill: gray.darken(30%))[三题正确性全部通过官方校验；最终区间取自各题性能图。]
#pagebreak()

// ========= 题目一 =========
#divider([题目一 · CONV], [二维卷积优化 · 齐思航])

#stitle[问题与基线]
- 任务：给图像与卷积核，求二维卷积输出（4 组不同尺寸的用例）
- 基线：朴素四重循环 + 外层 OpenMP 并行

#v(0.3em)
#table(
  stroke: 0.4pt,
  columns: 2,
  [*配置*], [*GFLOPS*],
  [OMP 1 线程], [1.00],
  [OMP 38 线程], [12.95],
)
#v(0.3em)
- 环境约束：donau 调度下单 NUMA 绑定会使 38 线程坍缩到单核，故按自由环境 38 线程评测
#pagebreak()

#stitle[V1 优化（一）：编译器友好化]
- 局部累加：`float sum` 临时量，消除对输出数组的反复读写
- `restrict`：声明 input / kernel / output 不重叠，解放寄存器分配与向量化
- OpenMP 静态调度：按输出行分配线程，降低调度开销
- 指针提升：预计算行指针 `inputRow` / `kernelRow`，内层免乘法寻址
#pagebreak()

#stitle[V1 优化（二）：NEON SIMD 四列同算]
- `float32x4_t` 思路：一次计算连续 4 个输出列

```c
float sum0, sum1, sum2, sum3;
for (int jk = 0; jk < kernelHeight; ++jk)
    for (int ik = 0; ik < kernelWidth; ++ik) {
        float w = kernel[jk * kernelWidth + ik];
        sum0 += input[(j+jk)*W + (i+ik)]   * w;
        sum1 += input[(j+jk)*W + (i+ik+1)] * w;
        sum2 += input[(j+jk)*W + (i+ik+2)] * w;
        sum3 += input[(j+jk)*W + (i+ik+3)] * w;
    }
```

- V1 后：12.95 → *165.57 GFLOPS*（12.8×）
#pagebreak()

#stitle[V2 优化（一）：输出列分块 + 四行融合]
- 输出列分块：一次处理 16 列 = 4 个 NEON 向量 × 4 组累加器
- 规模探索：8 列并行不足，20/32/64 列寄存器压力过大 —— 16 列是平衡点
- 四行融合：相邻输出行的卷积核窗口部分重叠，一次加载复用于 4 行

```c
CONV_ROW4_VP_FMA(s0, v, w, 0);
CONV_ROW4_VP_FMA(s1, v, w, 1);
CONV_ROW4_VP_FMA(s2, v, w, 2);
CONV_ROW4_VP_FMA(s3, v, w, 3);
```

- 实测四行优于二行/六行，单步 +22% → *621.42 GFLOPS*
#pagebreak()

#stitle[V2 优化（二）：输入滑动窗口]
- 相邻 `ik` 的输入窗口高度重叠，重复加载浪费带宽：

```
ik = 0: input[0..3]
ik = 1: input[1..4]
ik = 2: input[2..5]
```

- 三步滑窗：加载 5 个相邻向量 → `vextq_f32` 生成 4 个滑动窗口 → 分别完成 4 个卷积位置
- 第二轮优化后：*809.31 GFLOPS*（62× vs 基线；四用例 748–886，全部 PASS）
#pagebreak()

#stitle[CONV 最终性能]
#align(center, image("conv.png", width: 62%))
#text(size: 15pt, fill: gray.darken(30%))[四用例 748–886 GFLOPS，全部 PASS]
#pagebreak()

// ========= 题目二 =========
#divider([题目二 · TRSM], [三角方程组求解优化 · 王传宇])

#stitle[问题与基线]
- 任务：行主序下求解 $L X = B$（$L$ 为 $M times M$ 下三角），计算量约 $M^2 N$
- 三用例覆盖三种形状：$512 times 19968$（宽）、$2432 times 17024$（方）、$17024 times 512$（高）
- 基线：逐列前代消元 + 列间 OpenMP → 2.12 / 2.74 / 1.69 GFLOPS
- 瓶颈：行主序下按列跨步访存 $B$、标量点积无法向量化、列内依赖链限制并行
#pagebreak()

#stitle[优化一：分块算法 + BLAS 库编排]
- $L$ 沿对角切成宽度 512 的条带，串行推进，每步两段：
  - *对角段*：按列切片并行，每线程一次单线程 `cblas_dtrsm` 解出自己的列条带
  - *更新段*：Schur 补更新 $B_"rest" -= L_"rest,k" times X_k$，全部转化为规则 `dgemm`
- 形状自适应：目标块扁则按列分割广播 $L$，高则按行分割广播 $X_k$
- 库调用单线程化（`BlasSetNumThreads(1)`），外层 OpenMP 编排 8 个并发库调用
- 健康环境即达 96–213 GFLOPS
#pagebreak()

#stitle[更新段细节：三路连续访存的规则 GEMM]
- 更新段占高瘦用例 *约 97%* 计算量（$1 - 512/17024$），是性能主战场

```c
for (r = 0; r < below; ++r)          // 目标 B 的第 r 行
    for (j = 0; j < kb; ++j) {
        double w = L_ik[r][j];        // L：按行连续
        for (c = 0; c < n; ++c)
            B[r][c] -= w * X[j][c];   // B、X：按行连续
    }
```

#text(size: 14pt, fill: gray.darken(30%))[示意：与库 `cblas_dgemm` 内核等价的循环序]
- 三操作数全部按行流动：缓存行满载，无跨步浪费
- 内层"标量 × 向量"是 SIMD 完美形状（SVE 512-bit = 8 double/指令）
- 基线对照：累加目标只有一个标量 → 依赖链锁死、跨步访存
- $L$ 面板全程只流一遍；基线每解一列全量重读 $L$
#pagebreak()

#stitle[优化二之发现：合流坍缩]
- 现象：官方命令（38 线程 + `numactl -N 1` 单 NUMA）运行数秒后，全部线程被强制调度到同一物理核
  - 吞吐下降 20–30 倍，且*粘性*：同进程后续单线程也被降速
- 机理：集群亲和域内核对"掩码内严重过订阅"的强制合流，评估周期约 2 s，≤8 线程不触发
- 两个硬约束：
  - 短探针会被延迟生效期骗过（探针时正常，正式工作开始后坍缩）
  - "先试并行、失败退串行"不可行——试探本身触发粘性污染
#pagebreak()

#stitle[v10 自适应方案]
- *正式工作封顶 8 线程*：实测不触发合流，可借掩码内空闲核获得真实 8× 加速；即使被合流也只是退化为串行水平（最坏 8.3 s，无防护版本 95 s+）
- *一次性决策，绝不回退*：与正式工作同构的 8 线程探针（"测得即跑得"）
- 双信号判定：
  - `sched_getcpu()` 采样：8 线程每 20 ms 上报所在核，连续 5 个采样点全部同核 → 串行
  - FMA 吞吐比：低于 3 选串行，不低于 6 放行并行，模糊带二次采样取最小
#pagebreak()

#stitle[TRSM 最终性能：同条件对比]
#table(
  stroke: 0.4pt,
  columns: 5,
  [*用例 (M×N)*], [*基线*], [*v10 第1次*], [*v10 第2次*], [*加速比*],
  [512×19968], [2.12 GFLOPS], [81.00], [102.82], [48.4×],
  [2432×17024], [2.74 GFLOPS], [213.12], [190.85], [77.7×],
  [17024×512], [1.69 GFLOPS], [184.24], [194.45], [114.8×],
)
#v(0.2em)
- 总时间 126.78 s → 1.34 s（*94.5×*）
- 官方形态自适应：中等天气前两案并行、第三案探针转串行；毒化天气全程串行（有界，无灾难）
- 串行兜底分支 23.1 / 29.6 / 35.0 GFLOPS，仍为基线 11–20 倍
#pagebreak()

#stitle[TRSM 最终版运行输出（健康环境）]
#align(center, image("shot_trsm.png", width: 88%))
#text(size: 15pt, fill: gray.darken(30%))[三用例 102.8 / 190.9 / 194.5 GFLOPS，全部 PASS]
#pagebreak()

// ========= 题目三 =========
#divider([题目三 · ZGEMM], [复数矩阵乘法优化 · 齐思航])

#stitle[问题与基线]
- 任务：$C = alpha A B + beta C$，A、B、C 为复数矩阵，行主序、不转置（竞赛路径）
- 基线：朴素三重循环 → 49.03 / 9.81 / 5.73 GFLOPS
- 复数交错存储（实虚交替）是后续 SIMD 优化的主要障碍
#pagebreak()

#stitle[V1–V2：竞赛路径分发 + 循环重排]
- V1：行主序、不转置的竞赛路径，分发到专优化实现

```c
if (Order == CblasRowMajor &&
    TransA == CblasNoTrans &&
    TransB == CblasNoTrans) {
    zgemm_row_notrans_real_simd(...);
    return;
}
```

- V2：i-j-k 改为 i-k-j，B 数组按行连续访问，缓存命中提升

```c
for (i) for (k) {
    const complex scaled_a = alpha * A[i][k];
    for (j = 0; j < N; ++j)
        C[i][j] += scaled_a * B[k][j];
}
```
#pagebreak()

#stitle[V3–V4：分块 + 实虚分离]
- V3：C 按 8×128 分块计算、一次性写回；局部累加器驻留

```c
double acc_re[8][128];
double acc_im[8][128];
```

- V4：复数展开为实/虚两部，`omp simd` 生成 128 位 SIMD

```c
acc_re += ar * br - ai * bi;
acc_im += ar * bi + ai * br;
#pragma omp simd
for (j = 0; j < jmax; ++j) ...
```

- 实测：49.03 / 9.81 / 5.73 → *110.87 / 106.14 / 98.56 GFLOPS*
#pagebreak()

#stitle[V5–V6：实虚部打包初试 + perf 定位]
- V5：A/B 拆为连续实虚数组，消除交错加载 → 初测 *122.88 / 137.72 / 135.89 GFLOPS*
- V6：`perf stat` 热点分析，确认方向后再深化
  - IPC 0.53、后端停顿 79%、cache miss 约 5.5e10
  - 结论：*memory-bound*，瓶颈在后端访存等待，而非 SIMD 未生效
  - 方法论：不盲目加微内核宽度，转向访存优化
#pagebreak()

#stitle[V7–V8：打包深化与终验]
- V7 B 打包（单 NUMA 38 线程，对照实测）：Case1 1.04×、Case2 1.16×、Case3 1.26×
  - 后端停顿仍高：降低了拆分开销，未消除内存层级瓶颈 → 保留为默认路径
- V8 A 打包：计算核心只访问连续实数数组
- 最终配置：`8×128` 分块 + A/B 实虚部打包 + `omp simd`
- 终验（单 NUMA 38 线程）：*122.88 / 137.72 / 135.90 GFLOPS*，误差 7.90e-13 / 7.63e-13 / 1.48e-12，全部 PASS
#pagebreak()

#stitle[V11 / V13：探索与放弃]
- V11 M 方向分块（C 分块默认 8×128，提高 A 行块复用）：

#table(
  stroke: 0.4pt,
  columns: 4,
  [*配置*], [*Case1*], [*Case2*], [*Case3*],
  [M=4], [102.04], [110.49], [102.02],
  [M=16], [130.18], [136.72], [142.43],
)

- M=4 明显降低复用效率，淘汰；M=16 在 Case2 未稳定超过默认 → 保守保 M=8
- V13 K 方向分块（64 段）：120.96 / 137.81 / 127.87，backend idle 74.68% → 74.31% 无稳定提升——循环控制开销抵消缓存收益，弃用
#pagebreak()

#stitle[V15 诊断 + ZGEMM 最终性能]
- `-fopt-info-vec` 确认内层已 128 位向量化，限制在 backend memory stall
- 登录节点复测（同代码，误差指纹一致）：*160.24 / 206.47 / 207.19 GFLOPS*
#align(center, image("zgemm1.png", width: 78%))
#pagebreak()

// ========= 总结 =========
#stitle[总结与方法论]
- 三题全部通过正确性校验，加速比 2.5–115×
- 测量纪律：min-of-N、同环境对比、`perf` 先定位再动手
- 优化针对实测瓶颈，不追求"看起来更高级"的写法（ZGEMM 向量化尾段两次失败即回退）
- 环境适应：TRSM 的同构探针一次决策，把最坏情形结构性约束在串行水平
- 生成式 AI 工具声明：DeepSeek Harness（glm-5.3）辅助，全部方案经实验验证
#pagebreak()

// ========= 致谢 =========
#align(center + horizon)[
  #v(1fr)
  #text(size: 48pt, weight: "bold", fill: accent)[谢谢聆听]
  #v(0.5em)
  #text(size: 22pt)[欢迎提问]
  #v(1fr)
]
