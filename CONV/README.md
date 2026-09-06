# 鲲鹏高性能计算全球挑战赛（S2赛季）CONV 优化赛题

## 编译
``` shell
gcc -O3 bench_conv.c conv2d.c -o conv2d_test -lm -fopenmp
```

## 运行
正式评测节点提供 38 个 CPU 核心。正式评测时直接使用 OpenMP 运行，不加
`numactl -N 1`；该选项是计算节点上的 NUMA 绑定选项，不适用于正式评测节点。

本次四个测试用例，运行方法如下：
``` shell
OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores ./conv2d_test 4096 6144 39 39 1
OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores ./conv2d_test 6144 4096 41 41 1
OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores ./conv2d_test 4256 6390 55 55 1
OMP_NUM_THREADS=38 OMP_PROC_BIND=close OMP_PLACES=cores ./conv2d_test 6390 4256 81 81 1

```

校验结果无误，按性能（GFLOPS）排名。

默认 AArch64 路径使用四行融合、垂直权重打包和 `ik` 四步滑动窗口；如需 A/B 回退到
普通输入加载，编译时增加 `-DCONV_DISABLE_ROW4_SLIDE4`。如需回退到未打包的四行路径，
增加 `-DCONV_DISABLE_ROW4_VPACK`。

## 运行结果

见本目录 `conv.png`：登录节点复测（同一代码），四用例 748–886 GFLOPS，全部 PASS；开发节点实测最终 809.31 GFLOPS（基线 12.95 GFLOPS）。
