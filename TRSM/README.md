# 鲲鹏高性能计算全球挑战赛（S2赛季）TRSM 优化赛题

## 编译
``` shell
module use /work_ssd/software/modulefile/modules
module load kplblas/920FSVE
gcc -O3 -fopenmp -I/home/share/shenchao_common/kblas bench_trsm.c trsm.c -o trsm_test -lm \
    -L/work_ssd/software/soft/tool/kplblas/920FSVE/lib -lkplblas -Wl,-rpath,/work_ssd/software/soft/tool/kplblas/920FSVE/lib
```
注：竞赛原始 README 写 `-lkblas`，实际应使用 `kplblas/920FSVE` 模块（库名 `-lkplblas`）。

## 运行
本次三个测试用例，运行方法如下：
``` shell
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 512 19968 1 #参数依次是 m n test_runs表示重复测试次数，使性能稳定即可
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 2432 17024 1
OMP_NUM_THREADS=38 numactl -N 1 ./trsm_test 17024 512 1
```

## 调优
修改trsm.c源文件的l_trsm函数，优化性能。

校验结果无误，按性能（GFLOPS）排名。

## 运行结果

见本目录 `shot_trsm.png`：健康环境展示运行，三用例 102.82 / 190.85 / 194.45 GFLOPS，全部 PASS（基线 2.12 / 2.74 / 1.69 GFLOPS，总加速 94.5×）。