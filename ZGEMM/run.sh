#!/usr/bin/env bash
# ZGEMM 复数矩阵乘法优化 — 编译并运行全部测试用例
# 平台: 鲲鹏 920F, 无外部依赖, 双环境 (HPKit/NSCCSZ) 通用
# 运行方式按竞赛文档要求: OMP_NUM_THREADS=38 numactl -N 1
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:$PATH

gcc -O3 -fopenmp bench_zgemm.c zgemm.c -o zgemm_test -lm

OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 7427 7427 256 1
OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 14848 14848 256 1
OMP_NUM_THREADS=38 numactl -N 1 ./zgemm_test 37360 8192 512 1
