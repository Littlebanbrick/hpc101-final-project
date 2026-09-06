#!/bin/sh
# TRSM 三角方程组求解优化 — 编译并运行全部测试用例
# 平台: 鲲鹏 920F (ARMv9 SVE2)
# 运行方式按竞赛文档要求: OMP_NUM_THREADS=38 numactl -N 1
# 兼容: 评测环境 (HPKit/kblas) 与开发集群 (NSCCSZ/kplblas); POSIX sh 可执行
set -eu

cd "$(dirname "$0")"
# 判题机默认环境直接可用 (参考: CONV 赛题 run.sh 裸 gcc 即可); 仅在极简环境下补 PATH
command -v gcc >/dev/null 2>&1 || export PATH="/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:${PATH:-}"

# 初始化 module 命令 (非交互/非登录 shell 中不可用时; /etc/profile 最通用)
if ! command -v module >/dev/null 2>&1; then
    . /etc/profile 2>/dev/null \
        || . /usr/share/Modules/init/sh 2>/dev/null \
        || . /usr/share/Modules/init/bash 2>/dev/null \
        || . /usr/share/lmod/lmod/init/sh 2>/dev/null \
        || . /etc/profile.d/modules.sh 2>/dev/null \
        || true
fi

rm -f trsm_test
TRSM_BIN=""

# ---- 环境 1: 华为 HPKit (评测环境) ----
for MODPATH in /home/HPC/HPCKit/latest/modulefiles /home/HPC/HPKit/latest/modulefiles; do
    [ -d "$MODPATH" ] || continue
    module use "$MODPATH" 2>/dev/null || true
done
module load gcc/compiler12.3.1/gccmodule 2>/dev/null || true
module load gcc/kml25.2.0/kblas/multi 2>/dev/null || true
if gcc -O3 -fopenmp bench_trsm.c trsm.c -o trsm_test -lm -lkblas 2>/dev/null; then
    TRSM_BIN=trsm_test
    echo "[run.sh] built with HPKit kblas"
else
    # 兜底: 有界查找 kblas 头文件与库 (-print -quit 找到即停; timeout 防网络盘无限爬)
    FOUND_H=$(timeout 10 find /home/HPC /opt /usr/local -maxdepth 9 -name kblas.h -print -quit 2>/dev/null || true)
    FOUND_L=$(timeout 10 find /home/HPC /opt /usr/local -maxdepth 9 -name 'libkblas.so*' -print -quit 2>/dev/null || true)
    KBLAS_H_DIR=${FOUND_H%/*}
    KBLAS_LIB_DIR=${FOUND_L%/*}
    if [ -n "$FOUND_H" ] && [ -n "$FOUND_L" ]; then
        if gcc -O3 -fopenmp -I"$KBLAS_H_DIR" bench_trsm.c trsm.c -o trsm_test \
             -lm -L"$KBLAS_LIB_DIR" -lkblas -Wl,-rpath,"$KBLAS_LIB_DIR" 2>/dev/null; then
            TRSM_BIN=trsm_test
            echo "[run.sh] built with HPKit kblas (located: $KBLAS_LIB_DIR)"
        fi
    fi
fi

# ---- 环境 2: 深超算 NSCCSZ (开发集群) ----
if [ -z "$TRSM_BIN" ] && [ -d /work_ssd/software/soft/tool/kplblas ]; then
    KPL_LIB=/work_ssd/software/soft/tool/kplblas/920FSVE/lib
    KBLAS_H=/home/share/shenchao_common/kblas   # kblas.h 头文件 (kplblas 模块未附带)
    export LD_LIBRARY_PATH=$KPL_LIB:${LD_LIBRARY_PATH:-}
    gcc -O3 -fopenmp -I"$KBLAS_H" bench_trsm.c trsm.c -o trsm_test \
        -lm -L"$KPL_LIB" -lkplblas -Wl,-rpath,"$KPL_LIB"
    TRSM_BIN=trsm_test
    echo "[run.sh] built with NSCCSZ kplblas"
fi

if [ -z "$TRSM_BIN" ]; then
    echo "[run.sh] ERROR: 未找到可用的 KBLAS 环境 (HPKit / kplblas)" >&2
    exit 1
fi

# numactl 探测: 判题机若为单 NUMA (无 node 1) 或无 numactl, 裸跑降级 (CONV 成功样本即无 numactl)
NUMA_RUN=""
if command -v numactl >/dev/null 2>&1 && numactl -N 1 true 2>/dev/null; then
    NUMA_RUN="numactl -N 1"
fi

OMP_NUM_THREADS=38 $NUMA_RUN ./trsm_test 512 19968 1
OMP_NUM_THREADS=38 $NUMA_RUN ./trsm_test 2432 17024 1
OMP_NUM_THREADS=38 $NUMA_RUN ./trsm_test 17024 512 1
