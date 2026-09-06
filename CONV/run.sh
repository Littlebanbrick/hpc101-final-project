#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
gcc -O3 -fopenmp bench_conv.c conv2d.c -o conv2d_test -lm

threads="${OMP_NUM_THREADS:-38}"
run_case() {
    numactl -N 1 env OMP_NUM_THREADS="$threads" \
        OMP_PROC_BIND="${OMP_PROC_BIND:-close}" \
        OMP_PLACES="${OMP_PLACES:-cores}" ./conv2d_test "$@"
}

run_case 4096 6144 39 39 1
run_case 6144 4096 41 41 1
run_case 4256 6390 55 55 1
run_case 6390 4256 81 81 1
