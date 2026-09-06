#include "kblas.h"
#include <omp.h>

/*
 * TRSM v3: 分块右视算法, 并行结构完全由我们的 OpenMP 管理。
 *
 *   for k0 = 0, bm, 2*bm, ...:              (串行块扫描)
 *     1) 对角块求解 X_k = L_kk^-1 * B_k     (列条带并行:
 *        每线程一个列条带, 调单线程 cblas_dtrsm)
 *     2) 下方更新 B_i -= L_ik * X_k         (行带并行:
 *        每线程一个行带, 调单线程 cblas_dgemm)
 *
 * BlasSetNumThreads(1) 压住库的内部线程:
 *  - 避免 libomp 在 libgomp 区域内嵌套展开 (会灾难性 hang/降速)
 *  - 避开 kblas 多线程在散核环境下的低效扩展
 */

#ifndef TRSM_BM
#define TRSM_BM 256
#endif

static int imin(int a, int b) { return a < b ? a : b; }

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    BlasSetNumThreads(1);

    for (int k0 = 0; k0 < m; k0 += TRSM_BM) {
        const int kb = imin(TRSM_BM, m - k0);
        const double* Lkk = L + (size_t)k0 * lda + k0;
        double* Xk = B + (size_t)k0 * ldb;

        /* 1) 对角块求解: 列条带并行, 每线程单线程 dtrsm */
#pragma omp parallel
        {
            const int t = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            const int j0 = (int)((long)t * n / nt);
            const int j1 = (int)((long)(t + 1) * n / nt);
            if (j1 > j0) {
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, kb, j1 - j0, 1.0, Lkk, lda,
                            Xk + j0, ldb);
            }
        }

        /* 2) 下方更新: 行带并行, 每线程单线程 dgemm */
        const int below = m - k0 - kb;
        if (below > 0) {
#pragma omp parallel
            {
                const int t = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                const int r0 = k0 + kb + (int)((long)t * below / nt);
                const int r1 = k0 + kb + (int)((long)(t + 1) * below / nt);
                if (r1 > r0) {
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                r1 - r0, n, kb, -1.0,
                                L + (size_t)r0 * lda + k0, lda,
                                Xk, ldb,
                                1.0, B + (size_t)r0 * ldb, ldb);
                }
            }
        }
    }
}
