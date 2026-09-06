#include <complex.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(__aarch64__) && (defined(ZGEMM_ENABLE_NEON) || defined(ZGEMM_ENABLE_NEON_REAL))
#include <arm_neon.h>
#endif

// 类型定义
typedef int BLASINT;
typedef double _Complex zdouble;

// 枚举定义
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

// 辅助函数：获取转置后的维度
static inline BLASINT get_A_rows(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? M : K;
}

static inline BLASINT get_A_cols(enum CBLAS_TRANSPOSE TransA, BLASINT M, BLASINT K)
{
    return (TransA == CblasNoTrans) ? K : M;
}

static inline BLASINT get_B_rows(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? K : N;
}

static inline BLASINT get_B_cols(enum CBLAS_TRANSPOSE TransB, BLASINT K, BLASINT N)
{
    return (TransB == CblasNoTrans) ? N : K;
}

/*
 * Fast path for the only path used by the contest benchmark.  In row-major
 * storage, i-k-j makes both B[k, :] and C[i, :] contiguous and reuses one A
 * element across the whole output row.
 */
static void zgemm_row_notrans_fast(const BLASINT M, const BLASINT N, const BLASINT K,
                                   const zdouble alpha_val,
                                   const zdouble* restrict A_ptr, const BLASINT lda,
                                   const zdouble* restrict B_ptr, const BLASINT ldb,
                                   const zdouble beta_val,
                                   zdouble* restrict C_ptr, const BLASINT ldc)
{
#pragma omp parallel for schedule(static)
    for (BLASINT i = 0; i < M; ++i) {
        const zdouble* restrict a_row = A_ptr + (size_t)i * lda;
        zdouble* restrict c_row = C_ptr + (size_t)i * ldc;

        if (beta_val != 1.0) {
            for (BLASINT j = 0; j < N; ++j) {
                c_row[j] *= beta_val;
            }
        }

        for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
            const zdouble scaled_a = alpha_val * a_row[k_idx];
            const zdouble* restrict b_row = B_ptr + (size_t)k_idx * ldb;
            for (BLASINT j = 0; j < N; ++j) {
                c_row[j] += scaled_a * b_row[j];
            }
        }
    }
}

#ifndef ZGEMM_BLOCK_M
#define ZGEMM_BLOCK_M 16
#endif
#ifndef ZGEMM_BLOCK_N
#define ZGEMM_BLOCK_N 128
#endif
#if defined(ZGEMM_ADAPTIVE_N)
#define ZGEMM_STORAGE_N 256
#else
#define ZGEMM_STORAGE_N ZGEMM_BLOCK_N
#endif

#ifndef ZGEMM_DISABLE_BLOCKED

/* Blocked variant: keep a small C tile in local accumulators and apply alpha
 * and beta only once at write-back. */
static void zgemm_row_notrans_blocked(const BLASINT M, const BLASINT N, const BLASINT K,
                                      const zdouble alpha_val,
                                      const zdouble* restrict A_ptr, const BLASINT lda,
                                      const zdouble* restrict B_ptr, const BLASINT ldb,
                                      const zdouble beta_val,
                                      zdouble* restrict C_ptr, const BLASINT ldc)
{
#pragma omp parallel for schedule(static)
    for (BLASINT ii = 0; ii < M; ii += ZGEMM_BLOCK_M) {
        const BLASINT imax = (ii + ZGEMM_BLOCK_M < M) ? ZGEMM_BLOCK_M : (M - ii);
        const zdouble* restrict a_block = A_ptr + (size_t)ii * lda;

        for (BLASINT jj = 0; jj < N; jj += ZGEMM_BLOCK_N) {
            const BLASINT jmax = (jj + ZGEMM_BLOCK_N < N) ? ZGEMM_BLOCK_N : (N - jj);
            zdouble acc[ZGEMM_BLOCK_M][ZGEMM_BLOCK_N];

            for (BLASINT ib = 0; ib < imax; ++ib) {
                for (BLASINT jb = 0; jb < jmax; ++jb) {
                    acc[ib][jb] = 0.0;
                }
            }

            for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
                const zdouble* restrict b_row = B_ptr + (size_t)k_idx * ldb + jj;
                for (BLASINT ib = 0; ib < imax; ++ib) {
                    const zdouble aik = a_block[(size_t)ib * lda + k_idx];
                    for (BLASINT jb = 0; jb < jmax; ++jb) {
                        acc[ib][jb] += aik * b_row[jb];
                    }
                }
            }

            for (BLASINT ib = 0; ib < imax; ++ib) {
                zdouble* restrict c_row = C_ptr + (size_t)(ii + ib) * ldc + jj;
                for (BLASINT jb = 0; jb < jmax; ++jb) {
                    c_row[jb] = alpha_val * acc[ib][jb] + beta_val * c_row[jb];
                }
            }
        }
    }
}
#endif

#if !defined(ZGEMM_DISABLE_REAL_SIMD)
static void zgemm_row_notrans_real_simd(const BLASINT M, const BLASINT N, const BLASINT K,
                                         const zdouble alpha_val,
                                         const zdouble* restrict A_ptr, const BLASINT lda,
                                         const zdouble* restrict B_ptr, const BLASINT ldb,
                                         const zdouble beta_val,
                                         zdouble* restrict C_ptr, const BLASINT ldc)
{
    const double alpha_re = creal(alpha_val);
    const double alpha_im = cimag(alpha_val);
    const double beta_re = creal(beta_val);
    const double beta_im = cimag(beta_val);
#if !defined(ZGEMM_DISABLE_PACK_A)
    double* a_re_pack = (double*)malloc((size_t)M * K * sizeof(double));
    double* a_im_pack = (double*)malloc((size_t)M * K * sizeof(double));
    if (a_re_pack == NULL || a_im_pack == NULL) {
        free(a_re_pack);
        free(a_im_pack);
        zgemm_row_notrans_blocked(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb,
                                  beta_val, C_ptr, ldc);
        return;
    }
#pragma omp parallel for schedule(static)
    for (BLASINT i = 0; i < M; ++i) {
        const zdouble* restrict a_src = A_ptr + (size_t)i * lda;
        double* restrict a_re_dst = a_re_pack + (size_t)i * K;
        double* restrict a_im_dst = a_im_pack + (size_t)i * K;
        for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
            a_re_dst[k_idx] = creal(a_src[k_idx]);
            a_im_dst[k_idx] = cimag(a_src[k_idx]);
        }
    }
#endif
#if !defined(ZGEMM_DISABLE_PACK_B)
    double* b_re_pack = NULL;
    double* b_im_pack = NULL;
#if defined(ZGEMM_ALIGN_PACK_B)
    if (posix_memalign((void**)&b_re_pack, 64, (size_t)K * N * sizeof(double)) != 0) {
        b_re_pack = NULL;
    }
    if (posix_memalign((void**)&b_im_pack, 64, (size_t)K * N * sizeof(double)) != 0) {
        b_im_pack = NULL;
    }
#else
    b_re_pack = (double*)malloc((size_t)K * N * sizeof(double));
    b_im_pack = (double*)malloc((size_t)K * N * sizeof(double));
#endif
    if (b_re_pack == NULL || b_im_pack == NULL) {
        free(b_re_pack);
        free(b_im_pack);
#if !defined(ZGEMM_DISABLE_PACK_A)
        free(a_re_pack);
        free(a_im_pack);
#endif
        zgemm_row_notrans_blocked(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb,
                                  beta_val, C_ptr, ldc);
        return;
    }
#pragma omp parallel for schedule(static)
    for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
        const zdouble* restrict b_src = B_ptr + (size_t)k_idx * ldb;
        double* restrict b_re_dst = b_re_pack + (size_t)k_idx * N;
        double* restrict b_im_dst = b_im_pack + (size_t)k_idx * N;
        for (BLASINT j = 0; j < N; ++j) {
            b_re_dst[j] = creal(b_src[j]);
            b_im_dst[j] = cimag(b_src[j]);
        }
    }
#endif
#pragma omp parallel for schedule(static)
    for (BLASINT ii = 0; ii < M; ii += ZGEMM_BLOCK_M) {
        const BLASINT imax = (ii + ZGEMM_BLOCK_M < M) ? ZGEMM_BLOCK_M : (M - ii);
        for (BLASINT jj = 0; jj < N; jj += ZGEMM_BLOCK_N) {
            const BLASINT jmax = (jj + ZGEMM_BLOCK_N < N) ? ZGEMM_BLOCK_N : (N - jj);
            const zdouble* restrict a_block = A_ptr + (size_t)ii * lda;
            double acc_re[ZGEMM_BLOCK_M][ZGEMM_BLOCK_N];
            double acc_im[ZGEMM_BLOCK_M][ZGEMM_BLOCK_N];
            for (BLASINT ib = 0; ib < imax; ++ib) {
                for (BLASINT jb = 0; jb < jmax; ++jb) {
                    acc_re[ib][jb] = 0.0;
                    acc_im[ib][jb] = 0.0;
                }
            }
#if defined(ZGEMM_K_BLOCK)
            for (BLASINT kk = 0; kk < K; kk += ZGEMM_K_BLOCK) {
                const BLASINT kend = (kk + ZGEMM_K_BLOCK < K) ? (kk + ZGEMM_K_BLOCK) : K;
                for (BLASINT k_idx = kk; k_idx < kend; ++k_idx) {
#else
            for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
#endif
#if !defined(ZGEMM_DISABLE_PACK_B)
                const double* restrict b_re_row = b_re_pack + (size_t)k_idx * N + jj;
                const double* restrict b_im_row = b_im_pack + (size_t)k_idx * N + jj;
#else
                const zdouble* restrict b_row = B_ptr + (size_t)k_idx * ldb + jj;
#endif
                for (BLASINT ib = 0; ib < imax; ++ib) {
                    const zdouble aik = a_block[(size_t)ib * lda + k_idx];
#if !defined(ZGEMM_DISABLE_PACK_A)
                    const double ar = a_re_pack[(size_t)(ii + ib) * K + k_idx];
                    const double ai = a_im_pack[(size_t)(ii + ib) * K + k_idx];
#else
                    const double ar = creal(aik);
                    const double ai = cimag(aik);
#endif
#if defined(ZGEMM_SIMD_ALIGNED)
#pragma omp simd aligned(b_re_row, b_im_row, acc_re, acc_im : 16)
#else
#pragma omp simd
#endif
                    for (BLASINT jb = 0; jb < jmax; ++jb) {
#if !defined(ZGEMM_DISABLE_PACK_B)
                        const double br = b_re_row[jb];
                        const double bi = b_im_row[jb];
#else
                        const double br = creal(b_row[jb]);
                        const double bi = cimag(b_row[jb]);
#endif
                        acc_re[ib][jb] += ar * br - ai * bi;
                        acc_im[ib][jb] += ar * bi + ai * br;
                    }
                }
            }
#if defined(ZGEMM_K_BLOCK)
            }
#endif
            for (BLASINT ib = 0; ib < imax; ++ib) {
                zdouble* restrict c_row = C_ptr + (size_t)(ii + ib) * ldc + jj;
#if defined(ZGEMM_SIMD_ALIGNED)
#pragma omp simd aligned(c_row, acc_re, acc_im : 16)
#else
#pragma omp simd
#endif
                for (BLASINT jb = 0; jb < jmax; ++jb) {
                    const double sr = acc_re[ib][jb];
                    const double si = acc_im[ib][jb];
                    const double cr = creal(c_row[jb]);
                    const double ci = cimag(c_row[jb]);
                    const double out_re = alpha_re * sr - alpha_im * si +
                                          beta_re * cr - beta_im * ci;
                    const double out_im = alpha_re * si + alpha_im * sr +
                                          beta_re * ci + beta_im * cr;
                    c_row[jb] = out_re + out_im * I;
                }
            }
        }
    }
#if !defined(ZGEMM_DISABLE_PACK_B)
    free(b_re_pack);
    free(b_im_pack);
#endif
#if !defined(ZGEMM_DISABLE_PACK_A)
    free(a_re_pack);
    free(a_im_pack);
#endif
}
#endif

#if defined(__aarch64__) && defined(ZGEMM_ENABLE_NEON_REAL)
static inline float64x2_t zgemm_neon_real_mul(const float64x2_t ar,
                                               const float64x2_t ai,
                                               const float64x2_t br,
                                               const float64x2_t bi)
{
    return vfmaq_f64(vmulq_f64(ar, br), ai, vnegq_f64(bi));
}

static inline float64x2_t zgemm_neon_imag_mul(const float64x2_t ar,
                                               const float64x2_t ai,
                                               const float64x2_t br,
                                               const float64x2_t bi)
{
    return vfmaq_f64(vmulq_f64(ar, bi), ai, br);
}

static void zgemm_row_notrans_neon_real(const BLASINT M, const BLASINT N, const BLASINT K,
                                        const zdouble alpha_val,
                                        const zdouble* restrict A_ptr, const BLASINT lda,
                                        const zdouble* restrict B_ptr, const BLASINT ldb,
                                        const zdouble beta_val,
                                        zdouble* restrict C_ptr, const BLASINT ldc)
{
    const double alpha_re = creal(alpha_val);
    const double alpha_im = cimag(alpha_val);
    const double beta_re = creal(beta_val);
    const double beta_im = cimag(beta_val);
#pragma omp parallel for schedule(static)
    for (BLASINT ii = 0; ii < M; ii += ZGEMM_BLOCK_M) {
        const BLASINT imax = (ii + ZGEMM_BLOCK_M < M) ? ZGEMM_BLOCK_M : (M - ii);
        const zdouble* restrict a_block = A_ptr + (size_t)ii * lda;
#if defined(ZGEMM_ADAPTIVE_N)
        const BLASINT block_n = (M >= 10000 || N >= 8000) ? 256 : 128;
#else
        const BLASINT block_n = ZGEMM_BLOCK_N;
#endif
        for (BLASINT jj = 0; jj < N; jj += block_n) {
            const BLASINT jmax = (jj + block_n < N) ? block_n : (N - jj);
            double acc_re[ZGEMM_BLOCK_M][ZGEMM_STORAGE_N];
            double acc_im[ZGEMM_BLOCK_M][ZGEMM_STORAGE_N];
            for (BLASINT ib = 0; ib < imax; ++ib) {
                for (BLASINT jb = 0; jb < jmax; ++jb) {
                    acc_re[ib][jb] = 0.0;
                    acc_im[ib][jb] = 0.0;
                }
            }
            for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
                const zdouble* restrict b_row = B_ptr + (size_t)k_idx * ldb + jj;
                BLASINT jb = 0;
                for (; jb + 1 < jmax; jb += 2) {
                    const float64x2x2_t b = vld2q_f64((const double*)(b_row + jb));
                    for (BLASINT ib = 0; ib < imax; ++ib) {
                        const zdouble av = a_block[(size_t)ib * lda + k_idx];
                        const float64x2_t ar_vec = vdupq_n_f64(creal(av));
                        const float64x2_t ai_vec = vdupq_n_f64(cimag(av));
                        double* re = &acc_re[ib][jb];
                        double* im = &acc_im[ib][jb];
                        const float64x2_t re_old = vld1q_f64(re);
                        const float64x2_t im_old = vld1q_f64(im);
                        vst1q_f64(re, vaddq_f64(re_old,
                            zgemm_neon_real_mul(ar_vec, ai_vec, b.val[0], b.val[1])));
                        vst1q_f64(im, vaddq_f64(im_old,
                            zgemm_neon_imag_mul(ar_vec, ai_vec, b.val[0], b.val[1])));
                    }
                }
                if (jb < jmax) {
                    const zdouble* b = b_row + jb;
                    for (BLASINT ib = 0; ib < imax; ++ib) {
                        const zdouble av = a_block[(size_t)ib * lda + k_idx];
                        acc_re[ib][jb] += creal(av) * creal(*b) - cimag(av) * cimag(*b);
                        acc_im[ib][jb] += creal(av) * cimag(*b) + cimag(av) * creal(*b);
                    }
                }
            }
            for (BLASINT ib = 0; ib < imax; ++ib) {
                zdouble* restrict c_row = C_ptr + (size_t)(ii + ib) * ldc + jj;
                BLASINT jb = 0;
                for (; jb + 1 < jmax; jb += 2) {
                    const float64x2x2_t c = vld2q_f64((const double*)(c_row + jb));
                    const float64x2_t sr = vld1q_f64(&acc_re[ib][jb]);
                    const float64x2_t si = vld1q_f64(&acc_im[ib][jb]);
                    const float64x2_t cr = c.val[0];
                    const float64x2_t ci = c.val[1];
                    const float64x2_t out_re = vaddq_f64(
                        zgemm_neon_real_mul(vdupq_n_f64(alpha_re), vdupq_n_f64(alpha_im), sr, si),
                        zgemm_neon_real_mul(vdupq_n_f64(beta_re), vdupq_n_f64(beta_im), cr, ci));
                    const float64x2_t out_im = vaddq_f64(
                        zgemm_neon_imag_mul(vdupq_n_f64(alpha_re), vdupq_n_f64(alpha_im), sr, si),
                        zgemm_neon_imag_mul(vdupq_n_f64(beta_re), vdupq_n_f64(beta_im), cr, ci));
                    float64x2x2_t out = { { out_re, out_im } };
                    vst2q_f64((double*)(c_row + jb), out);
                }
                for (; jb < jmax; ++jb) {
                    const double sr = acc_re[ib][jb];
                    const double si = acc_im[ib][jb];
                    const double cr = creal(c_row[jb]);
                    const double ci = cimag(c_row[jb]);
                    c_row[jb] = (alpha_re * sr - alpha_im * si + beta_re * cr - beta_im * ci) +
                                (alpha_re * si + alpha_im * sr + beta_re * ci + beta_im * cr) * I;
                }
            }
        }
    }
}
#endif

#if defined(__aarch64__) && defined(ZGEMM_ENABLE_NEON)
static inline float64x2_t zgemm_neon_cmul(const float64x2_t a, const float64x2_t b)
{
    const float64x2_t a_re = vdupq_n_f64(vgetq_lane_f64(a, 0));
    const float64x2_t a_im = vdupq_n_f64(vgetq_lane_f64(a, 1));
    const float64x2_t b_swap = vextq_f64(b, b, 1);
    const float64x2_t real_imag_sign = {-1.0, 1.0};
    const float64x2_t re_part = vmulq_f64(a_re, b);
    const float64x2_t im_part = vmulq_f64(a_im, b_swap);
    return vfmaq_f64(re_part, im_part, real_imag_sign);
}

static void zgemm_row_notrans_neon(const BLASINT M, const BLASINT N, const BLASINT K,
                                   const zdouble alpha_val,
                                   const zdouble* restrict A_ptr, const BLASINT lda,
                                   const zdouble* restrict B_ptr, const BLASINT ldb,
                                   const zdouble beta_val,
                                   zdouble* restrict C_ptr, const BLASINT ldc)
{
    const float64x2_t alpha_vec = vld1q_f64((const double*)&alpha_val);
    const float64x2_t beta_vec = vld1q_f64((const double*)&beta_val);
#pragma omp parallel for schedule(static)
    for (BLASINT ii = 0; ii < M; ii += 2) {
        const BLASINT imax = (ii + 2 < M) ? 2 : (M - ii);
        for (BLASINT jj = 0; jj < N; jj += 4) {
            const BLASINT jmax = (jj + 4 < N) ? 4 : (N - jj);
            if (imax < 2 || jmax < 4) {
                for (BLASINT ib = 0; ib < imax; ++ib) {
                    for (BLASINT jb = 0; jb < jmax; ++jb) {
                        zdouble sum = 0.0;
                        for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
                            sum += A_ptr[(size_t)(ii + ib) * lda + k_idx] *
                                   B_ptr[(size_t)k_idx * ldb + jj + jb];
                        }
                        zdouble* c = C_ptr + (size_t)(ii + ib) * ldc + jj + jb;
                        *c = alpha_val * sum + beta_val * *c;
                    }
                }
                continue;
            }

            float64x2_t acc00 = vdupq_n_f64(0.0);
            float64x2_t acc01 = vdupq_n_f64(0.0);
            float64x2_t acc02 = vdupq_n_f64(0.0);
            float64x2_t acc03 = vdupq_n_f64(0.0);
            float64x2_t acc10 = vdupq_n_f64(0.0);
            float64x2_t acc11 = vdupq_n_f64(0.0);
            float64x2_t acc12 = vdupq_n_f64(0.0);
            float64x2_t acc13 = vdupq_n_f64(0.0);
            for (BLASINT k_idx = 0; k_idx < K; ++k_idx) {
                const float64x2_t a0 = vld1q_f64((const double*)
                    (A_ptr + (size_t)ii * lda + k_idx));
                const float64x2_t a1 = vld1q_f64((const double*)
                    (A_ptr + (size_t)(ii + 1) * lda + k_idx));
                const float64x2_t b0 = vld1q_f64((const double*)
                    (B_ptr + (size_t)k_idx * ldb + jj));
                const float64x2_t b1 = vld1q_f64((const double*)
                    (B_ptr + (size_t)k_idx * ldb + jj + 1));
                const float64x2_t b2 = vld1q_f64((const double*)
                    (B_ptr + (size_t)k_idx * ldb + jj + 2));
                const float64x2_t b3 = vld1q_f64((const double*)
                    (B_ptr + (size_t)k_idx * ldb + jj + 3));
                acc00 = vaddq_f64(acc00, zgemm_neon_cmul(a0, b0));
                acc01 = vaddq_f64(acc01, zgemm_neon_cmul(a0, b1));
                acc02 = vaddq_f64(acc02, zgemm_neon_cmul(a0, b2));
                acc03 = vaddq_f64(acc03, zgemm_neon_cmul(a0, b3));
                acc10 = vaddq_f64(acc10, zgemm_neon_cmul(a1, b0));
                acc11 = vaddq_f64(acc11, zgemm_neon_cmul(a1, b1));
                acc12 = vaddq_f64(acc12, zgemm_neon_cmul(a1, b2));
                acc13 = vaddq_f64(acc13, zgemm_neon_cmul(a1, b3));
            }

            zdouble* c00 = C_ptr + (size_t)ii * ldc + jj;
            zdouble* c10 = C_ptr + (size_t)(ii + 1) * ldc + jj;
            const float64x2_t c0 = vld1q_f64((const double*)c00);
            const float64x2_t c1 = vld1q_f64((const double*)(c00 + 1));
            const float64x2_t c2 = vld1q_f64((const double*)(c00 + 2));
            const float64x2_t c3 = vld1q_f64((const double*)(c00 + 3));
            const float64x2_t c10v = vld1q_f64((const double*)c10);
            const float64x2_t c11v = vld1q_f64((const double*)(c10 + 1));
            const float64x2_t c12v = vld1q_f64((const double*)(c10 + 2));
            const float64x2_t c13v = vld1q_f64((const double*)(c10 + 3));
            acc00 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc00),
                              zgemm_neon_cmul(beta_vec, c0));
            acc01 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc01),
                              zgemm_neon_cmul(beta_vec, c1));
            acc02 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc02),
                              zgemm_neon_cmul(beta_vec, c2));
            acc03 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc03),
                              zgemm_neon_cmul(beta_vec, c3));
            acc10 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc10),
                              zgemm_neon_cmul(beta_vec, c10v));
            acc11 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc11),
                              zgemm_neon_cmul(beta_vec, c11v));
            acc12 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc12),
                              zgemm_neon_cmul(beta_vec, c12v));
            acc13 = vaddq_f64(zgemm_neon_cmul(alpha_vec, acc13),
                              zgemm_neon_cmul(beta_vec, c13v));
            vst1q_f64((double*)c00, acc00);
            vst1q_f64((double*)(c00 + 1), acc01);
            vst1q_f64((double*)(c00 + 2), acc02);
            vst1q_f64((double*)(c00 + 3), acc03);
            vst1q_f64((double*)c10, acc10);
            vst1q_f64((double*)(c10 + 1), acc11);
            vst1q_f64((double*)(c10 + 2), acc12);
            vst1q_f64((double*)(c10 + 3), acc13);
        }
    }
}
#endif

// 核心实现：标准三重循环
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const BLASINT M, const BLASINT N,
                 const BLASINT K, const void* alpha, const void* A, const BLASINT lda,
                 const void* B, const BLASINT ldb, const void* beta, void* C, const BLASINT ldc)
{

    // 转换为复数指针
    const zdouble* A_ptr = (const zdouble*)A;
    const zdouble* B_ptr = (const zdouble*)B;
    zdouble* C_ptr = (zdouble*)C;
    zdouble alpha_val = *(const zdouble*)alpha;
    zdouble beta_val = *(const zdouble*)beta;

    if (Order == CblasRowMajor && TransA == CblasNoTrans && TransB == CblasNoTrans) {
#if !defined(ZGEMM_DISABLE_REAL_SIMD)
        zgemm_row_notrans_real_simd(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb, beta_val,
                                    C_ptr, ldc);
#elif defined(__aarch64__) && defined(ZGEMM_ENABLE_NEON)
        zgemm_row_notrans_neon(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb, beta_val,
                               C_ptr, ldc);
#elif !defined(ZGEMM_DISABLE_BLOCKED)
        zgemm_row_notrans_blocked(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb, beta_val,
                                  C_ptr, ldc);
#else
        zgemm_row_notrans_fast(M, N, K, alpha_val, A_ptr, lda, B_ptr, ldb, beta_val,
                               C_ptr, ldc);
#endif
        return;
    }

    // 实际维度（考虑转置）
    BLASINT A_rows = get_A_rows(TransA, M, K);
    BLASINT A_cols = get_A_cols(TransA, M, K);
    BLASINT B_rows = get_B_rows(TransB, K, N);
    BLASINT B_cols = get_B_cols(TransB, K, N);

    // 根据存储顺序进行矩阵乘法
    if (Order == CblasRowMajor) {
        // ========== 行主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[i * ldc + j] *= beta_val;
                }
            }
        }
#pragma omp parallel for
        // 计算 C = alpha * A * B + C
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx]
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[k_idx * lda + i]);
                    }

                    // 获取 B[k_idx, j]
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[j * ldb + k_idx]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[i * ldc + j] = alpha_val * sum + C_ptr[i * ldc + j];
            }
        }
    } else { // CblasColMajor
        // ========== 列主序存储 ==========
        // 首先缩放 C = beta * C
        if (beta_val != 1.0) {
            for (BLASINT i = 0; i < M; i++) {
                for (BLASINT j = 0; j < N; j++) {
                    C_ptr[j * ldc + i] *= beta_val;
                }
            }
        }

// 计算 C = alpha * A * B + C
#pragma omp parallel for
        for (BLASINT i = 0; i < M; i++) {
            for (BLASINT j = 0; j < N; j++) {
                zdouble sum = 0.0;
                for (BLASINT k_idx = 0; k_idx < K; k_idx++) {
                    // 获取 A[i, k_idx] (列主序)
                    zdouble a_elem;
                    if (TransA == CblasNoTrans) {
                        a_elem = A_ptr[k_idx * lda + i];
                    } else if (TransA == CblasTrans) {
                        a_elem = A_ptr[i * lda + k_idx];
                    } else { // ConjTrans
                        a_elem = conj(A_ptr[i * lda + k_idx]);
                    }

                    // 获取 B[k_idx, j] (列主序)
                    zdouble b_elem;
                    if (TransB == CblasNoTrans) {
                        b_elem = B_ptr[j * ldb + k_idx];
                    } else if (TransB == CblasTrans) {
                        b_elem = B_ptr[k_idx * ldb + j];
                    } else { // ConjTrans
                        b_elem = conj(B_ptr[k_idx * ldb + j]);
                    }

                    sum += a_elem * b_elem;
                }
                C_ptr[j * ldc + i] = alpha_val * sum + C_ptr[j * ldc + i];
            }
        }
    }
}
