#include <string.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#if !defined(CONV_DISABLE_ROW4_VPACK) && !defined(CONV_ENABLE_ROW8) && \
    !defined(CONV_ENABLE_ROW4_W20) && !defined(CONV_ENABLE_ROW4_LANE4)
#define CONV_ENABLE_ROW4_VPACK
#endif
#if defined(CONV_ENABLE_ROW4_VPACK) && !defined(CONV_DISABLE_ROW4_SLIDE4)
#define CONV_ROW4_SLIDE4
#endif
/* Explicit scalar load plus vector broadcast is the stable default.  The
 * LD1R form remains available for an isolated compiler/codegen comparison. */
#if defined(CONV_USE_LD1R)
#define CONV_WEIGHT(ptr) vld1q_dup_f32(ptr)
#else
#define CONV_WEIGHT(ptr) vdupq_n_f32(*(ptr))
#endif
#if defined(CONV_ROW2_PACKED_WEIGHT)
static const float *conv_kernel_base;
static const float32x4_t *conv_packed_kernel_base;
#undef CONV_WEIGHT
#define CONV_WEIGHT(ptr) \
    vld1q_f32((const float *)(conv_packed_kernel_base + ((ptr) - conv_kernel_base)))
#endif
#if !defined(CONV_DISABLE_ROW4_VPACK)
static const float32x4_t *conv_row4_vpack_base;
#endif
#endif

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

#if defined(__aarch64__) && defined(CONV_ENABLE_ROW8)
#define CONV_ROW8_FMA2(S, V, W) do { \
    (S)[0] = vfmaq_f32((S)[0], (V)[0], (W)); \
    (S)[1] = vfmaq_f32((S)[1], (V)[1], (W)); \
} while (0)

static void conv2d_row8(const CONVFLOAT* restrict input, CONVINT inputWidth,
                        const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                        CONVINT kernelWidth, CONVFLOAT* restrict output,
                        CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 8) {
        const CONVINT rows = (outputHeight - j < 8) ? outputHeight - j : 8;
        CONVINT i = 0;
        if (rows < 8) {
            for (; i < outputWidth; ++i) {
                for (CONVINT q = 0; q < rows; ++q) {
                    CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                    *out = 0.0f;
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                            *out += input[(j + q + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
                }
            }
            continue;
        }

        for (; i + 7 < outputWidth; i += 8) {
            float32x4_t s0[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s1[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s2[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s3[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s4[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s5[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s6[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s7[2] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };

            for (CONVINT r = 0; r < 7; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t v[2] = {
                        vld1q_f32(inputRow + ik), vld1q_f32(inputRow + ik + 4)};
                    for (CONVINT q = 0; q <= r; ++q)
                        CONV_ROW8_FMA2((q == 0 ? s0 : q == 1 ? s1 : q == 2 ? s2 :
                                        q == 3 ? s3 : q == 4 ? s4 : q == 5 ? s5 : s6),
                                       v, CONV_WEIGHT(kernel + (r - q) * kernelWidth + ik));
                }
            }

            for (CONVINT r = 7; r < kernelHeight; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + r * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t v[2] = {
                        vld1q_f32(inputRow + ik), vld1q_f32(inputRow + ik + 4)};
                    const float32x4_t w0 = CONV_WEIGHT(krow + ik);
                    const float32x4_t w1 = CONV_WEIGHT(krow - kernelWidth + ik);
                    const float32x4_t w2 = CONV_WEIGHT(krow - 2 * kernelWidth + ik);
                    const float32x4_t w3 = CONV_WEIGHT(krow - 3 * kernelWidth + ik);
                    const float32x4_t w4 = CONV_WEIGHT(krow - 4 * kernelWidth + ik);
                    const float32x4_t w5 = CONV_WEIGHT(krow - 5 * kernelWidth + ik);
                    const float32x4_t w6 = CONV_WEIGHT(krow - 6 * kernelWidth + ik);
                    const float32x4_t w7 = CONV_WEIGHT(krow - 7 * kernelWidth + ik);
                    CONV_ROW8_FMA2(s0, v, w0); CONV_ROW8_FMA2(s1, v, w1);
                    CONV_ROW8_FMA2(s2, v, w2); CONV_ROW8_FMA2(s3, v, w3);
                    CONV_ROW8_FMA2(s4, v, w4); CONV_ROW8_FMA2(s5, v, w5);
                    CONV_ROW8_FMA2(s6, v, w6); CONV_ROW8_FMA2(s7, v, w7);
                }
            }

            for (CONVINT r = kernelHeight; r < kernelHeight + 7; ++r) {
                const CONVINT qfirst = r - kernelHeight + 1;
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t v[2] = {
                        vld1q_f32(inputRow + ik), vld1q_f32(inputRow + ik + 4)};
                    for (CONVINT q = qfirst; q < 8; ++q)
                        CONV_ROW8_FMA2((q == 0 ? s0 : q == 1 ? s1 : q == 2 ? s2 :
                                        q == 3 ? s3 : q == 4 ? s4 : q == 5 ? s5 :
                                        q == 6 ? s6 : s7),
                                       v, CONV_WEIGHT(kernel + (r - q) * kernelWidth + ik));
                }
            }

            CONVFLOAT* const out0 = output + j * outputWidth + i;
            for (CONVINT q = 0; q < 8; ++q) {
                CONVFLOAT* const out = out0 + q * outputWidth;
                const float32x4_t* const s =
                    (q == 0 ? s0 : q == 1 ? s1 : q == 2 ? s2 : q == 3 ? s3 :
                     q == 4 ? s4 : q == 5 ? s5 : q == 6 ? s6 : s7);
                vst1q_f32(out, s[0]);
                vst1q_f32(out + 4, s[1]);
            }
        }

        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < 8; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#undef CONV_ROW8_FMA2
#endif

#if defined(__aarch64__) && !defined(CONV_DISABLE_ROW4_PAIR)
#if defined(CONV_PREFETCH_DISTANCE)
#define CONV_ROW4_PREFETCH(J, R, I) \
    __builtin_prefetch(input + ((J) + (R) + CONV_PREFETCH_DISTANCE) * inputWidth + (I), 0, 1)
#else
#define CONV_ROW4_PREFETCH(J, R, I) do { } while (0)
#endif
#define CONV_ROW4_FMA4(S, V, W) do { \
    (S)[0] = vfmaq_f32((S)[0], (V)[0], (W)); \
    (S)[1] = vfmaq_f32((S)[1], (V)[1], (W)); \
    (S)[2] = vfmaq_f32((S)[2], (V)[2], (W)); \
    (S)[3] = vfmaq_f32((S)[3], (V)[3], (W)); \
} while (0)

#define CONV_ROW4_LOAD(P, OFF) \
    const float32x4_t P[4] = { \
        vld1q_f32(inputRow + (OFF)), vld1q_f32(inputRow + (OFF) + 4), \
        vld1q_f32(inputRow + (OFF) + 8), vld1q_f32(inputRow + (OFF) + 12) }

static void conv2d_row4_pair(const CONVFLOAT* restrict input, CONVINT inputWidth,
                             const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                             CONVINT kernelWidth, CONVFLOAT* restrict output,
                             CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 4) {
        const CONVINT rows = (outputHeight - j < 4) ? outputHeight - j : 4;
        CONVINT i = 0;
        if (rows < 4) {
            for (; i < outputWidth; ++i) {
                for (CONVINT q = 0; q < rows; ++q) {
                    CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                    *out = 0.0f;
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                            *out += input[(j + q + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
                }
            }
            continue;
        }

        for (; i + 15 < outputWidth; i += 16) {
            float32x4_t s0[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s1[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s2[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s3[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };

            /* First three input rows have progressively more active outputs. */
            {
                const CONVFLOAT* const inputRow = input + j * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s0, v, CONV_WEIGHT(kernel + ik));
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + 1) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s0, v, CONV_WEIGHT(krow + ik));
                    CONV_ROW4_FMA4(s1, v, CONV_WEIGHT(kernel + ik));
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + 2) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + 2 * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s0, v, CONV_WEIGHT(krow + ik));
                    CONV_ROW4_FMA4(s1, v, CONV_WEIGHT(krow - kernelWidth + ik));
                    CONV_ROW4_FMA4(s2, v, CONV_WEIGHT(krow - 2 * kernelWidth + ik));
                }
            }

            /* Interior rows update all four outputs without active-row tests. */
            for (CONVINT r = 3; r < kernelHeight; ++r) {
                CONV_ROW4_PREFETCH(j, r, i);
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + r * kernelWidth;
                const CONVFLOAT* const krow1 = krow - kernelWidth;
                const CONVFLOAT* const krow2 = krow - 2 * kernelWidth;
                const CONVFLOAT* const krow3 = krow - 3 * kernelWidth;
#if defined(CONV_ROW4_UNROLL2)
#pragma GCC unroll 2
#endif
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    const float32x4_t w0 = CONV_WEIGHT(krow + ik);
                    const float32x4_t w1 = CONV_WEIGHT(krow1 + ik);
                    const float32x4_t w2 = CONV_WEIGHT(krow2 + ik);
                    const float32x4_t w3 = CONV_WEIGHT(krow3 + ik);
                    CONV_ROW4_FMA4(s0, v, w0);
                    CONV_ROW4_FMA4(s1, v, w1);
                    CONV_ROW4_FMA4(s2, v, w2);
                    CONV_ROW4_FMA4(s3, v, w3);
                }
            }

            /* Last three input rows belong progressively to fewer outputs. */
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + (kernelHeight - 1) * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s1, v, CONV_WEIGHT(krow + ik));
                    CONV_ROW4_FMA4(s2, v, CONV_WEIGHT(krow - kernelWidth + ik));
                    CONV_ROW4_FMA4(s3, v, CONV_WEIGHT(krow - 2 * kernelWidth + ik));
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight + 1) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + (kernelHeight - 1) * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s2, v, CONV_WEIGHT(krow + ik));
                    CONV_ROW4_FMA4(s3, v, CONV_WEIGHT(krow - kernelWidth + ik));
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight + 2) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + (kernelHeight - 1) * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LOAD(v, ik);
                    CONV_ROW4_FMA4(s3, v, CONV_WEIGHT(krow + ik));
                }
            }

            CONVFLOAT* const out0 = output + j * outputWidth + i;
            CONVFLOAT* const out1 = out0 + outputWidth;
            CONVFLOAT* const out2 = out1 + outputWidth;
            CONVFLOAT* const out3 = out2 + outputWidth;
            for (CONVINT q = 0; q < 4; ++q) {
                vst1q_f32(out0 + 4 * q, s0[q]);
                vst1q_f32(out1 + 4 * q, s1[q]);
                vst1q_f32(out2 + 4 * q, s2[q]);
                vst1q_f32(out3 + 4 * q, s3[q]);
            }
        }

        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < 4; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#undef CONV_ROW4_LOAD
#undef CONV_ROW4_FMA4
#undef CONV_ROW4_PREFETCH
#endif

/* Optional 4-row x 20-column probe.  It is kept behind a macro because the
 * extra four accumulators may increase register pressure on some Kunpeng
 * steppings; the 16-column kernel remains the default. */
#if defined(__aarch64__) && defined(CONV_ENABLE_ROW4_W20)
#define CONV_ROW4_W20_FMA(S, V, W) do { \
    (S)[0] = vfmaq_f32((S)[0], (V)[0], (W)); \
    (S)[1] = vfmaq_f32((S)[1], (V)[1], (W)); \
    (S)[2] = vfmaq_f32((S)[2], (V)[2], (W)); \
    (S)[3] = vfmaq_f32((S)[3], (V)[3], (W)); \
    (S)[4] = vfmaq_f32((S)[4], (V)[4], (W)); \
} while (0)

#define CONV_ROW4_W20_LOAD(P, OFF) \
    const float32x4_t P[5] = { \
        vld1q_f32(inputRow + (OFF)), vld1q_f32(inputRow + (OFF) + 4), \
        vld1q_f32(inputRow + (OFF) + 8), vld1q_f32(inputRow + (OFF) + 12), \
        vld1q_f32(inputRow + (OFF) + 16) }

static void conv2d_row4_w20(const CONVFLOAT* restrict input, CONVINT inputWidth,
                            const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                            CONVINT kernelWidth, CONVFLOAT* restrict output,
                            CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 4) {
        const CONVINT rows = (outputHeight - j < 4) ? outputHeight - j : 4;
        CONVINT i = 0;
        if (rows < 4) {
            for (; i < outputWidth; ++i) {
                for (CONVINT q = 0; q < rows; ++q) {
                    CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                    *out = 0.0f;
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                            *out += input[(j + q + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
                }
            }
            continue;
        }

        for (; i + 19 < outputWidth; i += 20) {
            float32x4_t s0[5] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f) };
            float32x4_t s1[5] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f) };
            float32x4_t s2[5] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f) };
            float32x4_t s3[5] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f) };

            for (CONVINT r = 0; r < 3; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_W20_LOAD(v, ik);
                    CONV_ROW4_W20_FMA(s0, v, CONV_WEIGHT(kernel + r * kernelWidth + ik));
                    if (r >= 1)
                        CONV_ROW4_W20_FMA(s1, v, CONV_WEIGHT(kernel + (r - 1) * kernelWidth + ik));
                    if (r >= 2)
                        CONV_ROW4_W20_FMA(s2, v, CONV_WEIGHT(kernel + (r - 2) * kernelWidth + ik));
                }
            }

            for (CONVINT r = 3; r < kernelHeight; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const krow = kernel + r * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_W20_LOAD(v, ik);
                    CONV_ROW4_W20_FMA(s0, v, CONV_WEIGHT(krow + ik));
                    CONV_ROW4_W20_FMA(s1, v, CONV_WEIGHT(krow - kernelWidth + ik));
                    CONV_ROW4_W20_FMA(s2, v, CONV_WEIGHT(krow - 2 * kernelWidth + ik));
                    CONV_ROW4_W20_FMA(s3, v, CONV_WEIGHT(krow - 3 * kernelWidth + ik));
                }
            }

            for (CONVINT r = kernelHeight; r < kernelHeight + 3; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVINT qfirst = r - kernelHeight + 1;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_W20_LOAD(v, ik);
                    if (qfirst <= 1)
                        CONV_ROW4_W20_FMA(s1, v, CONV_WEIGHT(kernel + (r - 1) * kernelWidth + ik));
                    if (qfirst <= 2)
                        CONV_ROW4_W20_FMA(s2, v, CONV_WEIGHT(kernel + (r - 2) * kernelWidth + ik));
                    CONV_ROW4_W20_FMA(s3, v, CONV_WEIGHT(kernel + (r - 3) * kernelWidth + ik));
                }
            }

            CONVFLOAT* const out0 = output + j * outputWidth + i;
            CONVFLOAT* const out1 = out0 + outputWidth;
            CONVFLOAT* const out2 = out1 + outputWidth;
            CONVFLOAT* const out3 = out2 + outputWidth;
            for (CONVINT x = 0; x < 5; ++x) {
                vst1q_f32(out0 + 4 * x, s0[x]);
                vst1q_f32(out1 + 4 * x, s1[x]);
                vst1q_f32(out2 + 4 * x, s2[x]);
                vst1q_f32(out3 + 4 * x, s3[x]);
            }
        }

        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < 4; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#undef CONV_ROW4_W20_LOAD
#undef CONV_ROW4_W20_FMA
#endif

/* Optional lane-weight probe.  One vector load supplies four adjacent kernel
 * coefficients, then each lane is consumed in the original ik order. */
#if defined(__aarch64__) && defined(CONV_ENABLE_ROW4_LANE4)
#define CONV_ROW4_LANE_FMA(S, V, W, L) do { \
    (S)[0] = vfmaq_laneq_f32((S)[0], (V)[0], (W), (L)); \
    (S)[1] = vfmaq_laneq_f32((S)[1], (V)[1], (W), (L)); \
    (S)[2] = vfmaq_laneq_f32((S)[2], (V)[2], (W), (L)); \
    (S)[3] = vfmaq_laneq_f32((S)[3], (V)[3], (W), (L)); \
} while (0)

#define CONV_ROW4_LANE_LOAD(P, OFF) \
    const float32x4_t P[4] = { \
        vld1q_f32(inputRow + (OFF)), vld1q_f32(inputRow + (OFF) + 4), \
        vld1q_f32(inputRow + (OFF) + 8), vld1q_f32(inputRow + (OFF) + 12) }

static void conv2d_row4_lane4(const CONVFLOAT* restrict input, CONVINT inputWidth,
                              const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                              CONVINT kernelWidth, CONVFLOAT* restrict output,
                              CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 4) {
        const CONVINT rows = (outputHeight - j < 4) ? outputHeight - j : 4;
        CONVINT i = 0;
        if (rows < 4) {
            for (; i < outputWidth; ++i) {
                for (CONVINT q = 0; q < rows; ++q) {
                    CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                    *out = 0.0f;
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                            *out += input[(j + q + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
                }
            }
            continue;
        }

        for (; i + 15 < outputWidth; i += 16) {
            float32x4_t s0[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s1[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s2[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            float32x4_t s3[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                  vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };

            for (CONVINT r = 0; r < 3; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const k0 = kernel + r * kernelWidth;
                const CONVFLOAT* const k1 = (r >= 1) ? k0 - kernelWidth : NULL;
                const CONVFLOAT* const k2 = (r >= 2) ? k0 - 2 * kernelWidth : NULL;
                CONVINT ik = 0;
                for (; ik + 3 < kernelWidth; ik += 4) {
                    const float32x4_t w0 = vld1q_f32(k0 + ik);
                    const float32x4_t w1 = (r >= 1) ? vld1q_f32(k1 + ik) : w0;
                    const float32x4_t w2 = (r >= 2) ? vld1q_f32(k2 + ik) : w0;
                    { CONV_ROW4_LANE_LOAD(v, ik);     CONV_ROW4_LANE_FMA(s0, v, w0, 0);
                    if (r >= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 0);
                    if (r >= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 0); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 1); CONV_ROW4_LANE_FMA(s0, v, w0, 1);
                    if (r >= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 1);
                    if (r >= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 1); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 2); CONV_ROW4_LANE_FMA(s0, v, w0, 2);
                    if (r >= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 2);
                    if (r >= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 2); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 3); CONV_ROW4_LANE_FMA(s0, v, w0, 3);
                    if (r >= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 3);
                    if (r >= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 3); }
                }
                for (; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LANE_LOAD(v, ik);
                    CONV_ROW4_LANE_FMA(s0, v, vdupq_n_f32(*(k0 + ik)), 0);
                    if (r >= 1) CONV_ROW4_LANE_FMA(s1, v, vdupq_n_f32(*(k1 + ik)), 0);
                    if (r >= 2) CONV_ROW4_LANE_FMA(s2, v, vdupq_n_f32(*(k2 + ik)), 0);
                }
            }

            for (CONVINT r = 3; r < kernelHeight; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const k0 = kernel + r * kernelWidth;
                const CONVFLOAT* const k1 = k0 - kernelWidth;
                const CONVFLOAT* const k2 = k0 - 2 * kernelWidth;
                const CONVFLOAT* const k3 = k0 - 3 * kernelWidth;
                CONVINT ik = 0;
                for (; ik + 3 < kernelWidth; ik += 4) {
                    const float32x4_t w0 = vld1q_f32(k0 + ik);
                    const float32x4_t w1 = vld1q_f32(k1 + ik);
                    const float32x4_t w2 = vld1q_f32(k2 + ik);
                    const float32x4_t w3 = vld1q_f32(k3 + ik);
                    { CONV_ROW4_LANE_LOAD(v, ik);     CONV_ROW4_LANE_FMA(s0, v, w0, 0); CONV_ROW4_LANE_FMA(s1, v, w1, 0); CONV_ROW4_LANE_FMA(s2, v, w2, 0); CONV_ROW4_LANE_FMA(s3, v, w3, 0); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 1); CONV_ROW4_LANE_FMA(s0, v, w0, 1); CONV_ROW4_LANE_FMA(s1, v, w1, 1); CONV_ROW4_LANE_FMA(s2, v, w2, 1); CONV_ROW4_LANE_FMA(s3, v, w3, 1); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 2); CONV_ROW4_LANE_FMA(s0, v, w0, 2); CONV_ROW4_LANE_FMA(s1, v, w1, 2); CONV_ROW4_LANE_FMA(s2, v, w2, 2); CONV_ROW4_LANE_FMA(s3, v, w3, 2); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 3); CONV_ROW4_LANE_FMA(s0, v, w0, 3); CONV_ROW4_LANE_FMA(s1, v, w1, 3); CONV_ROW4_LANE_FMA(s2, v, w2, 3); CONV_ROW4_LANE_FMA(s3, v, w3, 3); }
                }
                for (; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LANE_LOAD(v, ik);
                    CONV_ROW4_LANE_FMA(s0, v, vdupq_n_f32(*(k0 + ik)), 0);
                    CONV_ROW4_LANE_FMA(s1, v, vdupq_n_f32(*(k1 + ik)), 0);
                    CONV_ROW4_LANE_FMA(s2, v, vdupq_n_f32(*(k2 + ik)), 0);
                    CONV_ROW4_LANE_FMA(s3, v, vdupq_n_f32(*(k3 + ik)), 0);
                }
            }

            for (CONVINT r = kernelHeight; r < kernelHeight + 3; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVINT qfirst = r - kernelHeight + 1;
                const CONVFLOAT* const k1 = kernel + (r - 1) * kernelWidth;
                const CONVFLOAT* const k2 = kernel + (r - 2) * kernelWidth;
                const CONVFLOAT* const k3 = kernel + (r - 3) * kernelWidth;
                CONVINT ik = 0;
                for (; ik + 3 < kernelWidth; ik += 4) {
                    const float32x4_t w1 = vld1q_f32(k1 + ik);
                    const float32x4_t w2 = vld1q_f32(k2 + ik);
                    const float32x4_t w3 = vld1q_f32(k3 + ik);
                    { CONV_ROW4_LANE_LOAD(v, ik);     if (qfirst <= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 0); if (qfirst <= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 0); CONV_ROW4_LANE_FMA(s3, v, w3, 0); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 1); if (qfirst <= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 1); if (qfirst <= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 1); CONV_ROW4_LANE_FMA(s3, v, w3, 1); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 2); if (qfirst <= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 2); if (qfirst <= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 2); CONV_ROW4_LANE_FMA(s3, v, w3, 2); }
                    { CONV_ROW4_LANE_LOAD(v, ik + 3); if (qfirst <= 1) CONV_ROW4_LANE_FMA(s1, v, w1, 3); if (qfirst <= 2) CONV_ROW4_LANE_FMA(s2, v, w2, 3); CONV_ROW4_LANE_FMA(s3, v, w3, 3); }
                }
                for (; ik < kernelWidth; ++ik) {
                    CONV_ROW4_LANE_LOAD(v, ik);
                    if (qfirst <= 1) CONV_ROW4_LANE_FMA(s1, v, vdupq_n_f32(*(k1 + ik)), 0);
                    if (qfirst <= 2) CONV_ROW4_LANE_FMA(s2, v, vdupq_n_f32(*(k2 + ik)), 0);
                    CONV_ROW4_LANE_FMA(s3, v, vdupq_n_f32(*(k3 + ik)), 0);
                }
            }

            CONVFLOAT* const out0 = output + j * outputWidth + i;
            CONVFLOAT* const out1 = out0 + outputWidth;
            CONVFLOAT* const out2 = out1 + outputWidth;
            CONVFLOAT* const out3 = out2 + outputWidth;
            for (CONVINT x = 0; x < 4; ++x) {
                vst1q_f32(out0 + 4 * x, s0[x]);
                vst1q_f32(out1 + 4 * x, s1[x]);
                vst1q_f32(out2 + 4 * x, s2[x]);
                vst1q_f32(out3 + 4 * x, s3[x]);
            }
        }
        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < 4; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#undef CONV_ROW4_LANE_LOAD
#undef CONV_ROW4_LANE_FMA
#endif

#if defined(__aarch64__) && !defined(CONV_DISABLE_ROW4_VPACK)
#define CONV_ROW4_VP_LOAD(P, OFF) \
    const float32x4_t P[4] = { \
        vld1q_f32(inputRow + (OFF)), vld1q_f32(inputRow + (OFF) + 4), \
        vld1q_f32(inputRow + (OFF) + 8), vld1q_f32(inputRow + (OFF) + 12) }
#define CONV_ROW4_VP_FMA(S, V, W, L) do { \
    (S)[0] = vfmaq_laneq_f32((S)[0], (V)[0], (W), (L)); \
    (S)[1] = vfmaq_laneq_f32((S)[1], (V)[1], (W), (L)); \
    (S)[2] = vfmaq_laneq_f32((S)[2], (V)[2], (W), (L)); \
    (S)[3] = vfmaq_laneq_f32((S)[3], (V)[3], (W), (L)); \
} while (0)

static void conv2d_row4_vpack(const CONVFLOAT* restrict input, CONVINT inputWidth,
                              const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                              CONVINT kernelWidth,
                              CONVFLOAT* restrict output, CONVINT outputHeight,
                              CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 4) {
        const CONVINT rows = (outputHeight - j < 4) ? outputHeight - j : 4;
        CONVINT i = 0;
        if (rows < 4) {
            for (; i < outputWidth; ++i) {
                for (CONVINT q = 0; q < rows; ++q) {
                    CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                    *out = 0.0f;
                    for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                        for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                            *out += input[(j + q + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
                }
            }
            continue;
        }
        for (; i + 15 < outputWidth; i += 16) {
            float32x4_t s0[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t s1[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t s2[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t s3[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
            {
                const CONVFLOAT* const inputRow = input + j * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[ik];
                    CONV_ROW4_VP_FMA(s0, v, w, 0);
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + 1) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s0, v, w, 0);
                    CONV_ROW4_VP_FMA(s1, v, w, 1);
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + 2) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[2 * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s0, v, w, 0);
                    CONV_ROW4_VP_FMA(s1, v, w, 1);
                    CONV_ROW4_VP_FMA(s2, v, w, 2);
                }
            }
            for (CONVINT r = 3; r < kernelHeight; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
#if defined(CONV_ROW4_SLIDE4)
                CONVINT ik = 0;
                for (; ik + 3 < kernelWidth; ik += 4) {
                    const float32x4_t a0 = vld1q_f32(inputRow + ik);
                    const float32x4_t a1 = vld1q_f32(inputRow + ik + 4);
                    const float32x4_t a2 = vld1q_f32(inputRow + ik + 8);
                    const float32x4_t a3 = vld1q_f32(inputRow + ik + 12);
                    const float32x4_t a4 = vld1q_f32(inputRow + ik + 16);
                    {
                        const float32x4_t v[4] = { a0, a1, a2, a3 };
                        const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik];
                        CONV_ROW4_VP_FMA(s0, v, w, 0); CONV_ROW4_VP_FMA(s1, v, w, 1);
                        CONV_ROW4_VP_FMA(s2, v, w, 2); CONV_ROW4_VP_FMA(s3, v, w, 3);
                    }
                    {
                        const float32x4_t v[4] = {
                            vextq_f32(a0, a1, 1), vextq_f32(a1, a2, 1),
                            vextq_f32(a2, a3, 1), vextq_f32(a3, a4, 1) };
                        const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik + 1];
                        CONV_ROW4_VP_FMA(s0, v, w, 0); CONV_ROW4_VP_FMA(s1, v, w, 1);
                        CONV_ROW4_VP_FMA(s2, v, w, 2); CONV_ROW4_VP_FMA(s3, v, w, 3);
                    }
                    {
                        const float32x4_t v[4] = {
                            vextq_f32(a0, a1, 2), vextq_f32(a1, a2, 2),
                            vextq_f32(a2, a3, 2), vextq_f32(a3, a4, 2) };
                        const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik + 2];
                        CONV_ROW4_VP_FMA(s0, v, w, 0); CONV_ROW4_VP_FMA(s1, v, w, 1);
                        CONV_ROW4_VP_FMA(s2, v, w, 2); CONV_ROW4_VP_FMA(s3, v, w, 3);
                    }
                    {
                        const float32x4_t v[4] = {
                            vextq_f32(a0, a1, 3), vextq_f32(a1, a2, 3),
                            vextq_f32(a2, a3, 3), vextq_f32(a3, a4, 3) };
                        const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik + 3];
                        CONV_ROW4_VP_FMA(s0, v, w, 0); CONV_ROW4_VP_FMA(s1, v, w, 1);
                        CONV_ROW4_VP_FMA(s2, v, w, 2); CONV_ROW4_VP_FMA(s3, v, w, 3);
                    }
                }
                for (; ik < kernelWidth; ++ik) {
                    const CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s0, v, w, 0); CONV_ROW4_VP_FMA(s1, v, w, 1);
                    CONV_ROW4_VP_FMA(s2, v, w, 2); CONV_ROW4_VP_FMA(s3, v, w, 3);
                }
#else
#if defined(CONV_ROW4_VP_UNROLL2)
#pragma GCC unroll 2
#endif
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[r * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s0, v, w, 0);
                    CONV_ROW4_VP_FMA(s1, v, w, 1);
                    CONV_ROW4_VP_FMA(s2, v, w, 2);
                    CONV_ROW4_VP_FMA(s3, v, w, 3);
                }
#endif
            }
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[kernelHeight * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s1, v, w, 1);
                    CONV_ROW4_VP_FMA(s2, v, w, 2);
                    CONV_ROW4_VP_FMA(s3, v, w, 3);
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight + 1) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[(kernelHeight + 1) * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s2, v, w, 2);
                    CONV_ROW4_VP_FMA(s3, v, w, 3);
                }
            }
            {
                const CONVFLOAT* const inputRow = input + (j + kernelHeight + 2) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONV_ROW4_VP_LOAD(v, ik);
                    const float32x4_t w = conv_row4_vpack_base[(kernelHeight + 2) * kernelWidth + ik];
                    CONV_ROW4_VP_FMA(s3, v, w, 3);
                }
            }
            CONVFLOAT* const out0 = output + j * outputWidth + i;
            CONVFLOAT* const out1 = out0 + outputWidth;
            CONVFLOAT* const out2 = out1 + outputWidth;
            CONVFLOAT* const out3 = out2 + outputWidth;
            for (CONVINT x = 0; x < 4; ++x) {
                vst1q_f32(out0 + 4 * x, s0[x]); vst1q_f32(out1 + 4 * x, s1[x]);
                vst1q_f32(out2 + 4 * x, s2[x]); vst1q_f32(out3 + 4 * x, s3[x]);
            }
        }
        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < 4; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#undef CONV_ROW4_VP_LOAD
#undef CONV_ROW4_VP_FMA
#endif

#if defined(__aarch64__) && defined(CONV_ENABLE_ROW2_FAST)
#define CONV_ROW2_FMA8(S, V, W) do { \
    (S)[0] = vfmaq_f32((S)[0], (V)[0], (W)); \
    (S)[1] = vfmaq_f32((S)[1], (V)[1], (W)); \
    (S)[2] = vfmaq_f32((S)[2], (V)[2], (W)); \
    (S)[3] = vfmaq_f32((S)[3], (V)[3], (W)); \
    (S)[4] = vfmaq_f32((S)[4], (V)[4], (W)); \
    (S)[5] = vfmaq_f32((S)[5], (V)[5], (W)); \
    (S)[6] = vfmaq_f32((S)[6], (V)[6], (W)); \
    (S)[7] = vfmaq_f32((S)[7], (V)[7], (W)); \
} while (0)

static void conv2d_row2_fast(const CONVFLOAT* restrict input, CONVINT inputWidth,
                             const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                             CONVINT kernelWidth, CONVFLOAT* restrict output,
                             CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 2) {
        const int full = (j + 1 < outputHeight);
        CONVINT i = 0;
        if (!full) {
            for (; i < outputWidth; ++i) {
                CONVFLOAT* const out = output + j * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
            continue;
        }

        CONVFLOAT* const out0 = output + j * outputWidth;
        CONVFLOAT* const out1 = out0 + outputWidth;
        for (; i + 31 < outputWidth; i += 32) {
            float32x4_t s0[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            float32x4_t s1[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

            for (CONVINT r = 0; r < kernelHeight; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + r * kernelWidth;
#define CONV_ROW2_LOAD8(NAME, OFF) \
                const float32x4_t NAME[8] = { \
                    vld1q_f32(inputRow + (OFF)), vld1q_f32(inputRow + (OFF) + 4), \
                    vld1q_f32(inputRow + (OFF) + 8), vld1q_f32(inputRow + (OFF) + 12), \
                    vld1q_f32(inputRow + (OFF) + 16), vld1q_f32(inputRow + (OFF) + 20), \
                    vld1q_f32(inputRow + (OFF) + 24), vld1q_f32(inputRow + (OFF) + 28) }
                for (CONVINT ik = 0; ik + 1 < kernelWidth; ik += 2) {
                    CONV_ROW2_LOAD8(v, ik);
                    const float32x4_t w0 = CONV_WEIGHT(kernelRow + ik);
                    CONV_ROW2_FMA8(s0, v, w0);
                    if (r > 0) {
                        const float32x4_t w1 = CONV_WEIGHT(kernelRow - kernelWidth + ik);
                        CONV_ROW2_FMA8(s1, v, w1);
                    }
                    CONV_ROW2_LOAD8(vn, ik + 1);
                    const float32x4_t w0n = CONV_WEIGHT(kernelRow + ik + 1);
                    CONV_ROW2_FMA8(s0, vn, w0n);
                    if (r > 0) {
                        const float32x4_t w1n = CONV_WEIGHT(kernelRow - kernelWidth + ik + 1);
                        CONV_ROW2_FMA8(s1, vn, w1n);
                    }
                }
                if (kernelWidth & 1) {
                    const CONVINT ik = kernelWidth - 1;
                    CONV_ROW2_LOAD8(v, ik);
                    const float32x4_t w0 = CONV_WEIGHT(kernelRow + ik);
                    CONV_ROW2_FMA8(s0, v, w0);
                    if (r > 0) {
                        const float32x4_t w1 = CONV_WEIGHT(kernelRow - kernelWidth + ik);
                        CONV_ROW2_FMA8(s1, v, w1);
                    }
                }
#undef CONV_ROW2_LOAD8
            }

            /* The final input row belongs only to output row j+1. */
            {
                const CONVINT r = kernelHeight;
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + (r - 1) * kernelWidth;
                for (CONVINT ik = 0; ik + 1 < kernelWidth; ik += 2) {
                    const float32x4_t v[8] = {
                        vld1q_f32(inputRow + ik), vld1q_f32(inputRow + ik + 4),
                        vld1q_f32(inputRow + ik + 8), vld1q_f32(inputRow + ik + 12),
                        vld1q_f32(inputRow + ik + 16), vld1q_f32(inputRow + ik + 20),
                        vld1q_f32(inputRow + ik + 24), vld1q_f32(inputRow + ik + 28)};
                    const float32x4_t w = CONV_WEIGHT(kernelRow + ik);
                    CONV_ROW2_FMA8(s1, v, w);
                    const float32x4_t vn[8] = {
                        vld1q_f32(inputRow + ik + 1), vld1q_f32(inputRow + ik + 5),
                        vld1q_f32(inputRow + ik + 9), vld1q_f32(inputRow + ik + 13),
                        vld1q_f32(inputRow + ik + 17), vld1q_f32(inputRow + ik + 21),
                        vld1q_f32(inputRow + ik + 25), vld1q_f32(inputRow + ik + 29)};
                    const float32x4_t wn = CONV_WEIGHT(kernelRow + ik + 1);
                    CONV_ROW2_FMA8(s1, vn, wn);
                }
                if (kernelWidth & 1) {
                    const CONVINT ik = kernelWidth - 1;
                    const float32x4_t v[8] = {
                        vld1q_f32(inputRow + ik), vld1q_f32(inputRow + ik + 4),
                        vld1q_f32(inputRow + ik + 8), vld1q_f32(inputRow + ik + 12),
                        vld1q_f32(inputRow + ik + 16), vld1q_f32(inputRow + ik + 20),
                        vld1q_f32(inputRow + ik + 24), vld1q_f32(inputRow + ik + 28)};
                    const float32x4_t w = CONV_WEIGHT(kernelRow + ik);
                    CONV_ROW2_FMA8(s1, v, w);
                }
            }

            for (CONVINT x = 0; x < 8; ++x) {
                vst1q_f32(out0 + i + 4 * x, s0[x]);
                vst1q_f32(out1 + i + 4 * x, s1[x]);
            }
        }
        for (; i < outputWidth; ++i) {
            out0[i] = 0.0f;
            out1[i] = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT w = kernel[jk * kernelWidth + ik];
                    out0[i] += input[(j + jk) * inputWidth + i + ik] * w;
                    out1[i] += input[(j + 1 + jk) * inputWidth + i + ik] * w;
                }
        }
    }
}
#undef CONV_ROW2_FMA8
#endif

#if defined(__aarch64__) && defined(CONV_ENABLE_ROW3)
static void conv2d_row3(const CONVFLOAT* restrict input, CONVINT inputWidth,
                        const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                        CONVINT kernelWidth, CONVFLOAT* restrict output,
                        CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 3) {
        const CONVINT rows = (outputHeight - j < 3) ? outputHeight - j : 3;
        CONVINT i = 0;
        for (; i + 15 < outputWidth; i += 16) {
            float32x4_t sum[3][4];
            for (CONVINT q = 0; q < 3; ++q)
                for (CONVINT x = 0; x < 4; ++x)
                    sum[q][x] = vdupq_n_f32(0.0f);
            for (CONVINT r = 0; r < kernelHeight + rows - 1; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t v0 = vld1q_f32(inputRow + ik);
                    const float32x4_t v1 = vld1q_f32(inputRow + ik + 4);
                    const float32x4_t v2 = vld1q_f32(inputRow + ik + 8);
                    const float32x4_t v3 = vld1q_f32(inputRow + ik + 12);
                    for (CONVINT q = 0; q < 3; ++q) {
                        if (q < rows && r >= q && r < q + kernelHeight) {
                            const float32x4_t w =
                                CONV_WEIGHT(kernel + (r - q) * kernelWidth + ik);
                            sum[q][0] = vfmaq_f32(sum[q][0], v0, w);
                            sum[q][1] = vfmaq_f32(sum[q][1], v1, w);
                            sum[q][2] = vfmaq_f32(sum[q][2], v2, w);
                            sum[q][3] = vfmaq_f32(sum[q][3], v3, w);
                        }
                    }
                }
            }
            for (CONVINT q = 0; q < rows; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                vst1q_f32(out, sum[q][0]);
                vst1q_f32(out + 4, sum[q][1]);
                vst1q_f32(out + 8, sum[q][2]);
                vst1q_f32(out + 12, sum[q][3]);
            }
        }
        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < rows; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#endif

#if defined(__aarch64__) && defined(CONV_ENABLE_ROW4)
/* Four output rows by sixteen columns: keep the accumulator count at sixteen
 * vectors while reusing each loaded input vector across four row outputs. */
static void conv2d_row4(const CONVFLOAT* restrict input, CONVINT inputWidth,
                        const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                        CONVINT kernelWidth, CONVFLOAT* restrict output,
                        CONVINT outputHeight, CONVINT outputWidth)
{
#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; j += 4) {
        const CONVINT rows = (outputHeight - j < 4) ? outputHeight - j : 4;
        CONVINT i = 0;
        for (; i + 15 < outputWidth; i += 16) {
            float32x4_t sum[4][4];
            for (CONVINT q = 0; q < 4; ++q)
                for (CONVINT x = 0; x < 4; ++x)
                    sum[q][x] = vdupq_n_f32(0.0f);

            for (CONVINT r = 0; r < kernelHeight + rows - 1; ++r) {
                const CONVFLOAT* const inputRow = input + (j + r) * inputWidth + i;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t v0 = vld1q_f32(inputRow + ik);
                    const float32x4_t v1 = vld1q_f32(inputRow + ik + 4);
                    const float32x4_t v2 = vld1q_f32(inputRow + ik + 8);
                    const float32x4_t v3 = vld1q_f32(inputRow + ik + 12);
                    for (CONVINT q = 0; q < 4; ++q) {
                        if (q < rows && r >= q && r < q + kernelHeight) {
                            const float32x4_t w =
                                CONV_WEIGHT(kernel + (r - q) * kernelWidth + ik);
                            sum[q][0] = vfmaq_f32(sum[q][0], v0, w);
                            sum[q][1] = vfmaq_f32(sum[q][1], v1, w);
                            sum[q][2] = vfmaq_f32(sum[q][2], v2, w);
                            sum[q][3] = vfmaq_f32(sum[q][3], v3, w);
                        }
                    }
                }
            }

            for (CONVINT q = 0; q < rows; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                vst1q_f32(out, sum[q][0]);
                vst1q_f32(out + 4, sum[q][1]);
                vst1q_f32(out + 8, sum[q][2]);
                vst1q_f32(out + 12, sum[q][3]);
            }
        }

        for (; i < outputWidth; ++i) {
            for (CONVINT q = 0; q < rows; ++q) {
                CONVFLOAT* const out = output + (j + q) * outputWidth + i;
                *out = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        *out += input[(j + q + jk) * inputWidth + i + ik] *
                                kernel[jk * kernelWidth + ik];
            }
        }
    }
}
#endif

#if defined(__aarch64__) && !defined(CONV_DISABLE_ROW2)
/*
 * Fuse two adjacent output rows.  For rows j and j+1, input row j+r is
 * consumed by both outputs (except at the two boundaries), so one vector
 * load can feed two FMAs with different kernel rows.  The row order for each
 * output remains identical to the reference implementation.
 */
static void conv2d_row2(const CONVFLOAT* restrict input, CONVINT inputWidth,
                        const CONVFLOAT* restrict kernel, CONVINT kernelHeight,
                        CONVINT kernelWidth, CONVFLOAT* restrict output,
                        CONVINT outputHeight, CONVINT outputWidth)
{
#if defined(CONV_ROW2_PACKED_WEIGHT)
    float32x4_t *packedKernel = (float32x4_t *)malloc(
        (size_t)kernelHeight * (size_t)kernelWidth * sizeof(*packedKernel));
    if (packedKernel == NULL)
        return;
    for (CONVINT idx = 0; idx < kernelHeight * kernelWidth; ++idx)
        packedKernel[idx] = vdupq_n_f32(kernel[idx]);
    conv_kernel_base = kernel;
    conv_packed_kernel_base = packedKernel;
#endif
#pragma omp parallel for schedule(static)
    for (CONVINT pair = 0; pair < outputHeight; pair += 2) {
        const CONVINT j = pair;
        const CONVINT has_second = (j + 1 < outputHeight);
        CONVFLOAT* const out0 = output + j * outputWidth;
        CONVFLOAT* const out1 = out0 + outputWidth;
        CONVINT i = 0;

        for (; i + 31 < outputWidth; i += 32) {
            float32x4_t s0[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            float32x4_t s1[8] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

            for (CONVINT r = 0; r <= kernelHeight; ++r) {
                const int use0 = (r < kernelHeight);
                const int use1 = (has_second && r > 0);
                if (!use0 && !use1)
                    break;
                const CONVFLOAT* const inputRow =
                    input + (j + r) * inputWidth + i;
                const CONVFLOAT* const kernelRow0 =
                    use0 ? kernel + r * kernelWidth : kernel;
                const CONVFLOAT* const kernelRow1 =
                    use1 ? kernel + (r - 1) * kernelWidth : kernel;

#if defined(CONV_ROW2_UNROLL2)
#pragma GCC unroll 2
#endif
                for (CONVINT ik = 0; ik + 1 < kernelWidth; ik += 2) {
#if defined(CONV_ROW2_LOADX4)
                    const float32x4x4_t b0 = vld1q_f32_x4(inputRow + ik);
                    const float32x4x4_t b1 = vld1q_f32_x4(inputRow + ik + 16);
                    const float32x4_t v0 = b0.val[0], v1 = b0.val[1];
                    const float32x4_t v2 = b0.val[2], v3 = b0.val[3];
                    const float32x4_t v4 = b1.val[0], v5 = b1.val[1];
                    const float32x4_t v6 = b1.val[2], v7 = b1.val[3];
#else
                    const float32x4_t v0 = vld1q_f32(inputRow + ik);
                    const float32x4_t v1 = vld1q_f32(inputRow + ik + 4);
                    const float32x4_t v2 = vld1q_f32(inputRow + ik + 8);
                    const float32x4_t v3 = vld1q_f32(inputRow + ik + 12);
                    const float32x4_t v4 = vld1q_f32(inputRow + ik + 16);
                    const float32x4_t v5 = vld1q_f32(inputRow + ik + 20);
                    const float32x4_t v6 = vld1q_f32(inputRow + ik + 24);
                    const float32x4_t v7 = vld1q_f32(inputRow + ik + 28);
#endif
#if defined(CONV_ROW2_INTERLEAVE)
                    if (use0 && use1) {
                        const float32x4_t w0 = CONV_WEIGHT(kernelRow0 + ik);
                        const float32x4_t w1 = CONV_WEIGHT(kernelRow1 + ik);
                        s0[0] = vfmaq_f32(s0[0], v0, w0); s1[0] = vfmaq_f32(s1[0], v0, w1);
                        s0[1] = vfmaq_f32(s0[1], v1, w0); s1[1] = vfmaq_f32(s1[1], v1, w1);
                        s0[2] = vfmaq_f32(s0[2], v2, w0); s1[2] = vfmaq_f32(s1[2], v2, w1);
                        s0[3] = vfmaq_f32(s0[3], v3, w0); s1[3] = vfmaq_f32(s1[3], v3, w1);
                        s0[4] = vfmaq_f32(s0[4], v4, w0); s1[4] = vfmaq_f32(s1[4], v4, w1);
                        s0[5] = vfmaq_f32(s0[5], v5, w0); s1[5] = vfmaq_f32(s1[5], v5, w1);
                        s0[6] = vfmaq_f32(s0[6], v6, w0); s1[6] = vfmaq_f32(s1[6], v6, w1);
                        s0[7] = vfmaq_f32(s0[7], v7, w0); s1[7] = vfmaq_f32(s1[7], v7, w1);
                    } else
#endif
                    {
                        if (use0) {
                            const float32x4_t w0 = CONV_WEIGHT(kernelRow0 + ik);
                            s0[0] = vfmaq_f32(s0[0], v0, w0); s0[1] = vfmaq_f32(s0[1], v1, w0);
                            s0[2] = vfmaq_f32(s0[2], v2, w0); s0[3] = vfmaq_f32(s0[3], v3, w0);
                            s0[4] = vfmaq_f32(s0[4], v4, w0); s0[5] = vfmaq_f32(s0[5], v5, w0);
                            s0[6] = vfmaq_f32(s0[6], v6, w0); s0[7] = vfmaq_f32(s0[7], v7, w0);
                        }
                        if (use1) {
                            const float32x4_t w1 = CONV_WEIGHT(kernelRow1 + ik);
                            s1[0] = vfmaq_f32(s1[0], v0, w1); s1[1] = vfmaq_f32(s1[1], v1, w1);
                            s1[2] = vfmaq_f32(s1[2], v2, w1); s1[3] = vfmaq_f32(s1[3], v3, w1);
                            s1[4] = vfmaq_f32(s1[4], v4, w1); s1[5] = vfmaq_f32(s1[5], v5, w1);
                            s1[6] = vfmaq_f32(s1[6], v6, w1); s1[7] = vfmaq_f32(s1[7], v7, w1);
                        }
                    }
#if defined(CONV_ROW2_LOADX4)
                    const float32x4x4_t bn0 = vld1q_f32_x4(inputRow + ik + 1);
                    const float32x4x4_t bn1 = vld1q_f32_x4(inputRow + ik + 17);
                    const float32x4_t v0n = bn0.val[0], v1n = bn0.val[1];
                    const float32x4_t v2n = bn0.val[2], v3n = bn0.val[3];
                    const float32x4_t v4n = bn1.val[0], v5n = bn1.val[1];
                    const float32x4_t v6n = bn1.val[2], v7n = bn1.val[3];
#else
                    const float32x4_t v0n = vld1q_f32(inputRow + ik + 1);
                    const float32x4_t v1n = vld1q_f32(inputRow + ik + 5);
                    const float32x4_t v2n = vld1q_f32(inputRow + ik + 9);
                    const float32x4_t v3n = vld1q_f32(inputRow + ik + 13);
                    const float32x4_t v4n = vld1q_f32(inputRow + ik + 17);
                    const float32x4_t v5n = vld1q_f32(inputRow + ik + 21);
                    const float32x4_t v6n = vld1q_f32(inputRow + ik + 25);
                    const float32x4_t v7n = vld1q_f32(inputRow + ik + 29);
#endif
#if defined(CONV_ROW2_INTERLEAVE)
                    if (use0 && use1) {
                        const float32x4_t w0n = CONV_WEIGHT(kernelRow0 + ik + 1);
                        const float32x4_t w1n = CONV_WEIGHT(kernelRow1 + ik + 1);
                        s0[0] = vfmaq_f32(s0[0], v0n, w0n); s1[0] = vfmaq_f32(s1[0], v0n, w1n);
                        s0[1] = vfmaq_f32(s0[1], v1n, w0n); s1[1] = vfmaq_f32(s1[1], v1n, w1n);
                        s0[2] = vfmaq_f32(s0[2], v2n, w0n); s1[2] = vfmaq_f32(s1[2], v2n, w1n);
                        s0[3] = vfmaq_f32(s0[3], v3n, w0n); s1[3] = vfmaq_f32(s1[3], v3n, w1n);
                        s0[4] = vfmaq_f32(s0[4], v4n, w0n); s1[4] = vfmaq_f32(s1[4], v4n, w1n);
                        s0[5] = vfmaq_f32(s0[5], v5n, w0n); s1[5] = vfmaq_f32(s1[5], v5n, w1n);
                        s0[6] = vfmaq_f32(s0[6], v6n, w0n); s1[6] = vfmaq_f32(s1[6], v6n, w1n);
                        s0[7] = vfmaq_f32(s0[7], v7n, w0n); s1[7] = vfmaq_f32(s1[7], v7n, w1n);
                    } else
#endif
                    {
                        if (use0) {
                            const float32x4_t w0n = CONV_WEIGHT(kernelRow0 + ik + 1);
                            s0[0] = vfmaq_f32(s0[0], v0n, w0n); s0[1] = vfmaq_f32(s0[1], v1n, w0n);
                            s0[2] = vfmaq_f32(s0[2], v2n, w0n); s0[3] = vfmaq_f32(s0[3], v3n, w0n);
                            s0[4] = vfmaq_f32(s0[4], v4n, w0n); s0[5] = vfmaq_f32(s0[5], v5n, w0n);
                            s0[6] = vfmaq_f32(s0[6], v6n, w0n); s0[7] = vfmaq_f32(s0[7], v7n, w0n);
                        }
                        if (use1) {
                            const float32x4_t w1n = CONV_WEIGHT(kernelRow1 + ik + 1);
                            s1[0] = vfmaq_f32(s1[0], v0n, w1n); s1[1] = vfmaq_f32(s1[1], v1n, w1n);
                            s1[2] = vfmaq_f32(s1[2], v2n, w1n); s1[3] = vfmaq_f32(s1[3], v3n, w1n);
                            s1[4] = vfmaq_f32(s1[4], v4n, w1n); s1[5] = vfmaq_f32(s1[5], v5n, w1n);
                            s1[6] = vfmaq_f32(s1[6], v6n, w1n); s1[7] = vfmaq_f32(s1[7], v7n, w1n);
                        }
                    }
                }
                if (kernelWidth & 1) {
                    const CONVINT ik = kernelWidth - 1;
#if defined(CONV_ROW2_LOADX4)
                    const float32x4x4_t bo0 = vld1q_f32_x4(inputRow + ik);
                    const float32x4x4_t bo1 = vld1q_f32_x4(inputRow + ik + 16);
                    const float32x4_t v0 = bo0.val[0], v1 = bo0.val[1];
                    const float32x4_t v2 = bo0.val[2], v3 = bo0.val[3];
                    const float32x4_t v4 = bo1.val[0], v5 = bo1.val[1];
                    const float32x4_t v6 = bo1.val[2], v7 = bo1.val[3];
#else
                    const float32x4_t v0 = vld1q_f32(inputRow + ik);
                    const float32x4_t v1 = vld1q_f32(inputRow + ik + 4);
                    const float32x4_t v2 = vld1q_f32(inputRow + ik + 8);
                    const float32x4_t v3 = vld1q_f32(inputRow + ik + 12);
                    const float32x4_t v4 = vld1q_f32(inputRow + ik + 16);
                    const float32x4_t v5 = vld1q_f32(inputRow + ik + 20);
                    const float32x4_t v6 = vld1q_f32(inputRow + ik + 24);
                    const float32x4_t v7 = vld1q_f32(inputRow + ik + 28);
#endif
                    if (use0) {
                        const float32x4_t w0 = CONV_WEIGHT(kernelRow0 + ik);
                        s0[0] = vfmaq_f32(s0[0], v0, w0); s0[1] = vfmaq_f32(s0[1], v1, w0);
                        s0[2] = vfmaq_f32(s0[2], v2, w0); s0[3] = vfmaq_f32(s0[3], v3, w0);
                        s0[4] = vfmaq_f32(s0[4], v4, w0); s0[5] = vfmaq_f32(s0[5], v5, w0);
                        s0[6] = vfmaq_f32(s0[6], v6, w0); s0[7] = vfmaq_f32(s0[7], v7, w0);
                    }
                    if (use1) {
                        const float32x4_t w1 = CONV_WEIGHT(kernelRow1 + ik);
                        s1[0] = vfmaq_f32(s1[0], v0, w1); s1[1] = vfmaq_f32(s1[1], v1, w1);
                        s1[2] = vfmaq_f32(s1[2], v2, w1); s1[3] = vfmaq_f32(s1[3], v3, w1);
                        s1[4] = vfmaq_f32(s1[4], v4, w1); s1[5] = vfmaq_f32(s1[5], v5, w1);
                        s1[6] = vfmaq_f32(s1[6], v6, w1); s1[7] = vfmaq_f32(s1[7], v7, w1);
                    }
                }
            }

            vst1q_f32(out0 + i, s0[0]); vst1q_f32(out0 + i + 4, s0[1]);
            vst1q_f32(out0 + i + 8, s0[2]); vst1q_f32(out0 + i + 12, s0[3]);
            vst1q_f32(out0 + i + 16, s0[4]); vst1q_f32(out0 + i + 20, s0[5]);
            vst1q_f32(out0 + i + 24, s0[6]); vst1q_f32(out0 + i + 28, s0[7]);
            if (has_second) {
                vst1q_f32(out1 + i, s1[0]); vst1q_f32(out1 + i + 4, s1[1]);
                vst1q_f32(out1 + i + 8, s1[2]); vst1q_f32(out1 + i + 12, s1[3]);
                vst1q_f32(out1 + i + 16, s1[4]); vst1q_f32(out1 + i + 20, s1[5]);
                vst1q_f32(out1 + i + 24, s1[6]); vst1q_f32(out1 + i + 28, s1[7]);
            }
        }

        /* Small width tail: preserve scalar accumulation order exactly. */
        for (; i < outputWidth; ++i) {
            out0[i] = 0.0f;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                    out0[i] += input[(j + jk) * inputWidth + i + ik] *
                               kernel[jk * kernelWidth + ik];
            if (has_second) {
                out1[i] = 0.0f;
                for (CONVINT jk = 0; jk < kernelHeight; ++jk)
                    for (CONVINT ik = 0; ik < kernelWidth; ++ik)
                        out1[i] += input[(j + 1 + jk) * inputWidth + i + ik] *
                                    kernel[jk * kernelWidth + ik];
            }
        }
    }
#if defined(CONV_ROW2_PACKED_WEIGHT)
    conv_kernel_base = NULL;
    conv_packed_kernel_base = NULL;
    free(packedKernel);
#endif
}
#endif

// Compute adjacent output columns in NEON vectors. Each vector lane retains
// the baseline jk/ik accumulation order for its own output element.
void conv2d(const CONVFLOAT* restrict input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* restrict kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* restrict output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#if defined(__aarch64__)
#if defined(CONV_ENABLE_ROW4_VPACK)
    float32x4_t *row4VPack = (float32x4_t *)malloc(
        (size_t)(kernelHeight + 3) * (size_t)kernelWidth * sizeof(*row4VPack));
    if (row4VPack == NULL)
        abort();
    for (CONVINT r = 0; r < kernelHeight + 3; ++r) {
        for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
            float w[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (CONVINT q = 0; q < 4; ++q) {
                const CONVINT kr = r - q;
                if (kr >= 0 && kr < kernelHeight)
                    w[q] = kernel[kr * kernelWidth + ik];
            }
            row4VPack[r * kernelWidth + ik] = vld1q_f32(w);
        }
    }
    conv_row4_vpack_base = row4VPack;
    if (kernelWidth >= 32) {
        conv2d_row4_vpack(input, inputWidth, kernel, kernelHeight, kernelWidth,
                          output, outputHeight, outputWidth);
        conv_row4_vpack_base = NULL;
        free(row4VPack);
        return;
    }
    conv_row4_vpack_base = NULL;
    free(row4VPack);
#elif defined(CONV_ENABLE_ROW8)
    if (kernelWidth >= 32) {
        conv2d_row8(input, inputWidth, kernel, kernelHeight, kernelWidth,
                    output, outputHeight, outputWidth);
        return;
    }
#elif defined(CONV_ENABLE_ROW4_W20)
    if (kernelWidth >= 32) {
        conv2d_row4_w20(input, inputWidth, kernel, kernelHeight, kernelWidth,
                        output, outputHeight, outputWidth);
        return;
    }
#elif defined(CONV_ENABLE_ROW4_LANE4)
    if (kernelWidth >= 32) {
        conv2d_row4_lane4(input, inputWidth, kernel, kernelHeight, kernelWidth,
                          output, outputHeight, outputWidth);
        return;
    }
#elif !defined(CONV_DISABLE_ROW4_PAIR)
    if (kernelWidth >= 32) {
        conv2d_row4_pair(input, inputWidth, kernel, kernelHeight, kernelWidth,
                         output, outputHeight, outputWidth);
        return;
    }
#elif defined(CONV_ENABLE_ROW2_FAST)
    if (kernelWidth >= 32) {
        conv2d_row2_fast(input, inputWidth, kernel, kernelHeight, kernelWidth,
                         output, outputHeight, outputWidth);
        return;
    }
#elif defined(CONV_ENABLE_ROW3)
    if (kernelWidth >= 32) {
        conv2d_row3(input, inputWidth, kernel, kernelHeight, kernelWidth,
                    output, outputHeight, outputWidth);
        return;
    }
#elif defined(CONV_ENABLE_ROW4)
    if (kernelWidth >= 32) {
        conv2d_row4(input, inputWidth, kernel, kernelHeight, kernelWidth,
                    output, outputHeight, outputWidth);
        return;
    }
#elif !defined(CONV_DISABLE_ROW2)
    if (kernelWidth >= 32) {
        conv2d_row2(input, inputWidth, kernel, kernelHeight, kernelWidth,
                    output, outputHeight, outputWidth);
        return;
    }
#endif
#endif

#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* const outputRow = output + j * outputWidth;
        CONVINT i = 0;
#if defined(__aarch64__)
#if !defined(CONV_DISABLE_WIDE32)
        if (kernelWidth >= 32) {
            for (; i + 31 < outputWidth; i += 32) {
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                float32x4_t sum4 = vdupq_n_f32(0.0f);
                float32x4_t sum5 = vdupq_n_f32(0.0f);
                float32x4_t sum6 = vdupq_n_f32(0.0f);
                float32x4_t sum7 = vdupq_n_f32(0.0f);

                for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                    const CONVFLOAT* const inputRow = input + (j + jk) * inputWidth + i;
                    const CONVFLOAT* const kernelRow = kernel + jk * kernelWidth;

#if defined(CONV_DISABLE_UNROLL2)
                    for (CONVINT ik = 0; ik + 1 < kernelWidth; ik += 2) {
                        const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 8), weight);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 12), weight);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 16), weight);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 20), weight);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 24), weight);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 28), weight);
                        const float32x4_t weight_next = CONV_WEIGHT(kernelRow + ik + 1);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik + 1), weight_next);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 5), weight_next);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 9), weight_next);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 13), weight_next);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 17), weight_next);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 21), weight_next);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 25), weight_next);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 29), weight_next);
                    }
                    for (CONVINT ik = (kernelWidth & ~1); ik < kernelWidth; ++ik) {
                        const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 8), weight);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 12), weight);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 16), weight);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 20), weight);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 24), weight);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 28), weight);
                    }
#else
                    CONVINT ik = 0;
                    for (; ik + 1 < kernelWidth; ik += 2) {
                        const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 8), weight);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 12), weight);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 16), weight);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 20), weight);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 24), weight);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 28), weight);

                        const float32x4_t weight1 = CONV_WEIGHT(kernelRow + ik + 1);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik + 1), weight1);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 5), weight1);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 9), weight1);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 13), weight1);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 17), weight1);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 21), weight1);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 25), weight1);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 29), weight1);

                    }
                    for (; ik < kernelWidth; ++ik) {
                        const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 8), weight);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 12), weight);
                        sum4 = vfmaq_f32(sum4, vld1q_f32(inputRow + ik + 16), weight);
                        sum5 = vfmaq_f32(sum5, vld1q_f32(inputRow + ik + 20), weight);
                        sum6 = vfmaq_f32(sum6, vld1q_f32(inputRow + ik + 24), weight);
                        sum7 = vfmaq_f32(sum7, vld1q_f32(inputRow + ik + 28), weight);
                    }
#endif
                }

                vst1q_f32(outputRow + i, sum0);
                vst1q_f32(outputRow + i + 4, sum1);
                vst1q_f32(outputRow + i + 8, sum2);
                vst1q_f32(outputRow + i + 12, sum3);
                vst1q_f32(outputRow + i + 16, sum4);
                vst1q_f32(outputRow + i + 20, sum5);
                vst1q_f32(outputRow + i + 24, sum6);
                vst1q_f32(outputRow + i + 28, sum7);
            }
        }
#endif
#if !defined(CONV_DISABLE_WIDE16)
        if (kernelWidth >= 32) {
            for (; i + 15 < outputWidth; i += 16) {
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                    const CONVFLOAT* const inputRow = input + (j + jk) * inputWidth + i;
                    const CONVFLOAT* const kernelRow = kernel + jk * kernelWidth;

                    for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                        const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                        sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                        sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                        sum2 = vfmaq_f32(sum2, vld1q_f32(inputRow + ik + 8), weight);
                        sum3 = vfmaq_f32(sum3, vld1q_f32(inputRow + ik + 12), weight);
                    }
                }

                vst1q_f32(outputRow + i, sum0);
                vst1q_f32(outputRow + i + 4, sum1);
                vst1q_f32(outputRow + i + 8, sum2);
                vst1q_f32(outputRow + i + 12, sum3);
            }
        }
#endif

        for (; i + 7 < outputWidth; i += 8) {
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                    sum0 = vfmaq_f32(sum0, vld1q_f32(inputRow + ik), weight);
                    sum1 = vfmaq_f32(sum1, vld1q_f32(inputRow + ik + 4), weight);
                }
            }

            vst1q_f32(outputRow + i, sum0);
            vst1q_f32(outputRow + i + 4, sum1);
        }

        for (; i + 3 < outputWidth; i += 4) {
            float32x4_t sum = vdupq_n_f32(0.0f);

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const float32x4_t values = vld1q_f32(inputRow + ik);
                    const float32x4_t weight = CONV_WEIGHT(kernelRow + ik);
                    sum = vfmaq_f32(sum, values, weight);
                }
            }

            vst1q_f32(outputRow + i, sum);
        }
#endif

        for (; i < outputWidth; ++i) {
            // Keep the reference's scalar accumulation semantics for the
            // one-to-three-column tail (important for the strict tolerance).
            outputRow[i] = 0.0f;

            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* const inputRow = input + (j + jk) * inputWidth + i;
                const CONVFLOAT* const kernelRow = kernel + jk * kernelWidth;

                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    outputRow[i] += inputRow[ik] * kernelRow[ik];
                }
            }
        }
    }
}
