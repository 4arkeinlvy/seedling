/* M4: exact integer updates.
 *
 *   state[i] += sum over j of coefs[j] * sign(seeds[j], stream, i)
 *
 * Integer addition is exact and associative, so the result does not depend on the order
 * of elements, the number of threads, or the machine. Each element still adds its terms in
 * j order and saturates once at the end (no signed-overflow undefined behaviour).
 * Explained in docs/M4-explained.md.
 */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

#include "philox_internal.h"
#include "seedling.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static inline int32_t saturate(int64_t v) {
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return (int32_t)v;
}

int sl_apply_rademacher_i32(int32_t *state, size_t n, const uint64_t *seeds, const int32_t *coefs,
                            size_t k, uint32_t stream) {
    if (n > 0 && state == NULL) return -EINVAL;
    if (k > 0 && (seeds == NULL || coefs == NULL)) return -EINVAL;
    const size_t blocks = (n + SL_SIGNS_PER_BLOCK - 1) / SL_SIGNS_PER_BLOCK;

#pragma omp parallel for schedule(static) if (n > (1u << 16))
    for (size_t b = 0; b < blocks; b++) {
        const size_t start = b * SL_SIGNS_PER_BLOCK;
        const size_t len = (n - start < SL_SIGNS_PER_BLOCK) ? n - start : SL_SIGNS_PER_BLOCK;
        int64_t acc[SL_SIGNS_PER_BLOCK];
        for (size_t t = 0; t < len; t++) acc[t] = state[start + t];
        for (size_t j = 0; j < k; j++) {
            uint32_t words[4];
            sl_sign_block(seeds[j], stream, b, words);
            for (size_t t = 0; t < len; t++) acc[t] += (int64_t)coefs[j] * sl_sign_bit(words, (unsigned)t);
        }
        for (size_t t = 0; t < len; t++) state[start + t] = saturate(acc[t]);
    }
    return 0;
}

int sl_set_num_threads(int threads) {
    if (threads < 1) return -EINVAL;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    return 0;
}

/* Low-rank version for matrices (M6): with a_j[c] = sign(seeds[j], stream_a, c) and
 * b_j[r] = sign(seeds[j], stream_b, r),
 *   delta[r][c] += sum_j coefs[j] * b_j[r] * a_j[c]        (delta is rows x cols, row-major)
 * i.e. delta += sum_j coefs[j] * b_j a_j^T, one rank-1 matrix per seed. */
int sl_apply_lowrank_i32(int32_t *delta, size_t rows, size_t cols, const uint64_t *seeds,
                         const int32_t *coefs, size_t k, uint32_t stream_a, uint32_t stream_b) {
    if (rows * cols > 0 && delta == NULL) return -EINVAL;
    if (k > 0 && (seeds == NULL || coefs == NULL)) return -EINVAL;
    if (k == 0 || rows == 0 || cols == 0) return 0;

    int8_t *a = malloc(k * cols);
    int64_t *w = malloc(rows * k * sizeof *w);
    int8_t *b = malloc(rows);
    if (a == NULL || w == NULL || b == NULL) {
        free(a);
        free(w);
        free(b);
        return -ENOMEM;
    }
    for (size_t j = 0; j < k; j++) {
        sl_rademacher_i8(seeds[j], stream_a, 0, a + j * cols, cols);
        sl_rademacher_i8(seeds[j], stream_b, 0, b, rows);
        for (size_t r = 0; r < rows; r++) w[r * k + j] = (int64_t)coefs[j] * b[r];
    }

#pragma omp parallel for schedule(static) if (rows * cols > (1u << 16))
    for (size_t r = 0; r < rows; r++) {
        int32_t *row = delta + r * cols;
        const int64_t *wr = w + r * k;
        for (size_t c = 0; c < cols; c++) {
            int64_t acc = row[c];
            for (size_t j = 0; j < k; j++) acc += wr[j] * a[j * cols + c];
            row[c] = saturate(acc);
        }
    }
    free(a);
    free(w);
    free(b);
    return 0;
}
