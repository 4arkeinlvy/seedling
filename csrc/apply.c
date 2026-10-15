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
