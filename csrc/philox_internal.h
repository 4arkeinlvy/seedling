/* Internal helpers shared by philox.c and apply.c (not part of the public API).
 *
 * Sign stream: element i of (seed, stream) uses bit i % 128 of the Philox block
 *   counter = (i / 128 low 32 bits, i / 128 high 32 bits, stream, 0), key = (seed low, seed high)
 * bit set -> +1, bit clear -> -1.
 */
#ifndef SEEDLING_PHILOX_INTERNAL_H
#define SEEDLING_PHILOX_INTERNAL_H

#include <stdint.h>

#define SL_SIGNS_PER_BLOCK 128u

void sl_philox_block(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);

static inline void sl_sign_block(uint64_t seed, uint32_t stream, uint64_t block, uint32_t words[4]) {
    const uint32_t counter[4] = {(uint32_t)block, (uint32_t)(block >> 32), stream, 0u};
    const uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    sl_philox_block(counter, key, words);
}

static inline int sl_sign_bit(const uint32_t words[4], unsigned bit) {
    return ((words[bit >> 5] >> (bit & 31u)) & 1u) ? 1 : -1;
}

#endif
