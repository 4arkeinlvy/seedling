/* M4: Philox4x32-10, a counter-based random number generator (Salmon et al., SC'11).
 *
 * random block = philox(counter, key). No hidden state: any block can be computed
 * directly, in any order, on any machine, and gives the same 128 bits.
 * Explained in docs/M4-explained.md.
 */
#include <stdint.h>

#include "philox_internal.h"
#include "seedling.h"

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u
#define PHILOX_ROUNDS 10

static inline void mulhilo(uint32_t a, uint32_t b, uint32_t *hi, uint32_t *lo) {
    uint64_t product = (uint64_t)a * b;
    *hi = (uint32_t)(product >> 32);
    *lo = (uint32_t)product;
}

void sl_philox_block(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]) {
    uint32_t c0 = counter[0], c1 = counter[1], c2 = counter[2], c3 = counter[3];
    uint32_t k0 = key[0], k1 = key[1];
    for (int round = 0; round < PHILOX_ROUNDS; round++) {
        if (round > 0) {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo(PHILOX_M0, c0, &hi0, &lo0);
        mulhilo(PHILOX_M1, c2, &hi1, &lo1);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

int sl_philox4x32(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]) {
    sl_philox_block(counter, key, out);
    return 0;
}

int sl_rademacher_i8(uint64_t seed, uint32_t stream, uint64_t offset, int8_t *out, size_t n) {
    uint32_t words[4];
    uint64_t current = UINT64_MAX;
    for (size_t j = 0; j < n; j++) {
        uint64_t i = offset + j;
        uint64_t block = i / SL_SIGNS_PER_BLOCK;
        if (block != current) {
            sl_sign_block(seed, stream, block, words);
            current = block;
        }
        out[j] = (int8_t)sl_sign_bit(words, (unsigned)(i % SL_SIGNS_PER_BLOCK));
    }
    return 0;
}
