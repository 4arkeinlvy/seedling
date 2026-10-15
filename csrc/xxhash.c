/* M4: xxHash64, written from the specification (github.com/Cyan4973/xxHash, doc/xxhash_spec.md).
 * A fast 64-bit fingerprint: two replicas with equal hashes hold equal bytes (with
 * overwhelming probability). Explained in docs/M4-explained.md.
 */
#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "seedling.h"

#define P1 0x9E3779B185EBCA87ull
#define P2 0xC2B2AE3D27D4EB4Full
#define P3 0x165667B19E3779F9ull
#define P4 0x85EBCA77C2B2AE63ull
#define P5 0x27D4EB2F165667C5ull

static inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

static inline uint64_t read64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline uint32_t read32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static inline uint64_t lane_round(uint64_t acc, uint64_t lane) {
    acc += lane * P2;
    acc = rotl(acc, 31);
    return acc * P1;
}

static inline uint64_t merge_round(uint64_t acc, uint64_t value) {
    acc ^= lane_round(0, value);
    return acc * P1 + P4;
}

static uint64_t xxh64(const void *data, size_t len, uint64_t seed) {
    const uint8_t *p = data;
    const uint8_t *end = p + len;
    uint64_t h;

    if (len >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        const uint8_t *limit = end - 32;
        do {
            v1 = lane_round(v1, read64(p));
            v2 = lane_round(v2, read64(p + 8));
            v3 = lane_round(v3, read64(p + 16));
            v4 = lane_round(v4, read64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed + P5;
    }
    h += (uint64_t)len;

    for (; p + 8 <= end; p += 8) h = rotl(h ^ lane_round(0, read64(p)), 27) * P1 + P4;
    if (p + 4 <= end) {
        h = rotl(h ^ ((uint64_t)read32(p) * P1), 23) * P2 + P3;
        p += 4;
    }
    for (; p < end; p++) h = rotl(h ^ ((uint64_t)*p * P5), 11) * P1;

    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

int sl_xxh64_into(const void *data, size_t len, uint64_t seed, uint64_t *out) {
    static const uint8_t empty = 0;
    if (len > 0 && data == NULL) return -EINVAL;
    *out = xxh64(len == 0 ? &empty : data, len, seed);
    return 0;
}
