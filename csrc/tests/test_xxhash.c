/* C unit test for xxHash64; expected values come from the reference implementation. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "seedling.h"

static int failures = 0;

static void check(const void *data, size_t len, uint64_t seed, uint64_t want, const char *what) {
    uint64_t got = 0;
    if (sl_xxh64_into(data, len, seed, &got) != 0 || got != want) {
        fprintf(stderr, "test_xxhash: %s: got %016llx want %016llx\n", what,
                (unsigned long long)got, (unsigned long long)want);
        failures++;
    }
}

int main(void) {
    unsigned char pattern[1024], counting[37];
    for (int i = 0; i < 1024; i++) pattern[i] = (unsigned char)(i % 256);
    for (int i = 0; i < 37; i++) counting[i] = (unsigned char)i;

    check(NULL, 0, 0, 0xef46db3751d8e999ull, "empty input");
    check("a", 1, 0, 0xd24ec4f1a98c6e5bull, "one byte");
    check("abc", 3, 0, 0x44bc2cf5ad770999ull, "three bytes");
    check("seedling", 8, 7, 0xb1e1e4c3295e2999ull, "one lane, seed 7");
    check(pattern, sizeof pattern, 0, 0x6f3914f18fe4df57ull, "1024 bytes (stripes)");
    check(counting, sizeof counting, UINT64_MAX, 0xc8ced3dbe3bff419ull, "37 bytes, max seed");

    if (failures) return 1;
    printf("test_xxhash: OK\n");
    return 0;
}
