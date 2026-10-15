/* C unit test for Philox4x32-10 and the Rademacher sign stream. Run: make test-c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "seedling.h"

static int failures = 0;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "test_philox: FAILED %s\n", what);
        failures++;
    }
}

/* Known-answer vectors from Random123 (kat_vectors, philox4x32 with 10 rounds). */
static void known_answers(void) {
    const uint32_t cases[3][10] = {
        {0, 0, 0, 0, 0, 0, 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
        {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
         0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
        {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0,
         0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
    };
    for (int c = 0; c < 3; c++) {
        uint32_t out[4];
        sl_philox4x32(cases[c], cases[c] + 4, out);
        for (int w = 0; w < 4; w++) expect(out[w] == cases[c][6 + w], "Random123 known answer");
    }
}

static void signs_are_random_access(void) {
    enum { N = 1000, OFFSET = 333 };
    int8_t full[N + OFFSET], tail[N];
    sl_rademacher_i8(42, 3, 0, full, N + OFFSET);
    sl_rademacher_i8(42, 3, OFFSET, tail, N);
    int balance = 0;
    for (int i = 0; i < N; i++) {
        expect(tail[i] == full[OFFSET + i], "offset gives the same signs");
        expect(tail[i] == 1 || tail[i] == -1, "signs are +-1");
        balance += tail[i];
    }
    expect(abs(balance) < 150, "signs are roughly balanced");

    int8_t other[N];
    sl_rademacher_i8(42, 4, OFFSET, other, N);
    int same = 0;
    for (int i = 0; i < N; i++) same += tail[i] == other[i];
    expect(same > 400 && same < 600, "another stream gives independent signs");
}

int main(void) {
    known_answers();
    signs_are_random_access();
    if (failures) return 1;
    printf("test_philox: OK\n");
    return 0;
}
