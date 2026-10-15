/* C unit test for sl_apply_rademacher_i32: matches a naive loop, saturates, and gives the
 * same bits with 1 or 4 threads. Run: make test-c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "seedling.h"

static int failures = 0;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "test_apply: FAILED %s\n", what);
        failures++;
    }
}

static void matches_naive_loop(void) {
    enum { N = 1000, K = 5 };
    const uint64_t seeds[K] = {1, 2, 3, 0xffffffffffffull, 77};
    const int32_t coefs[K] = {10, -3, 1000, -70000, 1};
    int32_t state[N], want[N];
    int8_t signs[N];
    for (int i = 0; i < N; i++) state[i] = want[i] = i - 500;
    for (int j = 0; j < K; j++) {
        sl_rademacher_i8(seeds[j], 0, 0, signs, N);
        for (int i = 0; i < N; i++) want[i] += coefs[j] * signs[i];
    }
    expect(sl_apply_rademacher_i32(state, N, seeds, coefs, K, 0) == 0, "returns 0");
    expect(memcmp(state, want, sizeof state) == 0, "same as the naive loop");
}

static void saturates(void) {
    int32_t state[2] = {INT32_MAX - 5, INT32_MIN + 5};
    const uint64_t seeds[1] = {9};
    const int32_t coefs[1] = {INT32_MAX};
    int8_t signs[2];
    sl_rademacher_i8(9, 0, 0, signs, 2);
    sl_apply_rademacher_i32(state, 2, seeds, coefs, 1, 0);
    for (int i = 0; i < 2; i++) {
        int64_t exact = (int64_t)(i == 0 ? INT32_MAX - 5 : INT32_MIN + 5) + (int64_t)INT32_MAX * signs[i];
        int32_t want = exact > INT32_MAX ? INT32_MAX : exact < INT32_MIN ? INT32_MIN : (int32_t)exact;
        expect(state[i] == want, "saturates at the int32 limits");
    }
}

static void thread_count_does_not_matter(void) {
    const size_t n = (1u << 20) + 3;
    enum { K = 16 };
    uint64_t seeds[K];
    int32_t coefs[K];
    for (int j = 0; j < K; j++) {
        seeds[j] = 1000u + (uint64_t)j;
        coefs[j] = (j % 2 ? -1 : 1) * (j + 1) * 37;
    }
    int32_t *one = calloc(n, sizeof *one), *four = calloc(n, sizeof *four);
    sl_set_num_threads(1);
    sl_apply_rademacher_i32(one, n, seeds, coefs, K, 0);
    sl_set_num_threads(4);
    sl_apply_rademacher_i32(four, n, seeds, coefs, K, 0);
    expect(memcmp(one, four, n * sizeof *one) == 0, "1 thread == 4 threads");
    free(one);
    free(four);
}

int main(void) {
    matches_naive_loop();
    saturates();
    thread_count_does_not_matter();
    if (failures) return 1;
    printf("test_apply: OK\n");
    return 0;
}
