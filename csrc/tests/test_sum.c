/* C unit test for sl_sum_f32. Run: make test-c (built with AddressSanitizer + UBSan,
 * so reading or writing even one float past the end is reported). */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "seedling.h"

static void check(size_t n) {
    /* +1 so malloc(0) never returns NULL; ASan still catches overruns past n. */
    float *dst = malloc((n + 1) * sizeof *dst);
    float *src = malloc((n + 1) * sizeof *src);
    float *want = malloc((n + 1) * sizeof *want);
    if (!dst || !src || !want) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (size_t i = 0; i < n; i++) {
        /* small integers: their float sums are exact, so we can compare with == */
        dst[i] = (float)(i % 1000) - 500.0f;
        src[i] = (float)((i * 7) % 300) - 150.0f;
        want[i] = dst[i] + src[i];
    }
    int rc = sl_sum_f32(dst, src, n);
    if (rc == -ENOSYS) {
        fprintf(stderr, "test_sum: sl_sum_f32 is still a TODO (csrc/ring.c)\n");
        exit(1);
    }
    if (rc != 0) {
        fprintf(stderr, "test_sum: n=%zu returned %d\n", n, rc);
        exit(1);
    }
    for (size_t i = 0; i < n; i++) {
        if (dst[i] != want[i]) {
            fprintf(stderr, "test_sum: n=%zu mismatch at i=%zu: got %g want %g\n", n, i,
                    (double)dst[i], (double)want[i]);
            exit(1);
        }
    }
    free(dst);
    free(src);
    free(want);
}

int main(void) {
    const size_t sizes[] = {0, 1, 7, 8, 9, 1000, 65536, 65537, (1u << 20) + 3};
    for (size_t k = 0; k < sizeof sizes / sizeof sizes[0]; k++) check(sizes[k]);
    printf("test_sum: OK\n");
    return 0;
}
