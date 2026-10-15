/* C unit test for sl_sum_f32's big brother, sl_ring_allreduce_f32.
 *
 * Each "rank" is a thread; neighbours are joined by socketpair()s, so this runs in one
 * process under AddressSanitizer. Unix socket buffers are small (~200 KB), so an
 * implementation that blocks in send() before receiving will deadlock here: the alarm
 * below turns that into a clear failure instead of a hang.
 *
 * Run: make test-c
 */
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "seedling.h"

#define MAX_WORLD 5

struct rank_args {
    int rank, world, left_fd, right_fd, rc;
    size_t n;
    float *buf;
};

/* Integer-valued inputs, so the expected sum is exact in float32. */
static float input(int rank, size_t i) {
    return (float)((i * 31 + (size_t)rank * 17) % 201) - 100.0f;
}

static void *run_rank(void *p) {
    struct rank_args *a = p;
    a->rc = sl_ring_allreduce_f32(a->left_fd, a->right_fd, a->buf, a->n, a->rank, a->world);
    return NULL;
}

static void on_alarm(int sig) {
    (void)sig;
    static const char msg[] =
        "test_ring: TIMEOUT - ranks are stuck. Deadlock? (every rank blocked in send?)\n";
    ssize_t ignored = write(2, msg, sizeof msg - 1);
    (void)ignored;
    _exit(1);
}

static void check(int world, size_t n) {
    struct rank_args args[MAX_WORLD];
    pthread_t threads[MAX_WORLD];
    int fds[MAX_WORLD][2];
    int npairs = 0;

    memset(args, 0, sizeof args);
    for (int r = 0; r < world; r++) {
        args[r].rank = r;
        args[r].world = world;
        args[r].n = n;
        args[r].left_fd = args[r].right_fd = -1;
        args[r].buf = malloc((n + 1) * sizeof(float));
        if (!args[r].buf) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
        for (size_t i = 0; i < n; i++) args[r].buf[i] = input(r, i);
    }

    if (world == 2) {
        /* Same as the Python mesh: one connection, used as both left and right. */
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds[0]) != 0) { perror("socketpair"); exit(1); }
        npairs = 1;
        args[0].left_fd = args[0].right_fd = fds[0][0];
        args[1].left_fd = args[1].right_fd = fds[0][1];
    } else if (world > 2) {
        /* pair k joins rank k (its right side) to rank k+1 (its left side) */
        for (int k = 0; k < world; k++) {
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds[k]) != 0) { perror("socketpair"); exit(1); }
            args[k].right_fd = fds[k][0];
            args[(k + 1) % world].left_fd = fds[k][1];
        }
        npairs = world;
    }

    for (int r = 0; r < world; r++) pthread_create(&threads[r], NULL, run_rank, &args[r]);
    for (int r = 0; r < world; r++) pthread_join(threads[r], NULL);

    for (int r = 0; r < world; r++) {
        if (args[r].rc == -ENOSYS) {
            fprintf(stderr, "test_ring: sl_ring_allreduce_f32 is still a TODO (csrc/ring.c)\n");
            exit(1);
        }
        if (args[r].rc != 0) {
            fprintf(stderr, "test_ring: world=%d n=%zu rank=%d returned %d (%s)\n", world, n, r,
                    args[r].rc, strerror(-args[r].rc));
            exit(1);
        }
        for (size_t i = 0; i < n; i++) {
            float want = 0.0f;
            for (int q = 0; q < world; q++) want += input(q, i);
            if (args[r].buf[i] != want) {
                fprintf(stderr, "test_ring: world=%d n=%zu rank=%d mismatch at i=%zu: got %g want %g\n",
                        world, n, r, i, (double)args[r].buf[i], (double)want);
                exit(1);
            }
        }
    }

    for (int k = 0; k < npairs; k++) {
        close(fds[k][0]);
        close(fds[k][1]);
    }
    for (int r = 0; r < world; r++) free(args[r].buf);
}

int main(void) {
    signal(SIGALRM, on_alarm);
    alarm(30);
    const size_t sizes[] = {0, 1, 2, 3, 5, 1000, (1u << 20) + 7};
    for (int world = 1; world <= MAX_WORLD; world++)
        for (size_t k = 0; k < sizeof sizes / sizeof sizes[0]; k++) check(world, sizes[k]);
    printf("test_ring: OK\n");
    return 0;
}
