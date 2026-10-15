/* M1: the C data path. Python opens the sockets; C moves and adds the floats.
 *
 *   build:        make
 *   C tests:      make test-c      (AddressSanitizer + UBSan, threads as ranks)
 *   Python tests: uv run pytest tests/test_m1_collectives.py -k backend_c
 *   explained:    docs/M1-explained.md
 */
#define _GNU_SOURCE
#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <sys/socket.h>

#include "seedling.h"

#define POLL_TIMEOUT_MS 60000

/* dst[i] += src[i]. `restrict` lets the compiler vectorize (AVX2: 8 floats per add);
 * OpenMP threads only pay off for big arrays. */
int sl_sum_f32(float *restrict dst, const float *restrict src, size_t n) {
#pragma omp parallel for simd schedule(static) if (n > (1 << 16))
    for (size_t i = 0; i < n; i++) dst[i] += src[i];
    return 0;
}

/* Chunk c of the buffer covers elements [chunk_begin(c), chunk_begin(c + 1)). */
static size_t chunk_begin(size_t n, int world, int c) {
    return (size_t)c * n / (size_t)world;
}

/* a mod m, always in [0, m) (in C, -1 % 3 == -1). */
static int mod(int a, int m) { return ((a % m) + m) % m; }

/* Send slen bytes to right_fd WHILE receiving rlen bytes from left_fd.
 * Doing both at once is what prevents the ring deadlock. */
static int exchange(int left_fd, int right_fd, const char *sbuf, size_t slen, char *rbuf,
                    size_t rlen) {
    size_t sent = 0, got = 0;
    while (sent < slen || got < rlen) {
        struct pollfd pfd[2];
        int nfds = 0, si = -1, ri = -1;
        if (left_fd == right_fd) { /* world == 2: one socket, both directions */
            pfd[0].fd = left_fd;
            pfd[0].events = (short)((sent < slen ? POLLOUT : 0) | (got < rlen ? POLLIN : 0));
            pfd[0].revents = 0;
            si = ri = 0;
            nfds = 1;
        } else {
            if (sent < slen) {
                pfd[nfds] = (struct pollfd){.fd = right_fd, .events = POLLOUT};
                si = nfds++;
            }
            if (got < rlen) {
                pfd[nfds] = (struct pollfd){.fd = left_fd, .events = POLLIN};
                ri = nfds++;
            }
        }

        int ready = poll(pfd, (nfds_t)nfds, POLL_TIMEOUT_MS);
        if (ready < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (ready == 0) return -ETIMEDOUT;

        if (sent < slen && (pfd[si].revents & (POLLOUT | POLLERR | POLLHUP))) {
            /* MSG_NOSIGNAL: a dead peer gives EPIPE instead of killing us with SIGPIPE */
            ssize_t k = send(right_fd, sbuf + sent, slen - sent, MSG_DONTWAIT | MSG_NOSIGNAL);
            if (k > 0)
                sent += (size_t)k;
            else if (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                return -errno;
        }
        if (got < rlen && (pfd[ri].revents & (POLLIN | POLLERR | POLLHUP))) {
            ssize_t k = recv(left_fd, rbuf + got, rlen - got, MSG_DONTWAIT);
            if (k > 0)
                got += (size_t)k;
            else if (k == 0)
                return -ECONNRESET; /* peer closed the connection */
            else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                return -errno;
        }
    }
    return 0;
}

int sl_ring_allreduce_f32(int left_fd, int right_fd, float *buf, size_t n, int rank,
                          int world) {
    if (world < 1 || rank < 0 || rank >= world) return -EINVAL;
    if (world == 1 || n == 0) return 0;

    size_t max_chunk = (n + (size_t)world - 1) / (size_t)world;
    float *tmp = malloc(max_chunk * sizeof *tmp);
    if (!tmp) return -ENOMEM;

    int rc = 0;
    /* Reduce-scatter: after world-1 steps this rank owns the full sum of chunk rank+1. */
    for (int s = 0; s < world - 1 && rc == 0; s++) {
        int sc = mod(rank - s, world), rcv = mod(rank - s - 1, world);
        size_t sb = chunk_begin(n, world, sc), se = chunk_begin(n, world, sc + 1);
        size_t rb = chunk_begin(n, world, rcv), re = chunk_begin(n, world, rcv + 1);
        rc = exchange(left_fd, right_fd, (const char *)(buf + sb), (se - sb) * sizeof(float),
                      (char *)tmp, (re - rb) * sizeof(float));
        if (rc == 0) sl_sum_f32(buf + rb, tmp, re - rb);
    }
    /* All-gather: pass finished chunks around; receive straight into buf (a copy). */
    for (int s = 0; s < world - 1 && rc == 0; s++) {
        int sc = mod(rank + 1 - s, world), rcv = mod(rank - s, world);
        size_t sb = chunk_begin(n, world, sc), se = chunk_begin(n, world, sc + 1);
        size_t rb = chunk_begin(n, world, rcv), re = chunk_begin(n, world, rcv + 1);
        rc = exchange(left_fd, right_fd, (const char *)(buf + sb), (se - sb) * sizeof(float),
                      (char *)(buf + rb), (re - rb) * sizeof(float));
    }
    free(tmp);
    return rc;
}
