/* seedling C core: public API. Python calls these through ctypes (seedling/_c.py).
 *
 * Convention: every function returns 0 on success or a negative errno value
 * (e.g. -EINVAL, -ECONNRESET); Python turns that into OSError.
 */
#ifndef SEEDLING_H
#define SEEDLING_H

#include <stddef.h>

/* "seedling x.y.z": lets Python check that the library loaded. */
const char *sl_version(void);

/* dst[i] += src[i] for i in [0, n). dst and src must not overlap. */
int sl_sum_f32(float *restrict dst, const float *restrict src, size_t n);

/* In-place SUM all-reduce of buf[0..n) over a ring of `world` processes.
 *   left_fd:  connected socket to rank (rank - 1 + world) % world  (receive from it)
 *   right_fd: connected socket to rank (rank + 1) % world          (send to it)
 * When world == 2 both neighbours are the same rank, so left_fd == right_fd
 * (one TCP connection used in both directions). When world == 1 the fds are -1.
 * Afterwards every rank holds exactly the same bits. */
int sl_ring_allreduce_f32(int left_fd, int right_fd, float *buf, size_t n,
                          int rank, int world);

#endif /* SEEDLING_H */
