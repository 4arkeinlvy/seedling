/* seedling C core: public API. Python calls these through ctypes (seedling/_c.py).
 *
 * Convention: every function returns 0 on success or a negative errno value
 * (e.g. -EINVAL, -ECONNRESET); Python turns that into OSError.
 */
#ifndef SEEDLING_H
#define SEEDLING_H

#include <stddef.h>
#include <stdint.h>

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

/* ---- M4: bit-exact core ---------------------------------------------------- */

/* Philox4x32-10: out = philox(counter, key), 128 random bits. */
int sl_philox4x32(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);

/* out[j] = +1 or -1 for elements offset .. offset+n-1 of the sign stream (seed, stream).
 * Element i uses bit i % 128 of philox(counter = (i / 128, stream, 0), key = seed). */
int sl_rademacher_i8(uint64_t seed, uint32_t stream, uint64_t offset, int8_t *out, size_t n);

/* state[i] += sum_j coefs[j] * sign(seeds[j], stream, i), saturated to int32.
 * Exact: the same bits for any thread count. */
int sl_apply_rademacher_i32(int32_t *state, size_t n, const uint64_t *seeds, const int32_t *coefs,
                            size_t k, uint32_t stream);

/* delta (rows x cols, row-major) += sum_j coefs[j] * b_j a_j^T with
 * a_j[c] = sign(seeds[j], stream_a, c), b_j[r] = sign(seeds[j], stream_b, r). Exact, saturated. */
int sl_apply_lowrank_i32(int32_t *delta, size_t rows, size_t cols, const uint64_t *seeds,
                         const int32_t *coefs, size_t k, uint32_t stream_a, uint32_t stream_b);

/* xxHash64 of len bytes (spec-compatible with the reference implementation). */
int sl_xxh64_into(const void *data, size_t len, uint64_t seed, uint64_t *out);

/* Number of OpenMP threads for the functions above (tests use it to prove invariance). */
int sl_set_num_threads(int threads);

#endif /* SEEDLING_H */
