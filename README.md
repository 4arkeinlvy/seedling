# seedling

Train models on pooled, unreliable hardware (a laptop GPU, spare CPU cores, free
Kaggle/Colab GPUs, friends' PCs) by sending only **random seeds and scores** over the
network.

seedling uses evolution strategies (ES): each worker tries a few random changes to the
model (each change is regenerated from a seed), runs the model forward, and reports a
score. A training step is then just a list of `(seed, coefficient)` pairs, a few hundred
bytes. Every worker replays that list with exact integer math in C, so all copies of the
model stay **bit-for-bit identical**, and a whole training run becomes a small log you
can replay.

> Status: learning project, built from first principles. M0–M3 are done; M4 (bit-exact C core) is next.

## Roadmap

| | Milestone | What you build |
|---|---|---|
| [x] | **M0** tiny GPT | char-level GPT in PyTorch: the first workload |
| [x] | **M1** comm layer | framing, rendezvous store, collectives; ring all-reduce in Python and C |
| [x] | **M2** DDP | data-parallel training from scratch: the mainstream baseline |
| [x] | **M3** ES engine | evolution strategies on CartPole, MNIST, tiny GPT |
| [ ] | **M4** bit-exact core | Philox RNG, integer state updates, hashing in C |
| [ ] | **M5** runtime | coordinator + workers: replicated log, failures, join/leave, NAT |
| [ ] | **M6** LLM demo | fine-tune Qwen2.5-0.5B on Countdown with laptop + Kaggle |
| [ ] | **M7** package | pip install, CLI, CI, blog post |

## Setup (NixOS)

```bash
nix-shell                 # gcc, make, gdb, valgrind + lets PyTorch find the GPU
uv sync                   # Python 3.13 + PyTorch 2.14 (CUDA 13) + numpy + pytest
make                      # builds build/libseedling.so
uv run pytest             # the spec: tests for your TODOs fail until you write them
```

Always work inside `nix-shell`. On NixOS the NVIDIA driver library lives in
`/run/opengl-driver/lib`, and `shell.nix` puts it on `LD_LIBRARY_PATH`. Outside the shell
`torch.cuda.is_available()` is `False`, and `test_cuda_works` fails to remind you.

## How to work

- The tests are the spec. Don't edit them, except to add more.
- `tests/test_provided.py` passes from the start. If it fails, the environment is broken,
  not your code.
- `make test-c` runs the C tests under AddressSanitizer + UBSan.
- `uv run pytest -m heavy` runs the tests that need ~1 GB extra RAM.

## Try it

```bash
uv run python -m seedling.tasks.tiny_gpt                                  # M0: train + sample
uv run python -m bench.bench_allreduce                                    # M1: ring bandwidth
uv run python -m seedling.foundation.launch -n 2 seedling.tasks.train_ddp:main   # M2: 2-process DDP
uv run python -m bench.bench_ddp                                          # M2: step-time breakdown
uv run python -m seedling.tasks.es_demo --task cartpole                   # M3: also mnist, gpt
uv run python -m bench.plot_es                                            # M3: plots
```

## Layout

```
csrc/            C core (ctypes-loaded): ring all-reduce now; RNG + exact updates in M4
seedling/
  _c.py          ctypes bindings
  data.py        char tokenizer + batches
  es/            evolution strategies: noise, ES (ask/tell), optimizers, evaluators, BatchedMLP
  tasks/         workloads: tiny_gpt, train_ddp, cartpole, mnist, es_gpt, es_demo
  foundation/    launch.py, wire.py, store.py, collectives.py, ddp.py
tests/           the spec
bench/           measurements: all-reduce bandwidth, DDP step time, ES curves
```
