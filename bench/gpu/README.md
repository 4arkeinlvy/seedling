# GPU experiments (run these yourself)

The CPU benchmarks in `bench/bench_scale.py` run anywhere. These need a GPU: a laptop RTX 3050
(4 GB) or a free Kaggle notebook (2x T4, 16 GB each). Every command prints JSON lines; send
back the last lines, or the files they write, and they go into `RESULTS.md` marked as user-run.

Setup (Kaggle: a GPU notebook with internet on):

```bash
git clone -b m9-benchmarks https://github.com/4arkeinlvy/seedling && cd seedling
pip install -e ".[llm]"        # builds the C core (gcc); the bench/ scripts run from this folder
# T4 has no fast bfloat16: add --dtype float16 to the seedling.llm.train commands below
```

## 1. Countdown with the protocol of Qiu et al. (2025)

Same data (TinyZero Countdown, 3-4 numbers), prompt and score (0.1 for a tagged answer, 1 if
correct) as "Evolution Strategies at Scale", with their population (30) and noise (0.001).
Ours is rank-1 noise on 16 layers instead of full-parameter noise.

```bash
# base model only: should be close to the paper's base accuracy (about 0.1% for 0.5B)
python -m seedling.llm.train --protocol tinyzero --steps 0 --eval-problems 512 \
    --max-new-tokens 512 --max-rows 64 --out base.csv

# 200 ES steps, journaled (run the same command again to resume after a disconnect)
python -m seedling.llm.train --protocol tinyzero --pop 30 --sigma 0.001 --lr 2e-5 --steps 200 \
    --prompts 16 --eval-every 25 --eval-problems 512 --max-new-tokens 512 --max-rows 64 \
    --run-dir runs/countdown-tz --out countdown-tz.csv
```

`--model Qwen/Qwen2.5-1.5B-Instruct` for the bigger model (Kaggle T4; `--max-rows 32`).
Send: the first and last JSON lines and `countdown-tz.csv`.

## 2. Do honest GPUs agree? (sets the audit tolerance for LLMs)

```bash
python -m bench.bench_agreement --bench countdown sst2 --device cuda --out agreement-gpu.json
python -m bench.bench_agreement --bench countdown sst2 --device cpu  --out agreement-cpu.json
python -m bench.bench_agreement --compare agreement-cpu.json agreement-gpu.json
```

Send both JSON files. Greedy decoding can pick a different token when two logits nearly
tie, so CPU and GPU may write different answers: this measures how often.

## 3. A real liar against a real coordinator

```bash
# machine A (no GPU needed): the coordinator, with spot checks
seedling coordinator --task countdown --task-arg protocol=tinyzero --task-arg max_new_tokens=256 \
    --steps 20 --pop 30 --sigma 0.001 --lr 2e-5 --frac-bits 24 --pairs-per-task 2 \
    --run-dir runs/cd-audit --listen http://0.0.0.0:7080 --audit-rate 0.1 --probation 4
cloudflared tunnel --url http://localhost:7080          # gives https://<name>.trycloudflare.com

# two or more honest GPU workers (Kaggle notebooks, the laptop)
seedling worker --connect https://<name>.trycloudflare.com

# one liar
python -m bench.gpu.dishonest_worker --connect https://<name>.trycloudflare.com --behaviour flip
```

The coordinator prints `banned <id>: outvoted on pair ...`; the liar prints `"caught": true`.
Send the coordinator's output.
