"""M10: what a prefill audit costs, and how far honest tokens sit from the top logit.

A model with the exact architecture of Qwen2.5-0.5B (random weights: cost does not depend on
them, and no download is needed) generates completions; an auditor checks them with one
teacher-forced pass (seedling.llm.evidence). Reports the time of both, and the "gap" of each
honest token (max logit - its logit) when the auditor runs in the same precision through a
different code path (prefill vs the generator's KV cache) and in another precision.

    uv run python -m bench.bench_verify --rows 8 --new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def qwen05b(dtype: torch.dtype) -> torch.nn.Module:
    import transformers

    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
        num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=32768, tie_word_embeddings=True,
    )  # fmt: skip
    return transformers.Qwen2ForCausalLM(config).to(dtype).eval()


@torch.no_grad()
def gaps(model: torch.nn.Module, prompt: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    """max logit - claimed token's logit, at every completion position (one forward pass)."""
    ids = torch.cat([prompt, completion], dim=1)
    logits = model(ids, logits_to_keep=completion.shape[1] + 1).logits.float()[:, :-1]
    return logits.max(dim=-1).values - logits.gather(-1, completion[..., None])[..., 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    ns = parser.parse_args()
    torch.set_num_threads(ns.threads)

    fp32 = qwen05b(torch.float32)
    prompt = torch.randint(0, 151936, (ns.rows, ns.prompt_tokens), generator=torch.Generator().manual_seed(1))
    t0 = time.perf_counter()
    out = fp32.generate(prompt, max_new_tokens=ns.new_tokens, min_new_tokens=ns.new_tokens, do_sample=False)
    generate_s = time.perf_counter() - t0
    completion = out[:, ns.prompt_tokens :]

    t0 = time.perf_counter()
    same = gaps(fp32, prompt, completion)
    verify_s = time.perf_counter() - t0
    bf16 = qwen05b(torch.bfloat16)
    other = gaps(bf16, prompt, completion)

    def stats(g: torch.Tensor) -> dict[str, float]:
        return {"max": float(g.max()), "p99": float(g.flatten().quantile(0.99)), "share_nonzero": float((g > 0).float().mean())}

    result = {
        "model": "Qwen2.5-0.5B architecture (random weights), CPU",
        "rows": ns.rows, "prompt_tokens": ns.prompt_tokens, "new_tokens": ns.new_tokens, "threads": ns.threads,
        "generate_seconds": round(generate_s, 2), "verify_seconds": round(verify_s, 2),
        "audit_cost": round(verify_s / generate_s, 3),
        "gap_fp32_prefill_vs_fp32_generate": stats(same),
        "gap_bf16_prefill_vs_fp32_generate": stats(other),
    }  # fmt: skip
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
