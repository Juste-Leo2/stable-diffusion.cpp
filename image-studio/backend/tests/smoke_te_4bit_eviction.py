"""Validate eviction + cache-miss reload with te_4bit=True.

Sequence:
  1. Cold: generate prompt A. TE loads, encodes, evicts. Peak includes TE.
  2. Warm cache hit: generate prompt A again. TE stays None. Peak should drop.
  3. Cache miss: generate prompt B. Patched reload fires (4-bit path), encodes, evicts.

Also directly times load_te_4bit() in isolation to report the raw reload wall-time
from cold (after the first run evicts).

Run:
  .venv/bin/python -m backend.tests.smoke_te_4bit_eviction
"""
from __future__ import annotations

import gc
import hashlib
import os
import time
from pathlib import Path

import mlx.core as mx

from backend.pipeline import FluxPipeline, PipelineConfig
from backend.text_encoder_4bit import load_te_4bit

PROMPT_A = (
    "A serene Scandinavian woman in her 40s, soft window light, natural skin texture, "
    "shallow depth of field, 50mm portrait lens, neutral linen backdrop."
)
PROMPT_B = (
    "A rusting steam locomotive abandoned in a coastal fog at dawn, "
    "wet cobblestones, muted teal palette, cinematic wide shot, 35mm film grain."
)
SEED = 42
STEPS = 4
H = W = 1024
OUT_DIR = Path(os.environ.get("BONSAI_SMOKE_OUT_DIR") or Path(__file__).parent / "_te_evict_out")


def _reset_peak() -> None:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def _generate(pipe: FluxPipeline, label: str, prompt: str) -> dict:
    _reset_peak()
    t0 = time.perf_counter()
    png = pipe.generate_png(prompt=prompt, seed=SEED, steps=STEPS, height=H, width=W)
    gen_s = time.perf_counter() - t0
    peak_mb = mx.get_peak_memory() / (1024 * 1024)
    out_path = OUT_DIR / f"{label}.png"
    out_path.write_bytes(png)
    digest = hashlib.sha256(png).hexdigest()[:16]
    te_is_none = pipe._model.text_encoder is None
    return {
        "label": label,
        "gen_s": gen_s,
        "peak_mb": peak_mb,
        "png_path": str(out_path),
        "sha": digest,
        "te_is_none_after": te_is_none,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Init pipeline (te_4bit=True, evict_text_encoder=True) ===", flush=True)
    t0 = time.perf_counter()
    pipe = FluxPipeline(PipelineConfig(backend="bonsai-ternary-mlx", te_4bit=True, evict_text_encoder=True))
    init_s = time.perf_counter() - t0
    assert pipe._model._evict_text_encoder is True, "eviction must be on"
    assert pipe._model._studio_te_4bit is True, "4-bit marker must be set"
    print(f"  init_s={init_s:.2f}  te_4bit={pipe.config.te_4bit}  evict_te={pipe.config.evict_text_encoder}", flush=True)

    results = []

    print("\n=== Run 1: cold generate prompt A ===", flush=True)
    results.append(_generate(pipe, "cold_A", PROMPT_A))

    print("\n=== Run 2: cache-hit generate prompt A (TE should stay None) ===", flush=True)
    assert pipe._model.text_encoder is None, "TE must be evicted after run 1"
    results.append(_generate(pipe, "hit_A", PROMPT_A))
    assert results[-1]["te_is_none_after"], "TE should remain None through cache hit"

    print("\n=== Run 3: cache-miss generate prompt B (patched 4-bit reload) ===", flush=True)
    assert pipe._model.text_encoder is None, "TE must still be evicted before run 3"
    results.append(_generate(pipe, "miss_B", PROMPT_B))

    print("\n=== Direct timing: load_te_4bit() in isolation ===", flush=True)
    _reset_peak()
    overrides = pipe._model.model_config.text_encoder_overrides
    t0 = time.perf_counter()
    te_probe = load_te_4bit(overrides)
    mx.eval(te_probe.embed_tokens.weight)
    te_load_s = time.perf_counter() - t0
    del te_probe
    gc.collect()
    mx.clear_cache()
    print(f"  load_te_4bit cold = {te_load_s:.3f}s", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"{r['label']:10s}  gen={r['gen_s']:6.2f}s  peak_mb={r['peak_mb']:7.1f}  "
            f"te_none_after={r['te_is_none_after']}  sha={r['sha']}",
            flush=True,
        )
    print(f"\nTE 4-bit cold load wall-time (isolated) = {te_load_s:.3f}s", flush=True)
    print(
        f"peak_mb: cold_A={results[0]['peak_mb']:.0f}  "
        f"hit_A={results[1]['peak_mb']:.0f}  "
        f"miss_B={results[2]['peak_mb']:.0f}",
        flush=True,
    )
    print(
        f"Δ hit_A vs cold_A = {results[1]['peak_mb'] - results[0]['peak_mb']:+.1f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
