"""A/B: bf16 TE vs 4bit TE, same seed + prompt, both on bonsai-ternary-mlx backend.

Run:
  .venv/bin/python -m backend.tests.smoke_te_4bit_ab
"""
from __future__ import annotations

import gc
import hashlib
import os
import time
from pathlib import Path

import mlx.core as mx

from backend.pipeline import FluxPipeline, PipelineConfig

PROMPT = (
    "A serene Scandinavian woman in her 40s, soft window light, natural skin texture, "
    "shallow depth of field, 50mm portrait lens, neutral linen backdrop."
)
SEED = 42
STEPS = 4
H = W = 1024
OUT_DIR = Path(os.environ.get("BONSAI_SMOKE_OUT_DIR") or Path(__file__).parent / "_ab_out")


def run(label: str, *, te_4bit: bool) -> dict:
    print(f"\n=== {label} (te_4bit={te_4bit}) ===", flush=True)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    t0 = time.perf_counter()
    pipe = FluxPipeline(PipelineConfig(backend="bonsai-ternary-mlx", te_4bit=te_4bit))
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    png = pipe.generate_png(prompt=PROMPT, seed=SEED, steps=STEPS, height=H, width=W)
    gen_s = time.perf_counter() - t1

    peak_mb = mx.get_peak_memory() / (1024 * 1024)
    out_path = OUT_DIR / f"{label}.png"
    out_path.write_bytes(png)
    digest = hashlib.sha256(png).hexdigest()[:16]

    # Dispose pipeline between runs so peak-memory resets cleanly.
    del pipe
    gc.collect()
    mx.clear_cache()

    return {
        "label": label,
        "te_4bit": te_4bit,
        "load_s": load_s,
        "gen_s": gen_s,
        "peak_mb": peak_mb,
        "png_path": str(out_path),
        "sha256_16": digest,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        run("te_bf16", te_4bit=False),
        run("te_4bit", te_4bit=True),
    ]
    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"{r['label']:10s}  load={r['load_s']:6.2f}s  gen={r['gen_s']:6.2f}s  "
            f"peak_mb={r['peak_mb']:7.1f}  sha={r['sha256_16']}  -> {r['png_path']}",
            flush=True,
        )
    peak_delta = results[0]["peak_mb"] - results[1]["peak_mb"]
    print(f"\npeak_mb delta (bf16 - 4bit) = {peak_delta:+.1f} MB", flush=True)


if __name__ == "__main__":
    main()
