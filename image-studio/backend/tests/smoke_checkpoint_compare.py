"""Side-by-side checkpoint comparison on bonsai-ternary-mlx.

Same prompt + seed + steps + size against two checkpoint paths. Saves PNGs
and computes pixel statistics: a noise output has approximately uniform
per-channel mean ~127 and high std; a real image has skewed distributions.

Run:
  .venv/bin/python -m backend.tests.smoke_checkpoint_compare
"""
from __future__ import annotations

import gc
import io
import os
import time

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.pipeline import FluxPipeline, PipelineConfig

PROMPT = (
    "A serene Scandinavian woman in her 40s, soft window light, natural skin texture, "
    "shallow depth of field, 50mm portrait lens, neutral linen backdrop."
)
SEED = 42
STEPS = 4
H = W = 1024

CHECKPOINT_A = os.environ.get("BONSAI_SMOKE_CHECKPOINT_A", "/tmp/bonsai-checkpoints/a")
CHECKPOINT_B = os.environ.get("BONSAI_SMOKE_CHECKPOINT_B", "/tmp/bonsai-checkpoints/b")

OUT_DIR = os.environ.get("BONSAI_SMOKE_OUT_DIR", "/tmp/bonsai_smoke_diag")


def _stats(png_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return {
        "shape": arr.shape,
        "mean": tuple(float(arr.mean(axis=(0, 1))[i]) for i in range(3)),
        "std": tuple(float(arr.std(axis=(0, 1))[i]) for i in range(3)),
        "min": tuple(float(arr.min(axis=(0, 1))[i]) for i in range(3)),
        "max": tuple(float(arr.max(axis=(0, 1))[i]) for i in range(3)),
    }


def _run(label: str, model_path: str) -> None:
    print(f"\n=== {label}: {model_path} ===", flush=True)
    gc.collect()
    mx.clear_cache()

    t0 = time.perf_counter()
    pipe = FluxPipeline(PipelineConfig(
        backend="bonsai-ternary-mlx",
        baked_model_path=model_path,
        te_4bit=True,
        evict_text_encoder=True,
        evict_transformer=False,
        evict_vae=False,
    ))
    init_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    png = pipe.generate_png(prompt=PROMPT, seed=SEED, steps=STEPS, height=H, width=W)
    gen_s = time.perf_counter() - t0

    out = f"{OUT_DIR}/{label}.png"
    with open(out, "wb") as f:
        f.write(png)

    s = _stats(png)
    print(
        f"init_s={init_s:.2f} gen_s={gen_s:.2f} -> {out}\n"
        f"  shape={s['shape']}\n"
        f"  mean R/G/B = ({s['mean'][0]:.1f}, {s['mean'][1]:.1f}, {s['mean'][2]:.1f})\n"
        f"  std  R/G/B = ({s['std'][0]:.1f}, {s['std'][1]:.1f}, {s['std'][2]:.1f})\n"
        f"  min  R/G/B = ({s['min'][0]:.0f}, {s['min'][1]:.0f}, {s['min'][2]:.0f})\n"
        f"  max  R/G/B = ({s['max'][0]:.0f}, {s['max'][1]:.0f}, {s['max'][2]:.0f})",
        flush=True,
    )

    del pipe
    gc.collect()
    mx.clear_cache()


def main() -> None:
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    _run("checkpoint_a", CHECKPOINT_A)
    _run("checkpoint_b", CHECKPOINT_B)


if __name__ == "__main__":
    main()
