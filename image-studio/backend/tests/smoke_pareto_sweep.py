"""Pareto sweep: (peak MB, wall seconds) across TE / transformer / VAE eviction knobs.

Matrix (bonsai-ternary-mlx, 1024², 4 steps, same seed+prompt):

  cfg_id | te_4bit | evict_te | evict_tx | evict_vae | notes
  bl_bf16     F       T         F          F          bf16 baseline (flag-off)
  pin_4bit    T       F         F          F          pinned 4-bit TE (prior)
  ff          T       T         F          F          current default
  ft          T       T         F          T          VAE-only post-decode eviction
  tf          T       T         T          F          Klein bundles both; reload both next gen
  tt          T       T         T          T          Klein bundles both (equiv to tf)

Per config: fresh pipeline, one cold gen, one warm gen (same prompt, cache hit).
Records peak_mb and gen_s each. Pipeline destroyed between configs so peak
memory is isolated to that config.

Run:
  .venv/bin/python -m backend.tests.smoke_pareto_sweep
"""
from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass

import mlx.core as mx

from backend.pipeline import FluxPipeline, PipelineConfig

PROMPT = (
    "A serene Scandinavian woman in her 40s, soft window light, natural skin texture, "
    "shallow depth of field, 50mm portrait lens, neutral linen backdrop."
)
SEED = 42
STEPS = 4
H = W = 1024


@dataclass(frozen=True)
class Cfg:
    cfg_id: str
    te_4bit: bool
    evict_te: bool
    evict_tx: bool
    evict_vae: bool
    notes: str


CONFIGS = [
    Cfg("bl_bf16",  False, True,  False, False, "bf16 baseline (flag-off)"),
    Cfg("pin_4bit", True,  False, False, False, "pinned 4-bit TE"),
    Cfg("ff",       True,  True,  False, False, "current default"),
    Cfg("ft",       True,  True,  False, True,  "VAE-only eviction"),
    Cfg("tf",       True,  True,  True,  False, "Klein bundles tx+vae"),
    Cfg("tt",       True,  True,  True,  True,  "Klein bundles tx+vae"),
]


def _reset_peak() -> None:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def _gen(pipe: FluxPipeline) -> tuple[float, float, str]:
    _reset_peak()
    t0 = time.perf_counter()
    png = pipe.generate_png(prompt=PROMPT, seed=SEED, steps=STEPS, height=H, width=W)
    gen_s = time.perf_counter() - t0
    peak_mb = mx.get_peak_memory() / (1024 * 1024)
    sha = hashlib.sha256(png).hexdigest()[:12]
    return gen_s, peak_mb, sha


def _run_cfg(cfg: Cfg) -> dict:
    print(f"\n=== {cfg.cfg_id}: {cfg.notes}  (te4bit={cfg.te_4bit} "
          f"evict_te={cfg.evict_te} evict_tx={cfg.evict_tx} evict_vae={cfg.evict_vae}) ===",
          flush=True)
    gc.collect()
    mx.clear_cache()

    t0 = time.perf_counter()
    pipe = FluxPipeline(PipelineConfig(
        backend="bonsai-ternary-mlx",
        te_4bit=cfg.te_4bit,
        evict_text_encoder=cfg.evict_te,
        evict_transformer=cfg.evict_tx,
        evict_vae=cfg.evict_vae,
    ))
    init_s = time.perf_counter() - t0

    cold_s, cold_peak, cold_sha = _gen(pipe)
    warm_s, warm_peak, warm_sha = _gen(pipe)
    byte_identical = (cold_sha == warm_sha)

    del pipe
    gc.collect()
    mx.clear_cache()

    return {
        "cfg": cfg,
        "init_s": init_s,
        "cold_s": cold_s,
        "cold_peak_mb": cold_peak,
        "warm_s": warm_s,
        "warm_peak_mb": warm_peak,
        "byte_identical_cold_warm": byte_identical,
        "sha": warm_sha,
    }


def main() -> None:
    results = [_run_cfg(c) for c in CONFIGS]

    print("\n=== SUMMARY ===", flush=True)
    print(
        f"{'cfg_id':10s} {'te4':>3s} {'evTE':>4s} {'evTX':>4s} {'evVA':>4s}  "
        f"{'init_s':>7s} {'cold_s':>7s} {'cold_MB':>8s} {'warm_s':>7s} {'warm_MB':>8s}  identical",
        flush=True,
    )
    for r in results:
        c = r["cfg"]
        print(
            f"{c.cfg_id:10s} {int(c.te_4bit):>3d} {int(c.evict_te):>4d} "
            f"{int(c.evict_tx):>4d} {int(c.evict_vae):>4d}  "
            f"{r['init_s']:>7.2f} {r['cold_s']:>7.2f} {r['cold_peak_mb']:>8.1f} "
            f"{r['warm_s']:>7.2f} {r['warm_peak_mb']:>8.1f}  {r['byte_identical_cold_warm']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
