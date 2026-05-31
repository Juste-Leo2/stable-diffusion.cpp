"""End-to-end smoke for the GPU/gemlite Klein backend (Phase-5c-2).

Constructs `GpuPipeline`, runs `prewarm()`, then calls `diffusion_forward` to
turn a single prompt into a PIL.Image. Reports per-stage timings, HBM peak,
and writes the result PNG to disk.

This is the first time the gemlite kernels run with the actual Klein diffusion
loop shapes (transformer expects packed image-token grids, e.g. 4096 tokens at
1024x1024). Cold-start autotune cost is expected on the first forward pass —
the cached `gemlite_autotune.json` only covers shapes that the pack-time
warmup forward exercised.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--prompt", default="a serene bonsai tree on a rocky outcrop, dramatic golden-hour lighting, photorealistic")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--output", type=Path, default=Path("/root/bench_out/phase5c2_smoke.png"))
    p.add_argument("--repeat", type=int, default=1,
                   help="How many forwards to run after prewarm (>=2 separates cold-start from warm).")
    args = p.parse_args()

    log = logging.getLogger("smoke_e2e")
    log.info("args: %s", vars(args))

    import torch

    from backend_gpu.diffusion_klein import diffusion_forward
    from backend_gpu.pipeline_gpu import GpuPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("smoke is CUDA-only; run on a CUDA host.")

    pipe = GpuPipeline()
    log.info("constructed GpuPipeline backend=%s device=%s", pipe.backend, pipe.device)

    t0 = time.perf_counter()
    pipe.prewarm()
    prewarm_s = time.perf_counter() - t0
    log.info("prewarm complete in %.1fs (ready=%s)", prewarm_s, pipe.ready)

    # Memory after load.
    static_alloc_mib = torch.cuda.memory_allocated(pipe.device) / 1024 / 1024
    log.info("post-prewarm HBM allocated: %.1f MiB", static_alloc_mib)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    timings: list[float] = []
    peaks_mib: list[float] = []
    for i in range(args.repeat):
        torch.cuda.synchronize(pipe.device)
        torch.cuda.reset_peak_memory_stats(pipe.device)
        t0 = time.perf_counter()
        img = diffusion_forward(
            transformer=pipe._transformer,
            text_encoder=pipe._text_encoder,
            tokenizer=pipe._tokenizer,
            vae=pipe._vae,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_steps=args.steps,
            seed=args.seed,
            max_sequence_length=args.max_sequence_length,
            guidance=args.guidance,
            scheduler=pipe._scheduler,
        )
        torch.cuda.synchronize(pipe.device)
        forward_s = time.perf_counter() - t0
        peak_mib = torch.cuda.max_memory_allocated(pipe.device) / 1024 / 1024
        timings.append(forward_s)
        peaks_mib.append(peak_mib)
        log.info("[forward %d/%d] %.2fs  peak HBM %.1f MiB  output %s",
                 i + 1, args.repeat, forward_s, peak_mib, img.size)

        if i == 0:
            img.save(str(args.output))
            log.info("wrote first-iter image -> %s", args.output)

    print()
    print(f"=== Phase 5c-2 E2E smoke ({args.height}x{args.width} | steps={args.steps} | seed={args.seed}) ===")
    print(f"  prompt: {args.prompt!r}")
    print(f"  prewarm                  : {prewarm_s:8.1f} s")
    print(f"  post-prewarm static HBM  : {static_alloc_mib:8.1f} MiB")
    print()
    for i, (t, peak) in enumerate(zip(timings, peaks_mib)):
        print(f"  forward[{i}]              : {t:8.2f} s   peak HBM {peak:8.1f} MiB")
    if len(timings) >= 2:
        warm = timings[1:]
        print(f"  warm mean (n={len(warm)})           : {sum(warm) / len(warm):8.2f} s")
        print(f"  cold-start overhead      : {timings[0] - (sum(warm) / len(warm)):+8.2f} s")
    print()
    print(f"  output PNG: {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
