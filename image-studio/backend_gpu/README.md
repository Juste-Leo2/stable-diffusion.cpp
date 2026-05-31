# backend_gpu

GPU-side companion to `backend/`. Runs on a CUDA host and serves the
`bonsai-ternary-gemlite` backend over the same `/generate` +
`/generate/compare` JSON contract that
`backend.pipeline.RemoteGpuPipeline` POSTs to.

Pipeline: gemlite transformer + HQQ-int4 text encoder + bf16 VAE on a single
H100. 4-step Klein defaults (`steps=4`, `guidance=1.0`).

## Layout

```
backend_gpu/
    __init__.py
    pipeline_gpu.py        # GpuPipeline: 5-artifact prewarm + generate_png
    diffusion_klein.py     # Klein/Qwen3 text→image forward (4-step)
    server.py              # FastAPI app: /healthz, /generate, /generate/compare
    pyproject.toml         # deps manifest (gemlite, hqq, transformers, diffusers, torch)
    scripts/
        smoke_e2e.py       # local CUDA smoke (prewarm + diffusion_forward)
        smoke_remote.py    # exercise RemoteGpuPipeline against a deployed server
    tests/
        test_loaders.py    # unit-level coverage of every loader + generate_png contract
        test_server.py     # FastAPI auth, routing, schema, healthz
```

## Artifacts (defaults match `pipeline_gpu.py`)

| Path | Format | Size |
| --- | --- | --- |
| `<ternary-transformer-path>` | gemlite-packed ternary transformer | ~1.1 GiB |
| `<text-encoder-path>` | HQQ-packed text encoder | ~2.7 GiB |
| `<vae-path>` | bf16 VAE | ~161 MiB |

The text encoder artifact bundles its own `tokenizer/` subdir
(Qwen2TokenizerFast). The transformer artifact does NOT carry an HF
`scheduler/` subfolder; `_build_default_scheduler()` in `diffusion_klein.py`
provides the FLUX.2 dynamic-shift defaults (verified against MLX —
`base_shift=0.5, max_shift=1.15, base/max_image_seq_len=256/4096`).

## Environment

| Var | Required | Default | Purpose |
| --- | --- | --- | --- |
| `MFLUX_STUDIO_GPU_TOKEN` | yes | — | Bearer token; server refuses to start if unset. |
| `MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH` | no | (unset) | Packed ternary transformer. |
| `MFLUX_STUDIO_GPU_TRANSFORMER_PATH` | no | (legacy alias for the ternary path) | Retained for backward compatibility. |
| `MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH` | no | (unset) | HQQ-int4 text encoder. |
| `MFLUX_STUDIO_GPU_VAE_PATH` | no | (unset) | bf16 VAE snapshot. |
| `MFLUX_STUDIO_GPU_TOKENIZER_PATH` | no | `<TE>/tokenizer/` | Qwen2TokenizerFast directory. |
| `MFLUX_STUDIO_GPU_DEVICE` | no | `cuda:0` | Target device. |

## Run

```bash
# Install (assumes torch + gemlite + hqq stack already present in venv)
uv sync   # or: pip install -e .

# Launch
MFLUX_STUDIO_GPU_TOKEN=devtoken \
uvicorn backend_gpu.server:app --host 0.0.0.0 --port 8801
```

Boot does the full prewarm: 5 artifacts loaded onto `cuda:0`, gemlite
autotune cache restored from `gemlite_autotune.json` in the transformer
artifact dir. First `/generate` call may pay a one-time Triton
compile cost for any image-size / batch shape outside the cached set
(Klein training shapes are covered).

## Smoke tests

```bash
# healthz is unauthenticated
curl -s http://localhost:8801/healthz
# {"status":"ok"}

# /generate requires Bearer; returns image/png
curl -s -o out.png -D - \
    -H "Authorization: Bearer devtoken" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"a cat","steps":4,"width":512,"height":512,"guidance":1.0}' \
    http://localhost:8801/generate

# unauthenticated → 401
curl -s -i -H "Content-Type: application/json" \
    -d '{"prompt":"x"}' \
    http://localhost:8801/generate
```

Or run the bundled smoke directly on the GPU host (skips the HTTP layer):
```bash
.venv/bin/python -m backend_gpu.scripts.smoke_e2e --prompt "a bonsai"
```

## Tests

```bash
.venv/bin/python -m unittest backend_gpu.tests -v
```

Tests stub torch/gemlite/hqq/transformers via `sys.modules` so the suite
runs on macOS without a CUDA stack.

## Reference perf (H100, single device, fp16 stream)

- 1024² × 4-step warm: **1.45 s** wall, **6.4 GiB** peak HBM
- 512² × 4-step: smoke target (sub-second per Phase-6 numbers)

