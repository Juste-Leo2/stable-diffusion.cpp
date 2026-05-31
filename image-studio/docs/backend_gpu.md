# `backend_gpu/` — GPU arm

Standalone FastAPI server for the `bonsai-ternary-gemlite` backend. Runs on a CUDA host; `backend.pipeline.RemoteGpuPipeline` POSTs to it over HTTP.

Pipeline: gemlite transformer + HQQ-int4 text encoder + bf16 VAE on a single H100. 4-step Klein defaults.

This file is a pointer to the existing detailed docs in the component itself:

- [`backend_gpu/README.md`](../backend_gpu/README.md) — layout, artifacts, env vars, smoke tests, autotune cache, run instructions.

## Quick reference

```
backend_gpu/
    server.py              # FastAPI: /healthz, /generate, /generate/compare
    pipeline_gpu.py        # GpuPipeline: 5-artifact prewarm + generate_png
    diffusion_klein.py     # Klein/Qwen3 text→image forward
    scripts/smoke_*.py     # local CUDA + remote round-trip smokes
    tests/                 # loader + server unit tests
```

## API contract

Identical JSON shape to `backend/`'s `/generate`, plus a Bearer auth header. `/healthz` is unauthenticated.

```sh
MFLUX_STUDIO_GPU_TOKEN=devtoken \
uvicorn backend_gpu.server:app --host 0.0.0.0 --port 8801

curl -s http://localhost:8801/healthz
# {"status":"ok"}

curl -s -o out.png \
  -H 'Authorization: Bearer devtoken' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "...", "seed": 42, "steps": 4, "guidance": 1.0, "backend": "bonsai-ternary-gemlite", "height": 512, "width": 512}' \
  http://localhost:8801/generate
```

## Required env

| Var | Default | Purpose |
| --- | --- | --- |
| `MFLUX_STUDIO_GPU_TOKEN` | (required) | Bearer; server refuses to start without it |
| `MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH` | (unset) | ternary transformer pack path |
| `MFLUX_STUDIO_GPU_TRANSFORMER_PATH` | (legacy alias for the ternary path) | retained for backward compatibility |
| `MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH` | `/root/models/klein-4b-text-encoder-hqq-4bit-gemlite/` | HQQ-int4 TE pack |
| `MFLUX_STUDIO_GPU_VAE_PATH` | `/root/models/klein-4b-vae-bf16/` | bf16 VAE snapshot |
| `MFLUX_STUDIO_GPU_DEVICE` | `cuda:0` | target device |

Full table + artifact regen recipes in [`backend_gpu/README.md`](../backend_gpu/README.md).

## Tests

```sh
.venv/bin/python -m pytest backend_gpu/tests/
```

The unit tests stub out CUDA so they pass on any host. Real GPU smoke is `backend_gpu/scripts/smoke_e2e.py` (runs prewarm + a single forward) and `scripts/smoke_remote.py` (exercises `RemoteGpuPipeline` against a deployed server).
