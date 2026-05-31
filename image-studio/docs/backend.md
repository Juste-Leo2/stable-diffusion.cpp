# `backend/` — FastAPI Mac backend

Local FastAPI server that fronts three FLUX.2 Klein generation arms (two resident on Mac MLX, one proxied to a remote GPU). Talks to `frontend/` (Next.js) and the iOS app (Bonsai's remote backends, when an on-device path isn't selected).

## Layout

```
backend/
    server.py              # FastAPI app, /generate, /backends, /generate/compare
    pipeline.py            # Klein/mflux pipeline construction, backend selection, env-var read
    pipeline_remote_gpu.py # RemoteGpuPipeline — POSTs /generate to backend_gpu over HTTP
    tests/
        test_backends_endpoint.py   # /backends contract + GPU probe caching
        test_ensure_backend.py      # backend swap + locking semantics
        test_generate_compare.py    # /generate/compare side-by-side
        smoke_*                     # adhoc smoke scripts; not part of pytest collection
```

## Run

Requires Python 3.13 + `uv`.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/uvicorn backend.server:app --port 8000
```

## API

### `POST /generate`

```json
{
  "prompt": "a small red cube on a table",
  "seed": 42,
  "steps": 4,
  "guidance": 1.0,
  "backend": "bonsai-ternary-mlx",
  "height": 512,
  "width": 512
}
```

Returns `image/png` bytes.

### `GET /backends`

Returns `{available: [...], default: "...", gpu: {available: bool, reason: str}}`. `reason` is one of `force_disabled`, `no_gpu_host`, `no_gpu_token`, `healthz_failed:<status>`, `healthz_unreachable`. Cached 30 s.

### `POST /generate/compare`

Multi-arm side-by-side. Same request shape with `backends: ["bonsai-ternary-mlx", "bfl-klein-bf16"]`. Returns one entry per arm.

## The three backends

| ID | What | Where |
| --- | --- | --- |
| `bonsai-ternary-mlx` | Bonsai (ternary) Klein on MLX | local Mac |
| `bfl-klein-bf16` | upstream mflux on Klein bf16 (`full` VAE) | local Mac |
| `bonsai-ternary-gemlite` | gemlite + HQQ TE + bf16 VAE | remote (`backend_gpu/`) |

The two Mac arms share a single `Flux2Klein` slot; on switch the server evicts + rebuilds (`asyncio.Lock` serializes). The remote arm holds no in-process state; calls fan out over HTTP. See [`docs/backend_gpu.md`](backend_gpu.md).

## Checkpoint layout

The Mac MLX arms each consume a Klein root directory:

```
<klein-checkpoint-root>/
    transformer/                       # bf16 dense weights
    transformer-packed-mflux/          # uint32 packed weights + scales
    text_encoder/                      # Klein/Qwen3 bf16 TE
    tokenizer/                         # Qwen3 tokenizer
    vae/                               # FLUX.2 VAE
    scheduler/                         # FLUX.2 flow-match scheduler config
    model_index.json
    LICENSE.md                         # Apache 2.0 from upstream Klein
```

`bfl-klein-bf16` ignores the packed dir and resolves Klein from HF.

## Env vars

See [`README.md`](../README.md#env-vars). Most-load-bearing:

| Var | Required | Purpose |
| --- | --- | --- |
| `MFLUX_STUDIO_BAKED_MODEL_PATH` | for `bonsai-ternary-mlx` | absolute path to the ternary checkpoint root |
| `MFLUX_STUDIO_STOCK_MODEL_PATH` | optional, for `bfl-klein-bf16` | absolute path to a local Klein snapshot; falls back to HF default |
| `MFLUX_STUDIO_DEFAULT_BACKEND` | no | initial backend at boot |
| `MFLUX_STUDIO_GPU_HOST` + `MFLUX_STUDIO_GPU_TOKEN` | for `bonsai-ternary-gemlite` | base URL + bearer token to `backend_gpu/` |
| `MFLUX_STUDIO_FORCE_DISABLE_GPU` | no | hides the GPU arm regardless of probe |

## Tests

```sh
.venv/bin/python -m pytest backend/tests/
```

Smoke scripts (`smoke_*.py`) are not collected by pytest; run them directly when comparing baked checkpoints or evicting strategies.
