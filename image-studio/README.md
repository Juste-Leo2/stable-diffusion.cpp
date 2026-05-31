# image-studio

Image generation studio for FLUX.2 Klein on-device. Three components:

- **`backend/`** — FastAPI server fronting mflux backends on a Mac.
- **`backend_gpu/`** — separate FastAPI server for the GPU arm; deployed to a CUDA host.
- **`frontend/`** — Next.js web client that talks to `backend/`.

Per-component setup, API contracts, and deployment lives under [`docs/`](docs/). High-level pointers:

- [`docs/backend.md`](docs/backend.md) — `/generate` API + the backend selectors.
- [`docs/backend_gpu.md`](docs/backend_gpu.md) — gemlite GPU arm + deploy notes.
- [`docs/frontend.md`](docs/frontend.md) — Next.js client.

Contributor workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md). License: Apache 2.0 (see [`LICENSE`](LICENSE), [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md)).

## Quickstart (Next.js web client)

Requires Node 20+ and `npm`.

```sh
cd frontend
npm install
npm run dev          # http://localhost:3000
```

The frontend talks to a `backend/` FastAPI server at `http://localhost:8000` by default (`NEXT_PUBLIC_BACKEND_URL` to override). Start the backend first (next section) or it'll show "GPU unavailable" / "Unknown backend" until reachable.

Build for prod: `npm run build && npm start`. See [`docs/frontend.md`](docs/frontend.md) for env vars + deploy notes (Vercel-ready).

## Quickstart (local Mac backend)

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/uvicorn backend.server:app --port 8000
```

Set `MFLUX_STUDIO_BAKED_MODEL_PATH` to an absolute path to a FLUX.2 Klein ternary checkpoint root (containing `transformer-packed-mflux/`, `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`). See [`docs/backend.md`](docs/backend.md) for the checkpoint layout.

## Backends

`POST /generate` accepts a `backend` field with three values:

- `bonsai-ternary-mlx` — Bonsai (ternary) Klein on MLX.
- `bfl-klein-bf16` — BFL FLUX.2 Klein-4B at bf16.
- `bonsai-ternary-gemlite` — remote GPU arm (ternary).

The `bonsai-ternary-gemlite` arm is served by [`backend_gpu/`](docs/backend_gpu.md) over HTTP and is off by default; it gates on `MFLUX_STUDIO_GPU_HOST` + `MFLUX_STUDIO_GPU_TOKEN` being set.

Exactly one Mac-side backend is resident at a time. On a backend change the server evicts the transformer+VAE and rebuilds (`Flux2Klein`); expect a one-shot swap cost. Concurrent requests serialize behind an `asyncio.Lock`. The remote `bonsai-ternary-gemlite` arm holds no in-process model so its swap is a label change and concurrent calls fan out over HTTP.

`GET /backends` returns `{available, default, gpu: {available, reason}}` where `reason` is one of `force_disabled`, `no_gpu_host`, `no_gpu_token`, `healthz_failed:<status>`, or `healthz_unreachable`. Probe result is cached for 30 s. Pass `?force_disable=1` to hide the GPU arm for a single response.

## Env vars

- `MFLUX_STUDIO_DEFAULT_BACKEND` — one of the three backend values above (default `bonsai-ternary-mlx`). `/backends` falls through to `bonsai-ternary-mlx` when the configured default is unavailable.
- `MFLUX_STUDIO_BAKED_MODEL_PATH` — absolute dir for the ternary MLX arm. Override to point at your local checkpoint.
- `MFLUX_STUDIO_STOCK_MODEL_PATH` — absolute dir for `bfl-klein-bf16`; when unset mflux resolves its own HF default.
- `MFLUX_STUDIO_GPU_HOST` / `MFLUX_STUDIO_GPU_TOKEN` — base URL + bearer token for the remote `bonsai-ternary-gemlite` arm. Both required to expose it via `/backends`.
- `MFLUX_STUDIO_FORCE_DISABLE_GPU` — `1`/`true` hides the GPU arm regardless of probe. Read once at module load — restart the server to flip it.
- `MFLUX_STUDIO_BACKENDS_PROBE_TTL_SECONDS` — override the 30 s `/backends` cache (mostly tests).

Deprecated: `MFLUX_STUDIO_PRECISION` (`bf16`) and the request-body `precision` field still work but emit `DeprecationWarning` and will be removed next release.
