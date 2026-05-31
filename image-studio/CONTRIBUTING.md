# Contributing

## Dev setup

Requires Python 3.13 + [`uv`](https://docs.astral.sh/uv/) + Xcode 16+ (for the apple/ target).

### mflux + mlx git refs

`pyproject.toml`'s `[tool.uv.sources]` pins `mflux` and `mlx` to specific revisions under `PrismML-Eng/`. `uv sync` resolves both automatically — no sibling checkout required:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

Bump the `mlx` rev or switch `mflux` to a sha pin by editing `[tool.uv.sources]` if you need to lock further.

Per-component setup lives in [`docs/`](docs/).

## Running tests

Backend (FastAPI):
```sh
.venv/bin/python -m pytest backend/tests/
```

Backend GPU (deployed to a CUDA host; tests exercise the loaders + server stubs):
```sh
.venv/bin/python -m pytest backend_gpu/tests/
```

Apple (Mac Catalyst, fastest local loop):
```sh
xcodebuild test -project apple/Bonsai.xcodeproj -scheme Bonsai \
  -destination 'platform=macOS,variant=Mac Catalyst,arch=arm64'
```

iPhone parity, performance, and end-to-end tests are gated on a checkpoint payload at `BONSAI_TEST_CHECKPOINT_ROOT` (or the documented fallback). They `XCTSkip` cleanly when the payload is missing.

## Commit messages

Short, lower-case, imperative. Match `git log` style. No bodies unless the change is genuinely subtle. No co-author trailers, no AI-tool attribution.

```
apple: fused norm+RoPE megakernel + Klein parity tests
backend: pixel-stat diag for checkpoint sweep
docs: add LICENSE Apache 2.0
```

## Pull requests

- One concern per PR. If a change naturally splits into setup + behavior, split it.
- Don't auto-merge. Wait for review.
- Update [`docs/`](docs/) and [`README.md`](README.md) when behavior or env vars change.

## Code style

- Swift: follow the existing pattern in `apple/Bonsai/` — 4-space indent, no semicolons, terse `let` over `var`. Run the test suite under Mac Catalyst before pushing.
- Python: 4-space indent, type-hinted public APIs. The `backend/` and `backend_gpu/` modules are kept readable over clever; prefer explicit imports and short functions.
- TypeScript / Next.js (`frontend/`): `npm run lint` before pushing.
