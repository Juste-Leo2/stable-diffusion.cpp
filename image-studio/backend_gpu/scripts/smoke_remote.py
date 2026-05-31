#!/usr/bin/env python3
"""Phase 5b laptop-side smoke: round-trip RemoteGpuPipeline against a live
remote backend_gpu service.

Reads MFLUX_STUDIO_GPU_HOST + MFLUX_STUDIO_GPU_TOKEN from the environment.
Confirms:
  1. valid PNG bytes come back from /generate
  2. last_peak_memory_mb is parsed from the X-Peak-Memory-MB header
  3. /generate/compare round-trips (via direct httpx call, since
     RemoteGpuPipeline.generate_png hits /generate only)
  4. wrong token surfaces a clean RuntimeError

Run:
    MFLUX_STUDIO_GPU_HOST=http://127.0.0.1:8801 \
    MFLUX_STUDIO_GPU_TOKEN=<dev-token> \
    .venv/bin/python backend_gpu/scripts/smoke_remote.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

# Allow running from anywhere — pin sys.path to image-studio/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.pipeline import PipelineConfig, RemoteGpuPipeline  # noqa: E402

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _config(token_override: str | None = None) -> PipelineConfig:
    cfg = PipelineConfig.from_env()
    if token_override is not None:
        cfg = dataclasses.replace(cfg, gpu_token=token_override)
    return cfg


def smoke_happy_path() -> None:
    pipe = RemoteGpuPipeline(_config())
    try:
        png = pipe.generate_png(prompt="a cat", steps=4, width=128, height=128, seed=7)
    finally:
        pipe.close()
    assert png.startswith(_PNG_MAGIC), f"expected PNG magic, got {png[:16].hex()}"
    print(f"[OK] /generate happy path: {len(png)} bytes, peak_mb={pipe.last_peak_memory_mb}")


def smoke_wrong_token() -> None:
    pipe = RemoteGpuPipeline(_config(token_override="bogus-token"))
    try:
        try:
            pipe.generate_png(prompt="x", steps=1, width=64, height=64)
        except RuntimeError as exc:
            assert "401" in str(exc), f"expected 401 in error, got: {exc}"
            print(f"[OK] wrong token surfaces RuntimeError: {exc}")
            return
        raise AssertionError("expected RuntimeError on wrong token")
    finally:
        pipe.close()


def smoke_compare() -> None:
    # /generate/compare is exercised via raw httpx since RemoteGpuPipeline only
    # wraps /generate. Confirms the second route plus auth too.
    import base64

    import httpx

    cfg = _config()
    with httpx.Client(
        base_url=cfg.gpu_host,
        headers={"Authorization": f"Bearer {cfg.gpu_token}"},
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
    ) as client:
        response = client.post(
            "/generate/compare",
            json={"prompt": "x", "backends": ["bonsai-ternary-gemlite"], "width": 64, "height": 64, "steps": 1},
        )
        response.raise_for_status()
        body = response.json()
    assert len(body["results"]) == 1
    result = body["results"][0]
    decoded = base64.b64decode(result["png_b64"])
    assert decoded.startswith(_PNG_MAGIC)
    print(
        f"[OK] /generate/compare round-trip: backend={result['backend']} "
        f"wall={result['wall_seconds']:.4f}s swap={result['swap_seconds']:.4f}s png={len(decoded)}B"
    )


def main() -> int:
    if not os.environ.get("MFLUX_STUDIO_GPU_HOST") or not os.environ.get("MFLUX_STUDIO_GPU_TOKEN"):
        print("MFLUX_STUDIO_GPU_HOST and MFLUX_STUDIO_GPU_TOKEN must be set.", file=sys.stderr)
        return 2
    smoke_happy_path()
    smoke_compare()
    smoke_wrong_token()
    print("\nAll Phase 5b smoke checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
