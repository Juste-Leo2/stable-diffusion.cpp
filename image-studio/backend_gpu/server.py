from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from backend_gpu.pipeline_gpu import (
    DEFAULT_GPU_BACKEND,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    GPU_BACKENDS,
    GpuBackend,
    GpuPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _required_token() -> str:
    token = os.environ.get("MFLUX_STUDIO_GPU_TOKEN")
    if not token:
        raise RuntimeError(
            "MFLUX_STUDIO_GPU_TOKEN must be set; the GPU server refuses to start unauthenticated."
        )
    return token


_bearer_scheme = HTTPBearer(auto_error=False)


def _verify_bearer(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    expected: str = request.app.state.token
    if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    token = _required_token()
    backend_env = os.environ.get("MFLUX_STUDIO_GPU_DEFAULT_BACKEND", DEFAULT_GPU_BACKEND)
    from backend_gpu.pipeline_gpu import _normalize_gpu_backend
    try:
        backend_env = _normalize_gpu_backend(backend_env)
    except ValueError as exc:
        raise RuntimeError(
            f"MFLUX_STUDIO_GPU_DEFAULT_BACKEND={backend_env!r} not in {GPU_BACKENDS}."
        ) from exc
    pipeline = GpuPipeline(backend=backend_env)
    pipeline.prewarm()
    app.state.pipeline = pipeline
    app.state.token = token
    log.info("backend_gpu ready backend=%s", pipeline.backend)
    yield


app = FastAPI(lifespan=lifespan)
# Why: Bearer auth gates every protected route, so opening CORS is safe and lets
# the laptop frontend hit this directly during development without a proxy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    seed: int = DEFAULT_SEED
    steps: int = Field(default=DEFAULT_STEPS, ge=1)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0)
    backend: GpuBackend = DEFAULT_GPU_BACKEND
    height: int = Field(default=DEFAULT_HEIGHT, ge=16)
    width: int = Field(default=DEFAULT_WIDTH, ge=16)
    tiled_vae: bool | None = Field(default=None)
    max_sequence_length: int | None = Field(default=None, ge=1)


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    seed: int = DEFAULT_SEED
    steps: int = Field(default=DEFAULT_STEPS, ge=1)
    guidance: float = Field(default=DEFAULT_GUIDANCE, ge=0.0)
    height: int = Field(default=DEFAULT_HEIGHT, ge=16)
    width: int = Field(default=DEFAULT_WIDTH, ge=16)
    backends: list[GpuBackend] = Field(default_factory=lambda: list(GPU_BACKENDS))
    tiled_vae: bool | None = Field(default=None)
    max_sequence_length: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_backends(self) -> "CompareRequest":
        if not self.backends:
            raise ValueError("backends must contain at least one entry.")
        unknown = [b for b in self.backends if b not in GPU_BACKENDS]
        if unknown:
            raise ValueError(f"Unknown backend(s): {unknown}; expected subset of {list(GPU_BACKENDS)}.")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("backends must not contain duplicates.")
        return self


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/generate",
    response_class=Response,
    dependencies=[Depends(_verify_bearer)],
    responses={
        200: {
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
            "description": "Generated PNG image.",
        }
    },
)
async def generate(request: GenerateRequest) -> Response:
    pipeline: GpuPipeline = app.state.pipeline
    try:
        pipeline.ensure_backend(backend=request.backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gen_start = time.perf_counter()
    image_bytes = pipeline.generate_png(
        prompt=request.prompt,
        seed=request.seed,
        steps=request.steps,
        height=request.height,
        width=request.width,
        guidance=request.guidance,
        tiled_vae=request.tiled_vae,
        max_sequence_length=request.max_sequence_length,
    )
    wall_seconds = time.perf_counter() - gen_start
    headers = {"X-Wall-Seconds": f"{wall_seconds:.3f}"}
    if pipeline.last_peak_memory_mb is not None:
        headers["X-Peak-Memory-MB"] = f"{pipeline.last_peak_memory_mb:.1f}"
    return Response(content=image_bytes, media_type="image/png", headers=headers)


@app.post("/generate/compare", dependencies=[Depends(_verify_bearer)])
async def generate_compare(request: CompareRequest) -> dict:
    pipeline: GpuPipeline = app.state.pipeline
    results: list[dict] = []
    for target_backend in request.backends:
        swap_start = time.perf_counter()
        try:
            pipeline.ensure_backend(backend=target_backend)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        swap_seconds = time.perf_counter() - swap_start

        gen_start = time.perf_counter()
        image_bytes = pipeline.generate_png(
            prompt=request.prompt,
            seed=request.seed,
            steps=request.steps,
            height=request.height,
            width=request.width,
            guidance=request.guidance,
            tiled_vae=request.tiled_vae,
            max_sequence_length=request.max_sequence_length,
        )
        wall_seconds = time.perf_counter() - gen_start

        results.append(
            {
                "backend": target_backend,
                "png_b64": base64.b64encode(image_bytes).decode("ascii"),
                "wall_seconds": wall_seconds,
                "swap_seconds": swap_seconds,
            }
        )
    return {"results": results}


__all__ = [
    "app",
    "GenerateRequest",
    "CompareRequest",
    "generate",
    "generate_compare",
    "healthz",
]
