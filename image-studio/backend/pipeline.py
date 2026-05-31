from __future__ import annotations

import gc
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import ClassVar, Literal

import httpx
import mlx.core as mx
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from backend import eviction  # noqa: F401  (import installs per-component reload patch)
from backend.text_encoder_4bit import load_te_4bit

Backend = Literal[
    "bonsai-ternary-mlx",
    "bonsai-binary-mlx",
    "bonsai-binary-gemlite",
    "bonsai-ternary-gemlite",
]
BackendKind = Literal["mlx", "gemlite"]
ModelFamily = Literal["bonsai-binary", "bonsai-ternary"]
TiledMode = Literal["auto", "on", "off"]

LOCAL_BACKENDS: tuple[Backend, ...] = (
    "bonsai-ternary-mlx",
    "bonsai-binary-mlx",
)
REMOTE_BACKENDS: tuple[Backend, ...] = (
    "bonsai-binary-gemlite",
    "bonsai-ternary-gemlite",
)
BACKENDS: tuple[Backend, ...] = LOCAL_BACKENDS + REMOTE_BACKENDS

# Family order mirrors the frontend MODEL_FAMILIES rendering order.
MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "bonsai-binary",
    "bonsai-ternary",
)

BACKEND_TO_KIND: dict[Backend, BackendKind] = {
    "bonsai-ternary-mlx": "mlx",
    "bonsai-binary-mlx": "mlx",
    "bonsai-binary-gemlite": "gemlite",
    "bonsai-ternary-gemlite": "gemlite",
}
BACKEND_TO_FAMILY: dict[Backend, ModelFamily] = {
    "bonsai-ternary-mlx": "bonsai-ternary",
    "bonsai-binary-mlx": "bonsai-binary",
    "bonsai-binary-gemlite": "bonsai-binary",
    "bonsai-ternary-gemlite": "bonsai-ternary",
}

DEFAULT_BAKED_MODEL_PATH = "/tmp/bonsai-checkpoints/v1/"
DEFAULT_BAKED_BINARY_MODEL_PATH: str | None = None
DEFAULT_BACKEND: Backend = "bonsai-ternary-mlx"
DEFAULT_TILED_MODE: TiledMode = "auto"
DEFAULT_EVICT_TEXT_ENCODER = True
DEFAULT_LAZY_COMPONENTS = False
DEFAULT_EVICT_TRANSFORMER = False
DEFAULT_EVICT_VAE = False
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BUCKETED_SEQ_LEN = False
DEFAULT_TE_4BIT = True
DEFAULT_GPU_HOST: str | None = None
DEFAULT_GPU_TOKEN: str | None = None
DEFAULT_GPU_TIMEOUT_SECONDS = 600
DEFAULT_GPU_CONNECT_TIMEOUT_SECONDS = 30
DEFAULT_SEED = 42
DEFAULT_STEPS = 4
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GUIDANCE = 1.0


def _auto_tile_threshold() -> int:
    return 2 * TilingConfig().vae_decode_tile_size


def _normalize_backend_id(raw: str) -> Backend:
    if raw in BACKENDS:
        return raw  # type: ignore[return-value]
    raise ValueError(f"Unknown backend {raw!r}; expected one of {BACKENDS}.")


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false/1/0/yes/no/on/off), got {raw!r}.")


def _parse_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    backend: Backend = DEFAULT_BACKEND
    baked_model_path: str = DEFAULT_BAKED_MODEL_PATH
    baked_binary_model_path: str | None = DEFAULT_BAKED_BINARY_MODEL_PATH
    tiled_mode: TiledMode = DEFAULT_TILED_MODE
    evict_text_encoder: bool = DEFAULT_EVICT_TEXT_ENCODER
    lazy_components: bool = DEFAULT_LAZY_COMPONENTS
    evict_transformer: bool = DEFAULT_EVICT_TRANSFORMER
    evict_vae: bool = DEFAULT_EVICT_VAE
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    bucketed_seq_len: bool = DEFAULT_BUCKETED_SEQ_LEN
    te_4bit: bool = DEFAULT_TE_4BIT
    gpu_host: str | None = DEFAULT_GPU_HOST
    gpu_token: str | None = DEFAULT_GPU_TOKEN
    gpu_timeout_seconds: int = DEFAULT_GPU_TIMEOUT_SECONDS
    gpu_connect_timeout_seconds: int = DEFAULT_GPU_CONNECT_TIMEOUT_SECONDS

    @staticmethod
    def from_env() -> "PipelineConfig":
        backend_raw = os.getenv("MFLUX_STUDIO_DEFAULT_BACKEND")
        if backend_raw is not None:
            backend: Backend = _normalize_backend_id(backend_raw)
        else:
            backend = DEFAULT_BACKEND

        baked_model_path = os.getenv("MFLUX_STUDIO_BAKED_MODEL_PATH")
        if baked_model_path is None:
            legacy = os.getenv("MFLUX_STUDIO_MODEL_PATH")
            baked_model_path = legacy if legacy is not None else DEFAULT_BAKED_MODEL_PATH
        if not os.path.isabs(baked_model_path):
            raise ValueError(
                "MFLUX_STUDIO_BAKED_MODEL_PATH must be an absolute path, "
                f"got {baked_model_path!r}."
            )
        baked_binary_model_path = os.getenv(
            "MFLUX_STUDIO_BAKED_BINARY_MODEL_PATH", DEFAULT_BAKED_BINARY_MODEL_PATH
        )
        if baked_binary_model_path is not None and not os.path.isabs(baked_binary_model_path):
            raise ValueError(
                "MFLUX_STUDIO_BAKED_BINARY_MODEL_PATH must be an absolute path when set, "
                f"got {baked_binary_model_path!r}."
            )

        tiled_mode = os.getenv("MFLUX_STUDIO_TILED_VAE", DEFAULT_TILED_MODE)
        if tiled_mode not in {"auto", "on", "off"}:
            raise ValueError(
                "MFLUX_STUDIO_TILED_VAE must be 'auto', 'on', or 'off', "
                f"got {tiled_mode!r}."
            )
        evict_text_encoder = _parse_bool_env(
            "MFLUX_STUDIO_EVICT_TEXT_ENCODER", DEFAULT_EVICT_TEXT_ENCODER
        )
        lazy_components = _parse_bool_env(
            "MFLUX_STUDIO_LAZY_COMPONENTS", DEFAULT_LAZY_COMPONENTS
        )
        evict_transformer = _parse_bool_env(
            "MFLUX_STUDIO_EVICT_TRANSFORMER", DEFAULT_EVICT_TRANSFORMER
        )
        evict_vae = _parse_bool_env("MFLUX_STUDIO_EVICT_VAE", DEFAULT_EVICT_VAE)
        max_sequence_length = _parse_int_env(
            "MFLUX_STUDIO_MAX_SEQUENCE_LENGTH", DEFAULT_MAX_SEQUENCE_LENGTH
        )
        bucketed_seq_len = _parse_bool_env(
            "MFLUX_STUDIO_BUCKETED_SEQ_LEN", DEFAULT_BUCKETED_SEQ_LEN
        )
        te_4bit = _parse_bool_env("MFLUX_STUDIO_TE_4BIT", DEFAULT_TE_4BIT)
        gpu_host = os.getenv("MFLUX_STUDIO_GPU_HOST", DEFAULT_GPU_HOST)
        gpu_token = os.getenv("MFLUX_STUDIO_GPU_TOKEN", DEFAULT_GPU_TOKEN)
        gpu_timeout_seconds = _parse_int_env(
            "MFLUX_STUDIO_GPU_TIMEOUT_SECONDS", DEFAULT_GPU_TIMEOUT_SECONDS
        )
        gpu_connect_timeout_seconds = _parse_int_env(
            "MFLUX_STUDIO_GPU_CONNECT_TIMEOUT_SECONDS", DEFAULT_GPU_CONNECT_TIMEOUT_SECONDS
        )
        if backend in REMOTE_BACKENDS:
            if gpu_host is None:
                raise ValueError(
                    f"Backend {backend!r} requires MFLUX_STUDIO_GPU_HOST to be set."
                )
            if gpu_token is None:
                raise ValueError(
                    f"Backend {backend!r} requires MFLUX_STUDIO_GPU_TOKEN to be set."
                )
        return PipelineConfig(
            backend=backend,
            baked_model_path=baked_model_path,
            baked_binary_model_path=baked_binary_model_path,
            tiled_mode=tiled_mode,
            evict_text_encoder=evict_text_encoder,
            lazy_components=lazy_components,
            evict_transformer=evict_transformer,
            evict_vae=evict_vae,
            max_sequence_length=max_sequence_length,
            bucketed_seq_len=bucketed_seq_len,
            te_4bit=te_4bit,
            gpu_host=gpu_host,
            gpu_token=gpu_token,
            gpu_timeout_seconds=gpu_timeout_seconds,
            gpu_connect_timeout_seconds=gpu_connect_timeout_seconds,
        )


def _resolve_tiling_config(
    *,
    request_override: bool | None,
    server_default: TiledMode,
    height: int,
    width: int,
) -> TilingConfig | None:
    if request_override is True:
        return TilingConfig()
    if request_override is False:
        return None
    if server_default == "on":
        return TilingConfig()
    if server_default == "off":
        return None
    return TilingConfig() if max(height, width) >= _auto_tile_threshold() else None


def _build_model(
    *,
    backend: Backend,
    model_path: str | None,
    config: PipelineConfig,
) -> Flux2Klein:
    if backend in REMOTE_BACKENDS:
        raise ValueError(
            f"Backend {backend!r} is remote-only; build via RemoteGpuPipeline, "
            "not FluxPipeline._build_model."
        )
    if backend == "bonsai-ternary-mlx":
        return Flux2Klein(
            model_path=model_path,
            use_klein_fast_transformer=True,
            klein_fast_precision="2bit",
            vae_variant="small",
            evict_text_encoder=config.evict_text_encoder,
            lazy_components=config.lazy_components,
            bucketed_seq_len=config.bucketed_seq_len,
        )
    if backend == "bonsai-binary-mlx":
        return Flux2Klein(
            model_path=model_path,
            use_klein_fast_transformer=True,
            klein_fast_precision="1bit",
            vae_variant="small",
            evict_text_encoder=config.evict_text_encoder,
            lazy_components=config.lazy_components,
            bucketed_seq_len=config.bucketed_seq_len,
        )
    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}.")


def _default_model_path_for(backend: Backend, config: PipelineConfig) -> str | None:
    if backend == "bonsai-ternary-mlx":
        return config.baked_model_path
    if backend == "bonsai-binary-mlx":
        return config.baked_binary_model_path
    return None


class FluxPipeline:
    is_remote: ClassVar[bool] = False

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._backend: Backend | None = None
        self._model_path: str | None = None
        self._model: Flux2Klein | None = None
        self.last_swap_seconds: float | None = None
        self.last_peak_memory_mb: float | None = None
        self._load(backend=config.backend, model_path=_default_model_path_for(config.backend, config))

    @property
    def backend(self) -> Backend:
        assert self._backend is not None
        return self._backend

    @property
    def model_path(self) -> str | None:
        return self._model_path

    def _load(self, *, backend: Backend, model_path: str | None) -> None:
        start = time.perf_counter()
        self._model = _build_model(backend=backend, model_path=model_path, config=self.config)
        if self.config.te_4bit:
            # Replace the freshly loaded bf16 TE with the pre-quantized 4-bit Qwen3
            # from the local model dir (or fall back to mlx-community HF if not
            # bundled). The marker routes Klein's cache-miss reload path (patched
            # in text_encoder_4bit.py) back to load_te_4bit so eviction stays a win.
            # Pass model_path explicitly so load_te_4bit's local-first probe sees
            # the bundled text_encoder-mlx-4bit/ subdir — without it, this call
            # always defaulted to the HF download path.
            self._model._studio_te_4bit = True
            overrides = self._model.model_config.text_encoder_overrides
            self._model.text_encoder = load_te_4bit(overrides, model_path=model_path)
            gc.collect()
            mx.clear_cache()
        self._backend = backend
        self._model_path = model_path
        self.last_swap_seconds = time.perf_counter() - start
        log.info(
            "Loaded backend=%s model_path=%s evict_te=%s lazy=%s evict_tx=%s evict_vae=%s "
            "max_seq=%d bucketed=%s te_4bit=%s in %.3fs",
            backend,
            model_path,
            self.config.evict_text_encoder,
            self.config.lazy_components,
            self.config.evict_transformer,
            self.config.evict_vae,
            self.config.max_sequence_length,
            self.config.bucketed_seq_len,
            self.config.te_4bit,
            self.last_swap_seconds,
        )

    def ensure_backend(self, *, backend: Backend, model_path: str | None) -> None:
        resolved = model_path if model_path is not None else _default_model_path_for(backend, self.config)
        if backend == self._backend and resolved == self._model_path:
            return
        log.info(
            "Hot-swapping backend %s(%s) -> %s(%s)",
            self._backend,
            self._model_path,
            backend,
            resolved,
        )
        self._model = None
        gc.collect()
        mx.clear_cache()
        self._load(backend=backend, model_path=resolved)

    def generate_png(
        self,
        *,
        prompt: str,
        seed: int = DEFAULT_SEED,
        steps: int = DEFAULT_STEPS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        guidance: float = DEFAULT_GUIDANCE,
        tiled_vae: bool | None = None,
        max_sequence_length: int | None = None,
    ) -> bytes:
        assert self._model is not None
        tiling = _resolve_tiling_config(
            request_override=tiled_vae,
            server_default=self.config.tiled_mode,
            height=height,
            width=width,
        )
        self._model.tiling_config = tiling
        log.info(
            "generate backend=%s size=%dx%d tiled=%s tile_size=%s",
            self._backend,
            width,
            height,
            tiling is not None,
            tiling.vae_decode_tile_size if tiling is not None else None,
        )
        effective_max_seq = (
            max_sequence_length if max_sequence_length is not None else self.config.max_sequence_length
        )
        mx.reset_peak_memory()
        generated = self._model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            max_sequence_length=effective_max_seq,
            evict_transformer=self.config.evict_transformer,
        )
        # VAE-only eviction (tx_evict=False, vae_evict=True). When tx_evict is on
        # Klein has already bundled-evicted VAE via evict_transformer_and_vae.
        if self.config.evict_vae and not self.config.evict_transformer:
            self._model.vae = None
            gc.collect()
            mx.clear_cache()
        self.last_peak_memory_mb = mx.get_peak_memory() / (1024 * 1024)
        output = io.BytesIO()
        try:
            generated.image.save(output, format="PNG")
            return output.getvalue()
        finally:
            output.close()
            del generated
            gc.collect()


class RemoteGpuPipeline:
    """HTTP-proxy pipeline for the GPU arm (gemlite/HQQ on remote CUDA host).

    Mirrors FluxPipeline's surface so the FastAPI handler is byte-identical:
    same `ensure_backend(backend, model_path)` + `generate_png(...)` returning
    PNG bytes. Reuses one `httpx.Client` for HTTP/1.1 keep-alive across calls.
    """

    is_remote: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        if config.gpu_host is None:
            raise ValueError(
                "RemoteGpuPipeline requires MFLUX_STUDIO_GPU_HOST."
            )
        if config.gpu_token is None:
            raise ValueError(
                "RemoteGpuPipeline requires MFLUX_STUDIO_GPU_TOKEN."
            )
        if config.backend not in REMOTE_BACKENDS:
            raise ValueError(
                f"RemoteGpuPipeline cannot host local backend {config.backend!r}; "
                f"expected one of {REMOTE_BACKENDS}."
            )
        self.config = config
        self._backend: Backend = config.backend
        self._client = httpx.Client(
            base_url=config.gpu_host,
            headers={"Authorization": f"Bearer {config.gpu_token}"},
            timeout=httpx.Timeout(
                connect=float(config.gpu_connect_timeout_seconds),
                read=float(config.gpu_timeout_seconds),
                write=float(config.gpu_connect_timeout_seconds),
                pool=float(config.gpu_connect_timeout_seconds),
            ),
        )
        self.last_peak_memory_mb: float | None = None
        self.last_swap_seconds: float | None = 0.0
        log.info(
            "RemoteGpuPipeline ready host=%s backend=%s read_timeout=%ds",
            config.gpu_host,
            self._backend,
            config.gpu_timeout_seconds,
        )

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def model_path(self) -> str | None:
        return None

    def ensure_backend(self, *, backend: Backend, model_path: str | None) -> None:
        # Why: remote arm holds no in-process model, so "swap" is a label change.
        if backend not in REMOTE_BACKENDS:
            raise ValueError(
                f"Server started in remote-GPU mode; backend {backend!r} is local-only. "
                f"Restart with MFLUX_STUDIO_DEFAULT_BACKEND set to one of {LOCAL_BACKENDS}."
            )
        self._backend = backend

    def generate_png(
        self,
        *,
        prompt: str,
        seed: int = DEFAULT_SEED,
        steps: int = DEFAULT_STEPS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        guidance: float = DEFAULT_GUIDANCE,
        tiled_vae: bool | None = None,
        max_sequence_length: int | None = None,
    ) -> bytes:
        body: dict[str, object] = {
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "height": height,
            "width": width,
            "guidance": guidance,
            "backend": self._backend,
        }
        if tiled_vae is not None:
            body["tiled_vae"] = tiled_vae
        if max_sequence_length is not None:
            body["max_sequence_length"] = max_sequence_length
        log.info("remote-gpu generate backend=%s size=%dx%d", self._backend, width, height)
        response = self._client.post("/generate", json=body)
        if response.status_code != 200:
            detail: str
            try:
                detail = str(response.json().get("detail", response.text))
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"Remote GPU /generate returned {response.status_code}: {detail}"
            )
        peak_header = response.headers.get("X-Peak-Memory-MB")
        try:
            self.last_peak_memory_mb = float(peak_header) if peak_header is not None else None
        except ValueError:
            self.last_peak_memory_mb = None
        return response.content

    def close(self) -> None:
        self._client.close()


def make_pipeline(config: PipelineConfig) -> "FluxPipeline | RemoteGpuPipeline":
    """Factory: pick FluxPipeline (local MLX) or RemoteGpuPipeline (HTTP) by backend prefix."""
    if config.backend in REMOTE_BACKENDS:
        return RemoteGpuPipeline(config)
    return FluxPipeline(config)
