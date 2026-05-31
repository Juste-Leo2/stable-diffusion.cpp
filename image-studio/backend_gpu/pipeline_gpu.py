from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

GpuBackend = Literal[
    "bonsai-binary-gemlite",
    "bonsai-ternary-gemlite",
]

GPU_BACKENDS: tuple[GpuBackend, ...] = (
    "bonsai-binary-gemlite",
    "bonsai-ternary-gemlite",
)

DEFAULT_GPU_BACKEND: GpuBackend = "bonsai-binary-gemlite"
DEFAULT_SEED = 0
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512

# Model artifact paths intentionally have no built-in default — different
# deployments (Evan's laptop, Pasha's Colab demo, primeh200s) lay out the
# model dirs in incompatible places. We require an explicit env var (or
# constructor kwarg) so an unset path fails loudly at startup instead of
# silently fetching weights from HuggingFace or hitting a stale absolute
# path that does not exist on the host.
DEFAULT_BINARY_TRANSFORMER_PATH: str | None = None
DEFAULT_TERNARY_TRANSFORMER_PATH: str | None = None
DEFAULT_TRANSFORMER_PATH: str | None = None  # legacy alias
DEFAULT_TEXT_ENCODER_PATH: str | None = None
DEFAULT_VAE_PATH: str | None = None
DEFAULT_TOKENIZER_PATH: str | None = None
DEFAULT_DEVICE = "cuda:0"


def _required_path(
    explicit: str | Path | None,
    env_name: str,
    *,
    fallbacks: tuple[str | Path | None, ...] = (),
    purpose: str,
) -> Path:
    """Resolve an artifact path with clear error reporting.

    Tries (in order): the explicit constructor kwarg, the env var, then any
    extra fallbacks (e.g. a legacy env var). Returns a Path. Raises
    ValueError with a precise hint if none resolve.
    """
    candidate = explicit or os.environ.get(env_name)
    if candidate is None:
        for fb in fallbacks:
            if fb is not None:
                candidate = fb
                break
    if candidate is None:
        raise ValueError(
            f"GpuPipeline {purpose} path is unset. "
            f"Set {env_name}=<absolute path on this host> (or pass the matching "
            f"kwarg to GpuPipeline(...))."
        )
    return Path(candidate)

def _normalize_gpu_backend(raw: str) -> GpuBackend:
    if raw in GPU_BACKENDS:
        return raw  # type: ignore[return-value]
    raise ValueError(f"Unknown GPU backend {raw!r}; expected one of {GPU_BACKENDS}.")

log = logging.getLogger(__name__)


_GEMLITE_LAYER_KEYS = ("W_q", "bias", "scales", "zeros", "metadata", "orig_shape")


def _load_gemlite_layers_from_state(
    model: Any,
    state: dict[str, Any],
    *,
    bits: int,
    group_size: int,
    device: str,
    DType: Any,
    GemLiteLinearTriton: Any,
) -> tuple[int, dict[str, Any]]:
    """Bucket gemlite-layer keys in `state`, replace each `nn.Linear` with a
    `GemLiteLinearTriton` initialized via gemlite's custom per-layer
    `load_state_dict`, then move its tensors to `device`.

    Returns `(n_loaded, remainder_state)` where `remainder_state` excludes the
    consumed gemlite keys (caller loads it via `model.load_state_dict`).

    Why per-layer: pack() registers W_q/scales/zeros/metadata/orig_shape as
    nn.Parameters, so the saved `state_dict.pt` has those keys, but the
    constructor of an empty GemLiteLinearTriton does NOT pre-register the
    Parameter slots — `model.load_state_dict(strict=False)` reports them all
    as `unexpected`. Gemlite ships a custom `GemLiteLinearTriton.load_state_dict`
    that pops these keys and decodes `metadata` into the layer's scalar fields
    (W_nbits, group_size, dtypes, …). We dispatch that per layer.
    """
    import torch
    import torch.nn as nn

    buckets: dict[str, dict[str, torch.Tensor]] = {}
    remainder: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        fqn, _, leaf = k.rpartition(".")
        if leaf in _GEMLITE_LAYER_KEYS and fqn:
            buckets.setdefault(fqn, {})[leaf] = v
        else:
            remainder[k] = v

    n_loaded = 0
    target_device = torch.device(device)
    for fqn, layer_state in buckets.items():
        parent_fqn, _, child_name = fqn.rpartition(".")
        parent = model.get_submodule(parent_fqn) if parent_fqn else model
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            raise RuntimeError(
                f"state_dict has gemlite keys at {fqn} but model has {type(child).__name__}"
            )
        gl = GemLiteLinearTriton(
            W_nbits=bits,
            group_size=group_size,
            in_features=child.in_features,
            out_features=child.out_features,
            input_dtype=DType.FP16,
            output_dtype=DType.FP16,
        )
        gl.load_state_dict(dict(layer_state))
        gl.W_q = gl.W_q.to(target_device)
        gl.scales = gl.scales.to(target_device)
        gl.zeros = gl.zeros.to(target_device)
        if gl.bias is not None:
            gl.bias = gl.bias.to(target_device)
        gl.device = target_device
        setattr(parent, child_name, gl)
        n_loaded += 1
    log.info("loaded %d GemLiteLinearTriton layers from state_dict", n_loaded)
    return n_loaded, remainder


def _null_gemlite_weights(model: Any, GemLiteLinearTriton: Any) -> int:
    """Set `.weight = None` on every `GemLiteLinearTriton` in `model`.

    Why: `Flux2AttnProcessor`'s MuonClip-telemetry path eagerly evaluates
    `attn.to_q.weight` (and friends). After gemlite replacement those modules
    have no real `weight` tensor — but PyTorch still synthesises an empty one
    via `nn.Module.__getattr__` lookup against any registered Parameter slot.
    Forcing `.weight = None` via `object.__setattr__` (bypassing the Module
    parameter machinery) makes the access explicit-None instead of a phantom
    tensor; the telemetry path treats None as "skip".
    """
    nulled = 0
    for m in model.modules():
        if isinstance(m, GemLiteLinearTriton):
            object.__setattr__(m, "weight", None)
            nulled += 1
    log.info("nulled .weight on %d GemLiteLinearTriton modules", nulled)
    return nulled


def _load_gemlite_transformer(path: Path, *, device: str = DEFAULT_DEVICE) -> Any:
    """Load the gemlite-packed Klein-4B transformer onto `device`.

    Reads `state_dict.pt`, `config.json`, `quantization_config.json`,
    `gemlite_autotune.json` from `path`. Restores the global gemlite autotune
    cache via `gemlite.core.load_config(...)`. Calls `set_packing_bitwidth(...)`
    BEFORE patching/loading so kernel selection matches the pack run.

    Post-load:
      1. Cast the whole module to fp16 (matches the gemlite forward stream).
      2. Null the `.weight` attribute on every GemLiteLinearTriton (Phase-2
         carryover; see `_null_gemlite_weights` for the reason).
    """
    if not path.is_dir():
        raise FileNotFoundError(
            f"Gemlite transformer artifact not found at {path} — "
            "run scripts/pack_klein_to_gemlite.py to regenerate."
        )
    state_path = path / "state_dict.pt"
    config_path = path / "config.json"
    qcfg_path = path / "quantization_config.json"
    autotune_path = path / "gemlite_autotune.json"
    for f in (state_path, config_path, qcfg_path, autotune_path):
        if not f.is_file():
            raise FileNotFoundError(f"Gemlite transformer missing {f.name} at {f}")

    with config_path.open() as fh:
        cfg = json.load(fh)
    with qcfg_path.open() as fh:
        qcfg = json.load(fh)
    bits = int(qcfg.get("bits", 1))
    group_size = int(qcfg.get("group_size", 128))
    packing_bw = int(qcfg.get("packing_bitwidth", 8))

    import torch
    from diffusers import Flux2Transformer2DModel
    from gemlite.core import DType, GemLiteLinearTriton, set_packing_bitwidth

    set_packing_bitwidth(packing_bw)
    GemLiteLinearTriton.load_config(str(autotune_path))

    log.info(
        "loading gemlite transformer config: bits=%d gs=%d bw=%d",
        bits, group_size, packing_bw,
    )
    model = Flux2Transformer2DModel.from_config(cfg).to(torch.bfloat16)
    state = torch.load(str(state_path), map_location="cpu")
    _, remainder = _load_gemlite_layers_from_state(
        model, state,
        bits=bits, group_size=group_size, device=device,
        DType=DType, GemLiteLinearTriton=GemLiteLinearTriton,
    )
    missing, unexpected = model.load_state_dict(remainder, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected non-gemlite state_dict keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"missing non-gemlite state_dict keys: {missing[:8]}")

    model = model.to(torch.float16)
    _null_gemlite_weights(model, GemLiteLinearTriton)
    model = model.to(device).eval()
    # Marker consumed by diffusion_klein to pick the activation dtype for the
    # transformer-input casts (fp16 here; bf16 for the vanilla bf16 loader).
    model._inference_dtype = torch.float16  # type: ignore[attr-defined]
    return model


def _load_transformer_for_backend(
    backend: GpuBackend,
    path: Path,
    *,
    device: str = DEFAULT_DEVICE,
) -> Any:
    """Dispatch to the right loader based on the backend's quantization scheme."""
    return _load_gemlite_transformer(path, device=device)


def _load_text_encoder(path: Path, *, device: str = DEFAULT_DEVICE) -> Any:
    """Load the HQQ-4bit Klein text encoder and gemlite-patch it for inference.

    Output is a `Mistral3ForConditionalGeneration` with HQQLinear modules
    converted to gemlite kernels (fp16 stream).
    """
    if not path.is_dir():
        raise FileNotFoundError(
            f"Text encoder artifact not found at {path} — "
            "see scripts/pack_klein_text_encoder_to_gemlite.py."
        )
    import torch
    from gemlite.core import set_packing_bitwidth
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.utils.patching import prepare_for_inference

    set_packing_bitwidth(8)
    model = AutoHQQHFModel.from_quantized(
        str(path),
        compute_dtype=torch.float16,
        device=device,
    )
    prepare_for_inference(model, backend="gemlite")
    return model


def _load_vae(path: Path, *, device: str = DEFAULT_DEVICE) -> Any:
    """Load the bf16 `AutoencoderKLFlux2` from a local snapshot path."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"VAE snapshot not found at {path} — provision via `huggingface-cli download` "
            "or rerun the Phase-4 snapshot script."
        )
    import torch
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    return vae.to(device).eval()


def _load_tokenizer(path_or_repo: str) -> Any:
    """Load Klein's Qwen2TokenizerFast (text-encoder side, plain string content).

    `diffusion_klein._encode_klein_qwen3_prompt` only needs `apply_chat_template`
    and `__call__`, which `AutoTokenizer` provides. The Klein TE artifact ships
    its own `tokenizer/` subdir; default points there. NOT a Pixtral processor —
    that's the FLUX.2-dev (Mistral) path and produces wrong embeds for Klein.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path_or_repo)


def _load_scheduler(transformer_path: Path) -> Any | None:
    """Load `FlowMatchEulerDiscreteScheduler` from the transformer snapshot.

    Klein checkpoints ship a `scheduler/` subfolder; if present we load it so
    the diffusion forward inherits the trained dynamic-shift settings. Returns
    `None` if absent — `diffusion_klein.py` falls back to its FLUX.2 defaults.
    """
    sched_path = transformer_path / "scheduler"
    if not sched_path.is_dir():
        log.info("no scheduler/ subfolder in %s — diffusion_klein will use defaults", transformer_path)
        return None
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(str(transformer_path), subfolder="scheduler")


class GpuPipeline:
    """Server-side GPU pipeline (gemlite/HQQ on H100).

    `prewarm()` loads 5 artifacts: gemlite transformer, HQQ-gemlite text
    encoder, AutoencoderKLFlux2, Qwen2 tokenizer, and an optional scheduler.
    `generate_png` calls `backend_gpu.diffusion_klein.diffusion_forward` and
    encodes the returned PIL.Image as PNG bytes.
    """

    is_remote: ClassVar[bool] = False

    def __init__(
        self,
        *,
        backend: GpuBackend = DEFAULT_GPU_BACKEND,
        transformer_path: str | Path | None = None,
        binary_transformer_path: str | Path | None = None,
        ternary_transformer_path: str | Path | None = None,
        text_encoder_path: str | Path | None = None,
        vae_path: str | Path | None = None,
        tokenizer_path: str | None = None,
        device: str | None = None,
    ) -> None:
        backend = _normalize_gpu_backend(backend)
        self._backend: GpuBackend = backend
        self.last_peak_memory_mb: float | None = None
        self._ready: bool = False
        self._transformer: Any = None
        self._text_encoder: Any = None
        self._vae: Any = None
        self._tokenizer: Any = None
        self._scheduler: Any = None
        # `transformer_path` (and the matching MFLUX_STUDIO_GPU_TRANSFORMER_PATH
        # env) is a legacy alias for the binary path; pre-existing single-backend
        # callers keep working without setting the new BINARY-suffixed name.
        legacy_default_binary = (
            transformer_path
            or os.environ.get("MFLUX_STUDIO_GPU_TRANSFORMER_PATH")
        )
        self._transformer_paths: dict[GpuBackend, Path] = {
            "bonsai-binary-gemlite": _required_path(
                binary_transformer_path,
                "MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH",
                fallbacks=(legacy_default_binary, DEFAULT_BINARY_TRANSFORMER_PATH),
                purpose="binary transformer",
            ),
            "bonsai-ternary-gemlite": _required_path(
                ternary_transformer_path,
                "MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH",
                fallbacks=(DEFAULT_TERNARY_TRANSFORMER_PATH,),
                purpose="ternary transformer",
            ),
        }
        self.text_encoder_path: Path = _required_path(
            text_encoder_path,
            "MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH",
            fallbacks=(DEFAULT_TEXT_ENCODER_PATH,),
            purpose="text encoder",
        )
        self.vae_path: Path = _required_path(
            vae_path,
            "MFLUX_STUDIO_GPU_VAE_PATH",
            fallbacks=(DEFAULT_VAE_PATH,),
            purpose="VAE",
        )
        # tokenizer is loaded by HuggingFace `from_pretrained`, which accepts
        # a string (path or repo id). Keep that surface type as `str`.
        tok_resolved = _required_path(
            tokenizer_path,
            "MFLUX_STUDIO_GPU_TOKENIZER_PATH",
            fallbacks=(DEFAULT_TOKENIZER_PATH,),
            purpose="tokenizer",
        )
        self.tokenizer_path: str = str(tok_resolved)
        self.device: str = device or os.environ.get("MFLUX_STUDIO_GPU_DEVICE", DEFAULT_DEVICE)

    @property
    def backend(self) -> GpuBackend:
        return self._backend

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def transformer_path(self) -> Path:
        return self._transformer_paths[self._backend]

    def ensure_backend(self, *, backend: GpuBackend, model_path: str | None = None) -> None:
        backend = _normalize_gpu_backend(backend)
        if backend == self._backend and model_path is None:
            return
        if model_path is not None:
            self._transformer_paths[backend] = Path(model_path)
        # Different backend → drop the resident transformer and reload on next prewarm/use.
        if backend != self._backend:
            self._backend = backend
            self._transformer = None
            self._scheduler = None
            self._ready = False
            if self._text_encoder is not None and self._vae is not None and self._tokenizer is not None:
                t0 = time.perf_counter()
                self._transformer = _load_transformer_for_backend(
                    self._backend, self.transformer_path, device=self.device,
                )
                log.info(
                    "swapped transformer to %s in %.2fs",
                    self._backend, time.perf_counter() - t0,
                )
                self._scheduler = _load_scheduler(self.transformer_path)
                self._ready = True

    def prewarm(self) -> None:
        """Load all 5 artifacts; mark pipeline ready.

        Per-artifact errors surface verbatim so deploy diagnostics are
        unambiguous. Each step is timed at INFO. Autotune-warmup forward is
        deferred to `diffusion_klein.diffusion_forward`'s first call since the
        diffusion-loop shapes live there.
        """
        log.info(
            "GpuPipeline.prewarm starting backend=%s device=%s "
            "transformer=%s text_encoder=%s vae=%s tokenizer=%s",
            self._backend, self.device, self.transformer_path,
            self.text_encoder_path, self.vae_path, self.tokenizer_path,
        )
        t0 = time.perf_counter()
        self._transformer = _load_transformer_for_backend(
            self._backend, self.transformer_path, device=self.device,
        )
        log.info("loaded transformer (%s) in %.2fs", self._backend, time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._text_encoder = _load_text_encoder(self.text_encoder_path, device=self.device)
        log.info("loaded text encoder in %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._vae = _load_vae(self.vae_path, device=self.device)
        log.info("loaded VAE in %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._tokenizer = _load_tokenizer(self.tokenizer_path)
        log.info("loaded tokenizer in %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._scheduler = _load_scheduler(self.transformer_path)
        log.info("loaded scheduler in %.2fs (present=%s)",
                 time.perf_counter() - t0, self._scheduler is not None)

        self._ready = True
        log.info("GpuPipeline ready: 5 artifacts loaded (scheduler optional)")

    def generate_png(
        self,
        *,
        prompt: str,
        seed: int = DEFAULT_SEED,
        steps: int = DEFAULT_STEPS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        guidance: float = DEFAULT_GUIDANCE,
        tiled_vae: bool | None = None,  # accepted for API parity with MLX; H100 80GiB has no VAE memory pressure
        max_sequence_length: int | None = None,
    ) -> bytes:
        del tiled_vae
        if not self._ready:
            raise RuntimeError("GpuPipeline.prewarm() must be called before generate_png().")
        # Lazy: diffusion_klein imports `diffusers.pipelines.flux2`, which is
        # heavy and unavailable in test environments unless mocked.
        import torch
        from backend_gpu import diffusion_klein

        log.info(
            "generate backend=%s size=%dx%d steps=%d seed=%d guidance=%.2f",
            self._backend, width, height, steps, seed, guidance,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        forward_kwargs: dict[str, Any] = {
            "transformer": self._transformer,
            "text_encoder": self._text_encoder,
            "tokenizer": self._tokenizer,
            "vae": self._vae,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_steps": steps,
            "seed": seed,
            "guidance": guidance,
            "scheduler": self._scheduler,
        }
        if max_sequence_length is not None:
            forward_kwargs["max_sequence_length"] = max_sequence_length

        image = diffusion_klein.diffusion_forward(**forward_kwargs)

        self.last_peak_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


__all__ = [
    "GpuBackend",
    "GPU_BACKENDS",
    "DEFAULT_GPU_BACKEND",
    "DEFAULT_SEED",
    "DEFAULT_STEPS",
    "DEFAULT_GUIDANCE",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "DEFAULT_BINARY_TRANSFORMER_PATH",
    "DEFAULT_TERNARY_TRANSFORMER_PATH",
    "DEFAULT_TRANSFORMER_PATH",
    "DEFAULT_TEXT_ENCODER_PATH",
    "DEFAULT_VAE_PATH",
    "DEFAULT_TOKENIZER_PATH",
    "DEFAULT_DEVICE",
    "GpuPipeline",
    "_load_gemlite_transformer",
    "_load_text_encoder",
    "_load_vae",
    "_load_tokenizer",
    "_load_scheduler",
    "_load_gemlite_layers_from_state",
    "_null_gemlite_weights",
]
