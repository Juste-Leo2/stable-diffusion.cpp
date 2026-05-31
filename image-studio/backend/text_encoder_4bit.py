from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

TE_4BIT_REPO = "mlx-community/Qwen3-4B-4bit"
_QUANT_BITS = 4
_QUANT_GROUP_SIZE = 64


def load_te_4bit(text_encoder_overrides: dict, model_path: str | None = None) -> Qwen3TextEncoder:
    # Prefer the bundled text_encoder-mlx-4bit/ under model_path when present.
    # The Bonsai HF repos (prism-ml/bonsai-image-{ternary,binary}-4B-mlx-*bit)
    # ship the exact mlx-community Qwen3-4B-4bit weights as a subdir, so using
    # them locally avoids a ~3 GB HF download every fresh install AND keeps
    # the demo self-contained (no implicit HF dependency for the TE).
    local_dir = Path(model_path) / "text_encoder-mlx-4bit" if model_path else None
    if local_dir and (local_dir / "model.safetensors").is_file():
        root = local_dir
    else:
        root = Path(
            snapshot_download(
                repo_id=TE_4BIT_REPO,
                allow_patterns=["*.safetensors", "config.json"],
            )
        )
    raw = mx.load(str(root / "model.safetensors"))
    stripped = {k[len("model."):]: v for k, v in raw.items() if k.startswith("model.")}
    nested = tree_unflatten(list(stripped.items()))

    te = Qwen3TextEncoder(**text_encoder_overrides)
    nn.quantize(
        te,
        class_predicate=lambda _, m: hasattr(m, "to_quantized"),
        bits=_QUANT_BITS,
        group_size=_QUANT_GROUP_SIZE,
    )
    te.update(nested)
    return te


# Patch Klein's cache-miss reload to honor the 4-bit flag per-instance. Without
# this, eviction + next cache-miss would reinstate bf16 and blow the memory win.
#
# Also short-circuits when pointed at a "slim" packed-mflux checkpoint — i.e. a
# root that has `transformer-packed-mflux/` but no bf16 `transformer/`. mflux's
# stock `_load_weights` would crash trying to read a missing transformer dir
# before the per-instance `_studio_te_4bit` flag is ever set (the flag is set in
# `FluxPipeline._load`, *after* `Flux2Klein()` returns — too late for the init
# path that runs `reload_text_encoder` from inside the constructor). Slim
# checkpoints are exactly what `prism-ml/bonsai-image-*-4B-mlx-*bit` ships, and
# the Bonsai-image-demo download lands here, so this auto-detection lets a fresh
# install generate without any extra knobs.
_ORIGINAL_RELOAD_TEXT_ENCODER = Flux2Initializer.reload_text_encoder


def _is_slim_checkpoint(model_path: str | None) -> bool:
    if not model_path:
        return False
    root = Path(model_path)
    return (root / "transformer-packed-mflux").exists() and not (root / "transformer").exists()


def _reload_text_encoder_with_4bit(model) -> None:
    if getattr(model, "_studio_te_4bit", False) or _is_slim_checkpoint(getattr(model, "_model_path", None)):
        model.text_encoder = load_te_4bit(
            model.model_config.text_encoder_overrides,
            model_path=getattr(model, "_model_path", None),
        )
        return
    _ORIGINAL_RELOAD_TEXT_ENCODER(model)


Flux2Initializer.reload_text_encoder = staticmethod(_reload_text_encoder_with_4bit)
