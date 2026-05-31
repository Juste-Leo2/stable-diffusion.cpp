"""Per-component reload for Klein's transformer + VAE.

Klein's stock `load_transformer_and_vae` rebuilds both unless both are already
resident. That makes VAE-only eviction useless: the next gen reloads the
transformer too. We patch the loader so a call with one component resident and
the other None rebuilds only the missing one.

The patch also short-circuits the slim packed-mflux checkpoint case (only
`transformer-packed-mflux/`, no bf16 `transformer/`) when
`use_klein_fast_transformer = True`. The stock loader's first move is
`_load_weights(model_path)`, which scans the full Klein layout and crashes
when `transformer/` is missing. For klein-fast + slim we build the VAE from
the small-decoder weights and the transformer from the packed artifact
directly, skipping the failing scan entirely.

Eviction side stays stock — Klein's `evict_transformer_and_vae` still bundles
both when `evict_transformer=True` is passed to `generate_image`. The pipeline
handles the VAE-only case by setting `model.vae = None` post-decode itself.
"""
from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx

from mflux.models.flux2.flux2_initializer import (
    FULL_DECODER_CHANNELS,
    SMALL_DECODER_CHANNELS,
    Flux2Initializer,
)
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition


_ORIGINAL_LOAD = Flux2Initializer.load_transformer_and_vae


def _is_slim_checkpoint(model_path: str | None) -> bool:
    if not model_path:
        return False
    root = Path(model_path)
    return (root / "transformer-packed-mflux").exists() and not (root / "transformer").exists()


def _load_slim_klein_fast(model) -> None:
    """Build VAE (small-decoder weights) + klein-fast transformer for slim ckpts."""
    if model.vae is None:
        decoder_channels = (
            SMALL_DECODER_CHANNELS if model._vae_variant == "small" else FULL_DECODER_CHANNELS
        )
        model.vae = Flux2VAE(decoder_block_out_channels=decoder_channels)
        # Small-decoder weights are the only VAE source we trust for slim ckpts;
        # full VAE would need vae/ on disk, which slim does ship — fall through if so.
        if model._vae_variant == "small":
            vae_weights = Flux2Initializer._load_small_decoder_weights()
        else:
            # Slim ckpts include vae/; let the stock per-component loader pull it.
            from mflux.models.common.resolution.path_resolution import PathResolution
            from mflux.models.common.weights.loading.weight_loader import WeightLoader

            root = PathResolution.resolve(
                path=model._model_path,
                patterns=Flux2KleinWeightDefinition.get_download_patterns(),
            )
            vae_component = next(
                c for c in Flux2KleinWeightDefinition.get_components() if c.name == "vae"
            )
            vae_weights, _, _ = WeightLoader._load_component(root, vae_component)
        loaded = LoadedWeights(
            components={"vae": vae_weights},
            meta_data=MetaData(quantization_level=None, mflux_version=None),
        )
        WeightApplier.apply_and_quantize(
            weights=loaded,
            quantize_arg=model._quantize_arg,
            weight_definition=Flux2KleinWeightDefinition,
            models={"vae": model.vae},
        )

    if model.transformer is None:
        Flux2Initializer._load_klein_fast_transformer_weights(
            model, model._model_path, precision=model._klein_fast_precision
        )
        Flux2Initializer._apply_lora(model, model._lora_paths_arg, model._lora_scales_arg)

    gc.collect()
    mx.clear_cache()


def _load_transformer_and_vae_per_component(model) -> None:
    tx_resident = model.transformer is not None
    vae_resident = model.vae is not None
    if tx_resident and vae_resident:
        return

    if bool(getattr(model, "_use_klein_fast_transformer", False)) and _is_slim_checkpoint(
        getattr(model, "_model_path", None)
    ):
        _load_slim_klein_fast(model)
        return

    if not tx_resident and not vae_resident:
        _ORIGINAL_LOAD(model)
        return

    weights = Flux2Initializer._load_weights(model._model_path)

    if not vae_resident:
        decoder_channels = (
            SMALL_DECODER_CHANNELS if model._vae_variant == "small" else FULL_DECODER_CHANNELS
        )
        model.vae = Flux2VAE(decoder_block_out_channels=decoder_channels)
        if model._vae_variant == "small":
            weights.components["vae"] = Flux2Initializer._load_small_decoder_weights()
        WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=model._quantize_arg,
            weight_definition=Flux2KleinWeightDefinition,
            models={"vae": model.vae},
        )

    if not tx_resident:
        if model._use_klein_fast_transformer:
            Flux2Initializer._load_klein_fast_transformer_weights(
                model, model._model_path, precision=model._klein_fast_precision
            )
        else:
            model.transformer = Flux2Transformer(**model.model_config.transformer_overrides)
            WeightApplier.apply_and_quantize(
                weights=weights,
                quantize_arg=model._quantize_arg,
                weight_definition=Flux2KleinWeightDefinition,
                models={"transformer": model.transformer},
            )
        Flux2Initializer._apply_lora(model, model._lora_paths_arg, model._lora_scales_arg)

    gc.collect()
    mx.clear_cache()


Flux2Initializer.load_transformer_and_vae = staticmethod(_load_transformer_and_vae_per_component)
