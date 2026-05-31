"""Klein 4B inference forward (text -> PIL.Image) for the GPU/gemlite backend.

This is the glue between text encoder, transformer, and VAE. It mirrors the
upstream FLUX.2 inference path but strips out img2img, condition images, real
CFG, T-LoRA, and callback machinery that the GPU backend doesn't need.

Reused upstream helpers (from diffusers):
    Flux2Pipeline._prepare_text_ids                    # 4-axis RoPE for text tokens
    Flux2Pipeline._prepare_latent_ids                  # 4-axis RoPE for latent grid
    Flux2Pipeline._pack_latents                        # (B,C,H,W) -> (B,H*W,C)
    Flux2Pipeline._unpack_latents_with_ids             # scatter packed -> (B,C,H,W)
    Flux2Pipeline._unpatchify_latents                  # 2x2 unpack: (B,128,H,W) -> (B,32,2H,2W)
    retrieve_timesteps                                 # scheduler.set_timesteps wrapper

Empirical mu (resolution-dependent shift) is computed locally via
`_mflux_empirical_mu` to match the mflux + iOS Swift port byte-for-byte;
diffusers' built-in linear shift is NOT used.

Text encode is inlined locally (`_encode_klein_qwen3_prompt`) rather than reusing
`Flux2Pipeline._get_mistral_3_small_prompt_embeds` — that helper is the FLUX.2-dev
Mistral path (layers 10/20/30, Pixtral system message, `add_generation_prompt=False`)
and produces off-distribution embeddings for Klein/Qwen3.

Klein has `guidance_embeds=False`, so the inference path is single-forward
with a guidance scalar — no two-pass CFG. Default 4 steps.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps


log = logging.getLogger(__name__)


def _mflux_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Resolution-dependent shift from mflux. Mirrors
    mflux.models.common.schedulers.flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler._compute_empirical_mu
    (and the iOS port at apple/Bonsai/Pipeline/FlowMatchEulerScheduler.swift::computeEmpiricalMu).
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


# Klein 4B: guidance_embeds=False, so the guidance scalar has no read path
# inside the transformer (Flux2TimestepGuidanceEmbeddings.forward short-circuits
# on `self.guidance_embedder is None`). Defaults: guidance=1.0, steps=4.
DEFAULT_GUIDANCE = 1.0
DEFAULT_NUM_STEPS = 4

# Klein/Qwen3 text-encoder layers stacked into the joint embedding.
KLEIN_OUTPUT_LAYERS = (9, 18, 27)


@torch.no_grad()
def _encode_klein_qwen3_prompt(
    text_encoder: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_sequence_length: int,
) -> torch.Tensor:
    """Klein/Qwen3 prompt encode.

    No system message, plain string content, `add_generation_prompt=True`,
    `enable_thinking=False`, hidden states stacked from layers (9, 18, 27).
    Returns `(1, max_sequence_length, 3*hidden_dim)` in the encoder's dtype.
    """
    device = text_encoder.device
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )
    out = torch.stack([output.hidden_states[k] for k in KLEIN_OUTPUT_LAYERS], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)


def _build_default_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Default FLUX.2 flow-matching scheduler.

    The Klein model ships a `scheduler/scheduler_config.json` on HF; if the
    loader (coder8) wires `_scheduler` into GpuPipeline, prefer that. This
    fallback uses diffusers' defaults plus FLUX.2-style dynamic shift so
    the mflux empirical mu can flow through `set_timesteps(..., mu=mu)`.
    """
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )


@torch.no_grad()
def diffusion_forward(
    transformer: nn.Module,
    text_encoder: nn.Module,
    tokenizer,
    vae: nn.Module,
    prompt: str,
    *,
    height: int,
    width: int,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int = 0,
    max_sequence_length: int = 512,
    guidance: float = DEFAULT_GUIDANCE,
    scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
) -> Image.Image:
    """Klein 4B text-to-image forward.

    Args:
        transformer: gemlite-patched Flux2Transformer2DModel (fp16 internal stream).
        text_encoder: Mistral-3 text encoder (bf16).
        tokenizer: PixtralProcessor / AutoProcessor for the text encoder.
        vae: AutoencoderKLFlux2 (bf16 native).
        prompt: text prompt.
        height/width: output image size in pixels (must be multiple of 32).
        num_steps: flow-matching denoising steps (default 4).
        seed: torch CPU generator seed for the initial noise.
        max_sequence_length: max text tokens (default 512).
        guidance: scalar guidance fed to the transformer (no CFG; single forward).
        scheduler: optional pre-loaded scheduler; defaults to FLUX.2 dynamic-shift.

    Returns: PIL.Image.Image, RGB, (height, width).
    """
    transformer_device = next(transformer.parameters()).device
    vae_device = next(vae.parameters()).device

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"height={height} and width={width} must be multiples of 32 (vae_scale_factor*2).")

    if scheduler is None:
        scheduler = _build_default_scheduler()

    # 1. Text encode (Klein/Qwen3, bf16 stream). The upstream Mistral helper
    #    would silently use the wrong layers + Pixtral system message and
    #    produce off-distribution embeds for Klein.
    log.info("encoding prompt (max_seq=%d, klein/qwen3)", max_sequence_length)
    prompt_embeds = _encode_klein_qwen3_prompt(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompt,
        max_sequence_length=max_sequence_length,
    )  # (1, max_seq, 3*hidden_dim)
    text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(transformer_device)  # (1, max_seq, 4)

    # Activation dtype: gemlite int1/int2 kernels require fp16. The bf16 arm
    # (`bfl-klein-bf16-gemlite`) wants native bf16 throughout. Loaders mark the
    # transformer with `_inference_dtype`; default to fp16 if unset for backwards
    # compatibility with callers that build a model outside this package.
    activation_dtype = getattr(transformer, "_inference_dtype", torch.float16)
    prompt_embeds_t = prompt_embeds.to(device=transformer_device, dtype=activation_dtype)

    # 2. Prepare initial latents in packed (B, image_seq_len, C) form.
    # vae_scale_factor=8 (8x VAE compression) and an additional 2x2 patch pack -> 16x total.
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))  # latent H pre-patchify (= H/8)
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))   # latent W pre-patchify
    in_channels_latents = transformer.config.in_channels // 4  # = 32 for Klein (in_channels=128)

    # Sample noise on CPU with explicit generator for determinism, then move.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_shape = (1, in_channels_latents * 4, h_lat // 2, w_lat // 2)  # (1, 128, H/16, W/16)
    latents_4d = torch.randn(noise_shape, generator=gen, dtype=torch.float32)
    latents_4d = latents_4d.to(device=transformer_device, dtype=activation_dtype)

    latent_ids = Flux2Pipeline._prepare_latent_ids(latents_4d).to(transformer_device)  # (1, image_seq_len, 4)
    latents = Flux2Pipeline._pack_latents(latents_4d)  # (1, image_seq_len, 128) fp16
    image_seq_len = latents.shape[1]
    log.info("latents: 4D=%s -> packed=%s  image_seq_len=%d",
             tuple(latents_4d.shape), tuple(latents.shape), image_seq_len)

    # 3. Schedule timesteps with FLUX.2 empirical-mu shift.
    mu = _mflux_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    timesteps, num_steps_eff = retrieve_timesteps(
        scheduler, num_steps, transformer_device, sigmas=sigmas, mu=mu,
    )
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    log.info("scheduling: num_steps=%d mu=%.4f", num_steps_eff, mu)

    # Guidance is a per-batch scalar fed to the transformer (no two-pass CFG).
    guidance_t = torch.full([1], guidance, device=transformer_device, dtype=torch.float32)
    guidance_t = guidance_t.expand(latents.shape[0])

    # 4. Denoising loop (single forward per step; gemlite + skip-list both fp16).
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_t,
            encoder_hidden_states=prompt_embeds_t,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]

        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    # 5. Unpack -> denormalize via VAE batch-norm stats -> unpatchify -> decode.
    # Cast to bf16 (vae dtype) on the way out of the transformer stream.
    latents = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids)  # (1, 128, H/16, W/16)
    latents = latents.to(device=vae_device, dtype=torch.bfloat16)

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents.device, latents.dtype,
    )
    latents = latents * bn_std + bn_mean
    latents = Flux2Pipeline._unpatchify_latents(latents)  # (1, 32, H/8, W/8)

    image = vae.decode(latents, return_dict=False)[0]  # (1, 3, H, W) bf16, range [-1, 1]

    # 6. Tensor -> PIL (range conversion, no diffusers VaeImageProcessor dep).
    img = image[0].clamp(-1.0, 1.0).float()
    img = (img + 1.0) * 127.5
    img = img.clamp(0.0, 255.0).round().to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()  # HWC
    return Image.fromarray(img, mode="RGB")


__all__ = ["diffusion_forward", "DEFAULT_GUIDANCE", "DEFAULT_NUM_STEPS"]
