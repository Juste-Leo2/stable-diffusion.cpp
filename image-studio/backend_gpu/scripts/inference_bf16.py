"""BF16 inference using local bonsai models, matching the C++ reference."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch


def _mflux_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


KLEIN_OUTPUT_LAYERS = (9, 18, 27)


@torch.no_grad()
def _encode_prompt(text_encoder, tokenizer, prompt: str, max_sequence_length: int) -> torch.Tensor:
    device = text_encoder.device
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True,
        max_length=max_sequence_length,
    )
    output = text_encoder(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        output_hidden_states=True, use_cache=False,
    )
    out = torch.stack([output.hidden_states[k] for k in KLEIN_OUTPUT_LAYERS], dim=1)
    _, num_channels, seq_len, hidden_dim = out.shape
    return out.permute(0, 2, 1, 3).reshape(1, seq_len, num_channels * hidden_dim)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("inference_bf16")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a cat sitting on a window sill")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/test_output_python.png"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("models/bonsai-image-binary-4B-unpacked"))
    args = parser.parse_args()
    log.info("args: %s", vars(args))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    model_dir = args.model_dir.resolve()
    device = "cuda:0"

    # 1. VAE
    log.info("loading VAE...")
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(
        str(model_dir / "vae"), torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # 2. Transformer
    log.info("loading transformer...")
    from diffusers import Flux2Transformer2DModel
    transformer = Flux2Transformer2DModel.from_pretrained(
        str(model_dir / "transformer"), torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # 3. Text encoder + tokenizer
    log.info("loading text encoder...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir / "tokenizer"), trust_remote_code=True,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        str(model_dir / "text_encoder"),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
    ).to(device).eval()

    # 4. Scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler",
    )

    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
    )
    pipe.set_progress_bar_config(disable=True)
    log.info("all models loaded")

    # Encode prompt (Klein/Qwen3 stacking)
    log.info("encoding prompt...")
    prompt_embeds = _encode_prompt(text_encoder, tokenizer, args.prompt, max_sequence_length=512)
    prompt_embeds = prompt_embeds.to(device=device, dtype=torch.bfloat16)

    # Generate
    log.info("generating...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    image = pipe(
        prompt_embeds=prompt_embeds,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
    ).images[0]
    elapsed = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(args.output))
    log.info("generated %dx%d in %.2fs, peak HBM %.0f MiB -> %s",
             image.width, image.height, elapsed, peak_mib, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())