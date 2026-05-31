from __future__ import annotations

# Loader-internal tests. Mocks `gemlite`, `hqq`, `transformers`, and
# `diffusers` via `sys.modules` so the loaders run on macOS without a
# CUDA stack.
#
#   .venv/bin/python -m unittest backend_gpu.tests.test_loaders -v

import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

# GpuPipeline now requires every artifact path to be explicitly set (no
# baked-in absolute defaults — different hosts have incompatible layouts).
# Inject "fake but set" paths at import time; the loaders themselves are
# patched in each test so these strings are never opened.
os.environ.setdefault("MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH", "/tmp/__binary__")
os.environ.setdefault("MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH", "/tmp/__ternary__")
os.environ.setdefault("MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH", "/tmp/__te__")
os.environ.setdefault("MFLUX_STUDIO_GPU_VAE_PATH", "/tmp/__vae__")
os.environ.setdefault("MFLUX_STUDIO_GPU_TOKENIZER_PATH", "/tmp/__tok__")


@contextmanager
def _inject_module(name: str, module: types.ModuleType):
    """Set sys.modules[name]=module then restore on exit (single key only).

    Why not `patch.dict(sys.modules, {...})`: that snapshots and wholesale-
    restores the entire dict on exit, which wipes any modules imported during
    the test (e.g. torch + all submodules). A second `import torch` then tries
    to re-init the C extension and fails ('docstring already set').
    """
    prev = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev

from backend_gpu.pipeline_gpu import (
    GpuPipeline,
    _load_gemlite_layers_from_state,
    _load_gemlite_transformer,
    _load_scheduler,
    _load_text_encoder,
    _load_tokenizer,
    _load_vae,
    _null_gemlite_weights,
)


class LoadGemliteLayersFromStateTest(unittest.TestCase):
    def test_buckets_keys_and_calls_per_layer_load(self) -> None:
        # Real torch model with two nn.Linear children at known FQNs. Saved
        # state_dict has gemlite-layer keys for one of them and ordinary
        # weights for the other; only the gemlite one should be replaced.
        import torch
        import torch.nn as nn

        class _FakeGemLite(nn.Module):
            constructed: list[tuple[int, int]] = []
            loaded: list[dict] = []

            def __init__(self, *, W_nbits, group_size, in_features, out_features, input_dtype, output_dtype):
                super().__init__()
                self.W_nbits = W_nbits
                self.group_size = group_size
                self.in_features = in_features
                self.out_features = out_features
                _FakeGemLite.constructed.append((in_features, out_features))
                # Tensor attrs filled by load_state_dict; pre-set so .to() works.
                self.W_q = torch.zeros(1)
                self.scales = torch.zeros(1)
                self.zeros = torch.zeros(1)
                self.bias = None

            def load_state_dict(self, sd, strict=True):  # noqa: ARG002 - mirror gemlite shape
                _FakeGemLite.loaded.append(dict(sd))
                # Mimic gemlite: pop tensor keys onto attrs (caller .to()s after).
                for k, v in sd.items():
                    setattr(self, k, v)

            def forward(self, x):
                return x

        class _FakeDType:
            FP16 = "fp16"

        model = nn.Module()
        model.attn = nn.Module()
        model.attn.to_q = nn.Linear(128, 256, bias=False)
        model.norm_out = nn.Linear(64, 32, bias=False)  # not in state's gemlite bucket

        state = {
            # gemlite-layer keys for attn.to_q
            "attn.to_q.W_q": torch.ones(2, 4, dtype=torch.uint8),
            "attn.to_q.scales": torch.ones(2),
            "attn.to_q.zeros": torch.ones(2),
            "attn.to_q.metadata": torch.zeros(8, dtype=torch.int32),
            "attn.to_q.orig_shape": torch.tensor([256, 128]),
            "attn.to_q.bias": torch.zeros(256),
            # plain weight for norm_out
            "norm_out.weight": torch.randn(32, 64),
            # head-level scalar (e.g. time_embed)
            "time_embed.weight": torch.randn(8, 8),
        }

        n, remainder = _load_gemlite_layers_from_state(
            model, state,
            bits=1, group_size=128, device="cpu",
            DType=_FakeDType, GemLiteLinearTriton=_FakeGemLite,
        )
        self.assertEqual(n, 1)
        self.assertEqual(_FakeGemLite.constructed, [(128, 256)])
        self.assertEqual(set(remainder.keys()), {"norm_out.weight", "time_embed.weight"})
        # The replaced child is the fake gemlite layer.
        self.assertIsInstance(model.attn.to_q, _FakeGemLite)
        # Custom load_state_dict was dispatched once with the 6 bucketed keys.
        self.assertEqual(len(_FakeGemLite.loaded), 1)
        self.assertEqual(
            set(_FakeGemLite.loaded[0].keys()),
            {"W_q", "bias", "scales", "zeros", "metadata", "orig_shape"},
        )

    def test_raises_when_state_targets_non_linear(self) -> None:
        import torch
        import torch.nn as nn

        class _FakeGemLite(nn.Module):
            def __init__(self, **kw):
                super().__init__()

            def forward(self, x):
                return x

        class _FakeDType:
            FP16 = "fp16"

        model = nn.Module()
        # No nn.Linear at "ghost" — just a Module.
        model.ghost = nn.Module()

        state = {
            "ghost.W_q": torch.ones(1, dtype=torch.uint8),
            "ghost.scales": torch.ones(1),
        }
        with self.assertRaisesRegex(RuntimeError, "but model has Module"):
            _load_gemlite_layers_from_state(
                model, state,
                bits=1, group_size=128, device="cpu",
                DType=_FakeDType, GemLiteLinearTriton=_FakeGemLite,
            )


class NullGemliteWeightsTest(unittest.TestCase):
    def test_nulls_only_gemlite_modules(self) -> None:
        import torch.nn as nn

        class _FakeGemLite(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = "still-here"

            def forward(self, x):
                return x

        model = nn.Module()
        model.gl = _FakeGemLite()
        model.linear = nn.Linear(4, 4, bias=False)
        original_linear_weight = model.linear.weight

        nulled = _null_gemlite_weights(model, _FakeGemLite)

        self.assertEqual(nulled, 1)
        self.assertIsNone(model.gl.weight)
        self.assertIs(model.linear.weight, original_linear_weight)


class TextEncoderLoaderTest(unittest.TestCase):
    def test_missing_path_raises_file_not_found(self) -> None:
        bogus = Path("/tmp/definitely-does-not-exist-te")
        with self.assertRaisesRegex(FileNotFoundError, "Text encoder artifact not found"):
            _load_text_encoder(bogus)

    def test_loads_via_hqq_when_path_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "te"
            artifact.mkdir()

            fake_torch = MagicMock(name="torch")
            fake_torch.float16 = "fp16-sentinel"
            fake_gemlite_core = MagicMock(name="gemlite.core")
            fake_gemlite = types.ModuleType("gemlite")
            fake_gemlite.core = fake_gemlite_core
            fake_hqq_models_hf_base = MagicMock(name="hqq.models.hf.base")
            fake_hqq_utils_patching = MagicMock(name="hqq.utils.patching")
            fake_loaded_model = MagicMock(name="loaded_te_model")
            fake_hqq_models_hf_base.AutoHQQHFModel.from_quantized.return_value = fake_loaded_model

            modules = {
                "torch": fake_torch,
                "gemlite": fake_gemlite,
                "gemlite.core": fake_gemlite_core,
                "hqq": types.ModuleType("hqq"),
                "hqq.models": types.ModuleType("hqq.models"),
                "hqq.models.hf": types.ModuleType("hqq.models.hf"),
                "hqq.models.hf.base": fake_hqq_models_hf_base,
                "hqq.utils": types.ModuleType("hqq.utils"),
                "hqq.utils.patching": fake_hqq_utils_patching,
            }
            with patch.dict(sys.modules, modules):
                result = _load_text_encoder(artifact, device="cuda:0")
            self.assertIs(result, fake_loaded_model)
            fake_gemlite_core.set_packing_bitwidth.assert_called_once_with(8)
            fake_hqq_models_hf_base.AutoHQQHFModel.from_quantized.assert_called_once_with(
                str(artifact), compute_dtype="fp16-sentinel", device="cuda:0",
            )
            fake_hqq_utils_patching.prepare_for_inference.assert_called_once_with(
                fake_loaded_model, backend="gemlite",
            )


class VaeLoaderTest(unittest.TestCase):
    def test_missing_path_raises(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "VAE snapshot not found"):
            _load_vae(Path("/tmp/__no_vae_snapshot__"))

    def test_calls_autoencoder_klflux2_from_pretrained(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vae_path = Path(td) / "vae"
            vae_path.mkdir()

            fake_torch = MagicMock(name="torch")
            fake_torch.bfloat16 = "bf16-sentinel"
            fake_diffusers = MagicMock(name="diffusers")
            fake_vae = MagicMock(name="vae_model")
            fake_diffusers.AutoencoderKLFlux2.from_pretrained.return_value = fake_vae
            fake_vae.to.return_value = fake_vae
            fake_vae.eval.return_value = fake_vae

            modules = {
                "torch": fake_torch,
                "diffusers": fake_diffusers,
            }
            with patch.dict(sys.modules, modules):
                result = _load_vae(vae_path, device="cuda:0")

            self.assertIs(result, fake_vae)
            fake_diffusers.AutoencoderKLFlux2.from_pretrained.assert_called_once_with(
                str(vae_path), torch_dtype="bf16-sentinel",
            )
            fake_vae.to.assert_called_once_with("cuda:0")
            fake_vae.eval.assert_called_once_with()


class TokenizerLoaderTest(unittest.TestCase):
    def test_calls_auto_tokenizer_from_pretrained(self) -> None:
        # Klein/Qwen3 path: `_encode_klein_qwen3_prompt` only needs
        # `apply_chat_template` + `__call__`, which AutoTokenizer provides.
        # The TE artifact ships its own `tokenizer/` subdir.
        fake_transformers = MagicMock(name="transformers")
        fake_tok = MagicMock(name="tokenizer")
        fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tok

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            result = _load_tokenizer("/root/models/klein-4b-text-encoder-hqq-4bit-gemlite/tokenizer/")

        self.assertIs(result, fake_tok)
        fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "/root/models/klein-4b-text-encoder-hqq-4bit-gemlite/tokenizer/",
        )


class SchedulerLoaderTest(unittest.TestCase):
    def test_returns_none_when_subfolder_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tx_path = Path(td) / "tx"
            tx_path.mkdir()
            self.assertIsNone(_load_scheduler(tx_path))

    def test_loads_when_subfolder_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tx_path = Path(td) / "tx"
            (tx_path / "scheduler").mkdir(parents=True)

            fake_diffusers = MagicMock(name="diffusers")
            fake_sched = MagicMock(name="scheduler")
            fake_diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained.return_value = fake_sched

            with patch.dict(sys.modules, {"diffusers": fake_diffusers}):
                result = _load_scheduler(tx_path)

            self.assertIs(result, fake_sched)
            fake_diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained.assert_called_once_with(
                str(tx_path), subfolder="scheduler",
            )


class GemliteTransformerLoaderTest(unittest.TestCase):
    def _make_artifact(self, td: str, *, qcfg_extra: dict | None = None) -> Path:
        path = Path(td) / "transformer"
        path.mkdir()
        (path / "state_dict.pt").write_bytes(b"fake")
        (path / "config.json").write_text(json.dumps({"in_channels": 16}))
        qcfg = {
            "format": "gemlite-int1-g128",
            "bits": 1,
            "group_size": 128,
            "packing_bitwidth": 8,
        }
        if qcfg_extra:
            qcfg.update(qcfg_extra)
        (path / "quantization_config.json").write_text(json.dumps(qcfg))
        (path / "gemlite_autotune.json").write_text("{}")
        return path

    def test_missing_path_raises(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Gemlite transformer artifact not found"):
            _load_gemlite_transformer(Path("/tmp/__nope__"))

    def test_missing_state_dict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._make_artifact(td)
            (path / "state_dict.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "state_dict.pt"):
                _load_gemlite_transformer(path)

    def test_missing_autotune_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = self._make_artifact(td)
            (path / "gemlite_autotune.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "gemlite_autotune.json"):
                _load_gemlite_transformer(path)

    def test_full_load_call_sequence(self) -> None:
        # Models the chain:
        #   set_packing_bitwidth(packing_bw)
        #   GemLiteLinearTriton.load_config(autotune_path)
        #   Flux2Transformer2DModel.from_config(cfg)         -> m_cfg
        #   m_cfg.to(bf16)                                   -> m_bf16
        #   torch.load(state_path)                           -> state
        #   _load_gemlite_layers_from_state(m_bf16, state, ...) -> (n, remainder)
        #   m_bf16.load_state_dict(remainder, strict=False)
        #   m_bf16.to(fp16)                                  -> m_fp16
        #   _null_gemlite_weights(m_fp16, GemLite)
        #   m_fp16.to(device)                                -> m_dev
        #   m_dev.eval()                                     -> m_dev
        with tempfile.TemporaryDirectory() as td:
            path = self._make_artifact(td)

            fake_torch = MagicMock(name="torch")
            fake_torch.bfloat16 = "bf16-sentinel"
            fake_torch.float16 = "fp16-sentinel"
            fake_state = {"fake.key": "v", "x.W_q": "g"}
            fake_torch.load.return_value = fake_state

            fake_gemlite_core = MagicMock(name="gemlite.core")
            fake_gemlite = types.ModuleType("gemlite")
            fake_gemlite.core = fake_gemlite_core

            fake_diffusers = MagicMock(name="diffusers")
            m_cfg = MagicMock(name="m_from_config")
            m_bf16 = MagicMock(name="m_bf16")
            m_fp16 = MagicMock(name="m_fp16")
            m_dev = MagicMock(name="m_dev")
            m_cfg.to.return_value = m_bf16
            m_bf16.load_state_dict.return_value = ([], [])
            m_bf16.to.return_value = m_fp16
            m_fp16.to.return_value = m_dev
            m_dev.eval.return_value = m_dev
            fake_diffusers.Flux2Transformer2DModel.from_config.return_value = m_cfg

            modules = {
                "torch": fake_torch,
                "gemlite": fake_gemlite,
                "gemlite.core": fake_gemlite_core,
                "diffusers": fake_diffusers,
            }
            stub_remainder = {"some.weight": "rem"}
            with (
                patch.dict(sys.modules, modules),
                patch(
                    "backend_gpu.pipeline_gpu._load_gemlite_layers_from_state",
                    return_value=(140, stub_remainder),
                ) as mock_loader,
                patch(
                    "backend_gpu.pipeline_gpu._null_gemlite_weights", return_value=140,
                ) as mock_nuller,
            ):
                result = _load_gemlite_transformer(path, device="cuda:0")

            self.assertIs(result, m_dev)
            # Module-level set_packing_bitwidth precedes classmethod load_config.
            fake_gemlite_core.set_packing_bitwidth.assert_called_once_with(8)
            fake_gemlite_core.GemLiteLinearTriton.load_config.assert_called_once_with(
                str(path / "gemlite_autotune.json"),
            )

            fake_diffusers.Flux2Transformer2DModel.from_config.assert_called_once()
            m_cfg.to.assert_called_once_with("bf16-sentinel")
            fake_torch.load.assert_called_once_with(str(path / "state_dict.pt"), map_location="cpu")

            mock_loader.assert_called_once()
            kw = mock_loader.call_args.kwargs
            self.assertEqual(kw["bits"], 1)
            self.assertEqual(kw["group_size"], 128)
            self.assertEqual(kw["device"], "cuda:0")
            # Loader gets the raw state from torch.load.
            self.assertIs(mock_loader.call_args.args[1], fake_state)

            # Remainder (non-gemlite keys) goes through model.load_state_dict.
            m_bf16.load_state_dict.assert_called_once_with(stub_remainder, strict=False)
            # bf16 → fp16 cast happens before device move.
            m_bf16.to.assert_called_once_with("fp16-sentinel")
            mock_nuller.assert_called_once()
            self.assertIs(mock_nuller.call_args.args[0], m_fp16)
            m_fp16.to.assert_called_once_with("cuda:0")
            m_dev.eval.assert_called_once_with()


class PrewarmErrorPathTest(unittest.TestCase):
    def test_missing_transformer_artifact_surfaces_error(self) -> None:
        pipe = GpuPipeline(
            transformer_path="/tmp/__no_transformer__",
            text_encoder_path="/tmp/__no_te__",
            vae_path="/tmp/__no_vae__",
            tokenizer_path="fake",
        )
        with self.assertRaisesRegex(FileNotFoundError, "Gemlite transformer artifact not found"):
            pipe.prewarm()
        self.assertFalse(pipe.ready)

    def test_missing_text_encoder_surfaces_file_not_found(self) -> None:
        pipe = GpuPipeline(
            transformer_path="/tmp/__no_transformer__",
            text_encoder_path="/tmp/__no_te__",
            vae_path="/tmp/__no_vae__",
            tokenizer_path="fake",
        )
        with patch("backend_gpu.pipeline_gpu._load_gemlite_transformer", return_value=MagicMock()):
            with self.assertRaisesRegex(FileNotFoundError, "Text encoder artifact not found"):
                pipe.prewarm()
        self.assertFalse(pipe.ready)

    def test_missing_vae_surfaces_file_not_found(self) -> None:
        pipe = GpuPipeline(
            transformer_path="/tmp/__no_transformer__",
            text_encoder_path="/tmp/__no_te__",
            vae_path="/tmp/__no_vae__",
            tokenizer_path="fake",
        )
        with (
            patch("backend_gpu.pipeline_gpu._load_gemlite_transformer", return_value=MagicMock()),
            patch("backend_gpu.pipeline_gpu._load_text_encoder", return_value=MagicMock()),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "VAE snapshot not found"):
                pipe.prewarm()
        self.assertFalse(pipe.ready)

    def test_prewarm_succeeds_with_all_loaders_mocked(self) -> None:
        pipe = GpuPipeline()
        with (
            patch("backend_gpu.pipeline_gpu._load_gemlite_transformer", return_value="tx"),
            patch("backend_gpu.pipeline_gpu._load_text_encoder", return_value="te"),
            patch("backend_gpu.pipeline_gpu._load_vae", return_value="vae"),
            patch("backend_gpu.pipeline_gpu._load_tokenizer", return_value="tok"),
            patch("backend_gpu.pipeline_gpu._load_scheduler", return_value="sched"),
        ):
            pipe.prewarm()
        self.assertTrue(pipe.ready)
        self.assertEqual(pipe._transformer, "tx")
        self.assertEqual(pipe._text_encoder, "te")
        self.assertEqual(pipe._vae, "vae")
        self.assertEqual(pipe._tokenizer, "tok")
        self.assertEqual(pipe._scheduler, "sched")

    def test_prewarm_tolerates_absent_scheduler(self) -> None:
        pipe = GpuPipeline()
        with (
            patch("backend_gpu.pipeline_gpu._load_gemlite_transformer", return_value="tx"),
            patch("backend_gpu.pipeline_gpu._load_text_encoder", return_value="te"),
            patch("backend_gpu.pipeline_gpu._load_vae", return_value="vae"),
            patch("backend_gpu.pipeline_gpu._load_tokenizer", return_value="tok"),
            patch("backend_gpu.pipeline_gpu._load_scheduler", return_value=None),
        ):
            pipe.prewarm()
        self.assertTrue(pipe.ready)
        self.assertIsNone(pipe._scheduler)


class GeneratePngTest(unittest.TestCase):
    """Unit-level coverage for `GpuPipeline.generate_png` after Phase 5c-3 wire-up.

    Server-level coverage lives in test_server.py; these test the kwargs-routing
    contract, ready-gate, and PNG roundtrip without going through FastAPI.
    """

    def _make_ready_pipeline(self) -> GpuPipeline:
        pipe = GpuPipeline()
        pipe._transformer = MagicMock(name="transformer")
        pipe._text_encoder = MagicMock(name="text_encoder")
        pipe._tokenizer = MagicMock(name="tokenizer")
        pipe._vae = MagicMock(name="vae")
        pipe._scheduler = MagicMock(name="scheduler")
        pipe._ready = True
        return pipe

    def _inject_fake_diffusion_klein(self, *, image_size: tuple[int, int] = (32, 32)):
        from PIL import Image

        fake_dk = types.ModuleType("backend_gpu.diffusion_klein")
        captured: dict = {}

        def fake_forward(**kw):
            captured.clear()
            captured.update(kw)
            return Image.new("RGB", image_size, (200, 100, 50))

        fake_dk.diffusion_forward = fake_forward
        return fake_dk, captured

    def test_raises_when_not_ready(self) -> None:
        pipe = GpuPipeline()
        with self.assertRaisesRegex(RuntimeError, "prewarm.*before generate_png"):
            pipe.generate_png(prompt="x")

    def test_passes_kwargs_to_diffusion_forward_and_returns_png(self) -> None:
        pipe = self._make_ready_pipeline()
        fake_dk, captured = self._inject_fake_diffusion_klein(image_size=(64, 48))

        with _inject_module("backend_gpu.diffusion_klein", fake_dk):
            result = pipe.generate_png(
                prompt="a bonsai", seed=7, steps=12, height=48, width=64, guidance=4.5,
            )

        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG\r\n\x1a\n"))
        # Routing contract: every constructor-injected artifact + the user kwargs
        # land in diffusion_forward with the expected names.
        self.assertIs(captured["transformer"], pipe._transformer)
        self.assertIs(captured["text_encoder"], pipe._text_encoder)
        self.assertIs(captured["tokenizer"], pipe._tokenizer)
        self.assertIs(captured["vae"], pipe._vae)
        self.assertIs(captured["scheduler"], pipe._scheduler)
        self.assertEqual(captured["prompt"], "a bonsai")
        self.assertEqual(captured["seed"], 7)
        self.assertEqual(captured["num_steps"], 12)
        self.assertEqual(captured["height"], 48)
        self.assertEqual(captured["width"], 64)
        self.assertEqual(captured["guidance"], 4.5)
        # max_sequence_length omitted when caller passes None — diffusion_forward's
        # own default (512) wins, not a hardcoded override.
        self.assertNotIn("max_sequence_length", captured)
        # No CUDA on macOS ⇒ peak memory recorded as 0.0 (not None).
        self.assertEqual(pipe.last_peak_memory_mb, 0.0)

    def test_passes_max_sequence_length_when_set(self) -> None:
        pipe = self._make_ready_pipeline()
        fake_dk, captured = self._inject_fake_diffusion_klein()

        with _inject_module("backend_gpu.diffusion_klein", fake_dk):
            pipe.generate_png(prompt="x", max_sequence_length=256)
        self.assertEqual(captured["max_sequence_length"], 256)

    def test_records_cuda_peak_memory_when_available(self) -> None:
        # Patch `torch.cuda` attrs in place rather than swapping `sys.modules["torch"]`
        # — torch's C-level init state breaks if reimported, so a clean restore is
        # not enough; we'd corrupt later test_server tests on the way out.
        import torch

        pipe = self._make_ready_pipeline()
        fake_dk, _ = self._inject_fake_diffusion_klein()

        with (
            _inject_module("backend_gpu.diffusion_klein", fake_dk),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "max_memory_allocated", return_value=256 * 1024 * 1024),
            patch.object(torch.cuda, "reset_peak_memory_stats") as mock_reset,
        ):
            pipe.generate_png(prompt="x")
        mock_reset.assert_called_once_with()
        self.assertEqual(pipe.last_peak_memory_mb, 256.0)


class ConfigEnvVarTest(unittest.TestCase):
    def test_env_vars_picked_up(self) -> None:
        # Use the modern, backend-suffixed env name — the legacy unsuffixed
        # MFLUX_STUDIO_GPU_TRANSFORMER_PATH is still honored as a fallback
        # (covered by test_explicit_kwargs_override_env's legacy probe).
        env_overrides = {
            "MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH": "/some/tx/path",
            "MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH": "/some/te/path",
            "MFLUX_STUDIO_GPU_VAE_PATH": "/some/vae/path",
            "MFLUX_STUDIO_GPU_TOKENIZER_PATH": "foo/bar-tok",
            "MFLUX_STUDIO_GPU_DEVICE": "cuda:1",
        }
        with patch.dict("os.environ", env_overrides, clear=False):
            pipe = GpuPipeline()
        self.assertEqual(pipe.transformer_path, Path("/some/tx/path"))
        self.assertEqual(pipe.text_encoder_path, Path("/some/te/path"))
        self.assertEqual(pipe.vae_path, Path("/some/vae/path"))
        self.assertEqual(pipe.tokenizer_path, "foo/bar-tok")
        self.assertEqual(pipe.device, "cuda:1")

    def test_explicit_kwargs_override_env(self) -> None:
        # Legacy `transformer_path` kwarg must beat the legacy unsuffixed env
        # (`MFLUX_STUDIO_GPU_TRANSFORMER_PATH`). Note: the BINARY-suffixed env
        # (when set) takes priority over the legacy kwarg by design — see the
        # fallback chain in GpuPipeline.__init__. We clear BINARY here so the
        # legacy-only path is exercised.
        env = os.environ.copy()
        env.pop("MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH", None)
        env["MFLUX_STUDIO_GPU_TRANSFORMER_PATH"] = "/from/legacy/env"
        with patch.dict("os.environ", env, clear=True):
            pipe = GpuPipeline(transformer_path="/from/kwarg")
        self.assertEqual(pipe.transformer_path, Path("/from/kwarg"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
