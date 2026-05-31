from __future__ import annotations

# Unit tests for ensure_backend swap logic.
# No MLX model is constructed — FluxPipeline internals are patched.
#
#   .venv/bin/python -m unittest backend.tests.test_ensure_backend -v

import unittest
from unittest.mock import MagicMock, patch

from mflux.models.common.vae.tiling_config import TilingConfig

from backend.pipeline import (
    FluxPipeline,
    PipelineConfig,
    _default_model_path_for,
    _resolve_tiling_config,
)
from backend.server import GenerateRequest


class _FakeModel:
    def __init__(self, *, backend: str, model_path: str | None, config) -> None:
        self.backend = backend
        self.model_path = model_path
        self.config = config
        self.tiling_config = None


def _make_pipeline(default_backend: str = "bonsai-ternary-mlx") -> FluxPipeline:
    config = PipelineConfig(
        backend=default_backend,  # type: ignore[arg-type]
        baked_model_path="/tmp/baked",
        baked_binary_model_path="/tmp/baked-binary",
        te_4bit=False,  # _FakeModel lacks model_config; skip the 4-bit-TE load path.
    )
    with patch("backend.pipeline._build_model", side_effect=_FakeModel):
        return FluxPipeline(config)


class EnsureBackendTest(unittest.TestCase):
    def test_no_swap_when_backend_matches(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")
        initial_model = pipeline._model
        with patch("backend.pipeline._build_model") as build:
            pipeline.ensure_backend(backend="bonsai-ternary-mlx", model_path=None)
            build.assert_not_called()
        self.assertIs(pipeline._model, initial_model)

    def test_swap_to_binary_uses_binary_baked_path(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")
        with patch("backend.pipeline._build_model", side_effect=_FakeModel):
            pipeline.ensure_backend(backend="bonsai-binary-mlx", model_path=None)
        self.assertEqual(pipeline.backend, "bonsai-binary-mlx")
        self.assertEqual(pipeline.model_path, "/tmp/baked-binary")

    def test_swap_back_to_ternary_uses_baked_path(self) -> None:
        pipeline = _make_pipeline("bonsai-binary-mlx")
        with patch("backend.pipeline._build_model", side_effect=_FakeModel) as build:
            pipeline.ensure_backend(backend="bonsai-ternary-mlx", model_path=None)
            build.assert_called_once()
        self.assertEqual(pipeline.backend, "bonsai-ternary-mlx")
        self.assertEqual(pipeline.model_path, "/tmp/baked")

    def test_override_model_path_triggers_swap(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")
        with patch("backend.pipeline._build_model", side_effect=_FakeModel) as build:
            pipeline.ensure_backend(backend="bonsai-ternary-mlx", model_path="/tmp/custom")
            build.assert_called_once()
        self.assertEqual(pipeline.model_path, "/tmp/custom")

    def test_unknown_backend_rejected(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")
        with self.assertRaises(ValueError):
            pipeline.ensure_backend(backend="bogus", model_path=None)  # type: ignore[arg-type]


class DefaultModelPathTest(unittest.TestCase):
    def _config(self) -> PipelineConfig:
        return PipelineConfig(
            baked_model_path="/tmp/baked",
            baked_binary_model_path="/tmp/baked-binary",
        )

    def test_ternary_mlx_uses_baked(self) -> None:
        self.assertEqual(_default_model_path_for("bonsai-ternary-mlx", self._config()), "/tmp/baked")

    def test_binary_mlx_uses_binary_baked(self) -> None:
        self.assertEqual(_default_model_path_for("bonsai-binary-mlx", self._config()), "/tmp/baked-binary")

    def test_remote_backends_return_none(self) -> None:
        cfg = self._config()
        self.assertIsNone(_default_model_path_for("bonsai-ternary-gemlite", cfg))
        self.assertIsNone(_default_model_path_for("bonsai-binary-gemlite", cfg))


class GenerateRequestDefaultsTest(unittest.TestCase):
    def test_no_backend_defaults_to_none(self) -> None:
        req = GenerateRequest(prompt="x")
        self.assertIsNone(req.backend)


class ResolveTilingConfigTest(unittest.TestCase):
    # Threshold follows 2 * TilingConfig().vae_decode_tile_size (128 -> 256).
    def test_auto_on_threshold_enables_tiling(self) -> None:
        threshold = 2 * TilingConfig().vae_decode_tile_size
        cfg = _resolve_tiling_config(
            request_override=None, server_default="auto", height=threshold, width=threshold
        )
        self.assertIsInstance(cfg, TilingConfig)

    def test_auto_below_threshold_disables_tiling(self) -> None:
        threshold = 2 * TilingConfig().vae_decode_tile_size
        cfg = _resolve_tiling_config(
            request_override=None,
            server_default="auto",
            height=threshold - 1,
            width=threshold - 1,
        )
        self.assertIsNone(cfg)

    def test_request_override_true_forces_tiling(self) -> None:
        cfg = _resolve_tiling_config(
            request_override=True, server_default="off", height=64, width=64
        )
        self.assertIsInstance(cfg, TilingConfig)

    def test_request_override_false_disables_tiling(self) -> None:
        cfg = _resolve_tiling_config(
            request_override=False, server_default="on", height=4096, width=4096
        )
        self.assertIsNone(cfg)


class GuidancePassThroughTest(unittest.TestCase):
    # Verify the guidance field reaches Flux2Klein.generate_image.
    def test_custom_guidance_reaches_model(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")

        fake_generated = MagicMock()
        image = MagicMock()
        image.save = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n")
        fake_generated.image = image

        pipeline._model.generate_image = MagicMock(return_value=fake_generated)  # type: ignore[attr-defined]
        pipeline.generate_png(prompt="x", guidance=3.5)

        _, kwargs = pipeline._model.generate_image.call_args  # type: ignore[attr-defined]
        self.assertEqual(kwargs["guidance"], 3.5)

    def test_default_guidance_is_one(self) -> None:
        pipeline = _make_pipeline("bonsai-ternary-mlx")
        fake_generated = MagicMock()
        image = MagicMock()
        image.save = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n")
        fake_generated.image = image
        pipeline._model.generate_image = MagicMock(return_value=fake_generated)  # type: ignore[attr-defined]
        pipeline.generate_png(prompt="x")

        _, kwargs = pipeline._model.generate_image.call_args  # type: ignore[attr-defined]
        self.assertEqual(kwargs["guidance"], 1.0)

    def test_request_guidance_defaults_to_one(self) -> None:
        req = GenerateRequest(prompt="x")
        self.assertEqual(req.guidance, 1.0)

    def test_request_guidance_custom(self) -> None:
        req = GenerateRequest(prompt="x", guidance=4.2)
        self.assertEqual(req.guidance, 4.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
