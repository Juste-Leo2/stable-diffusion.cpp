from __future__ import annotations

# Unit tests for backend_gpu.server. The 4 loaders are patched out so this
# whole suite runs anywhere — no GPU, no gemlite/HQQ/diffusers required.
#
#   .venv/bin/python -m unittest backend_gpu.tests.test_server -v

import base64
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from PIL import Image

from fastapi.testclient import TestClient


_TEST_TOKEN = "test-bearer-token"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@contextmanager
def _patched_app():
    # Why: GpuPipeline.prewarm() loads 5 heavy artifacts (gemlite/HQQ/
    # diffusers) and generate_png imports `backend_gpu.diffusion_klein`,
    # which itself pulls in `diffusers.pipelines.flux2`. We inject a fake
    # diffusion_klein module + patch the loaders so the server tests stay
    # GPU-free and don't need the heavy diffusers stack.
    os.environ["MFLUX_STUDIO_GPU_TOKEN"] = _TEST_TOKEN
    # The artifact paths now require explicit env vars (loaders are patched
    # below so the strings never actually get opened).
    os.environ.setdefault("MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH", "/root/models/bonsai-binary")
    os.environ.setdefault("MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH", "/root/models/bonsai-ternary")
    os.environ.setdefault("MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH", "/root/models/text-encoder")
    os.environ.setdefault("MFLUX_STUDIO_GPU_VAE_PATH", "/root/models/vae")
    os.environ.setdefault("MFLUX_STUDIO_GPU_TOKENIZER_PATH", "/root/models/text-encoder/tokenizer")
    fake_transformer = MagicMock(name="transformer")
    fake_te = MagicMock(name="text_encoder")
    fake_vae = MagicMock(name="vae")
    fake_tokenizer = MagicMock(name="tokenizer")

    fake_dk = types.ModuleType("backend_gpu.diffusion_klein")
    fake_dk.diffusion_forward = MagicMock(
        side_effect=lambda **kw: Image.new("RGB", (kw["width"], kw["height"]), (12, 34, 56)),
    )
    fake_dk.DEFAULT_NUM_STEPS = 4
    fake_dk.DEFAULT_GUIDANCE = 1.0

    # Why not patch.dict(sys.modules, ...): patch.dict snapshots the dict on
    # enter and *wholesale-restores* on exit, which wipes any modules (e.g.
    # torch + all submodules) loaded mid-test. A second `import torch` then
    # tries to re-init the C extension → "docstring already set". Manual
    # set+pop touches only the one key we own.
    prev_dk = sys.modules.get("backend_gpu.diffusion_klein")
    sys.modules["backend_gpu.diffusion_klein"] = fake_dk
    try:
        with (
            patch("backend_gpu.pipeline_gpu._load_gemlite_transformer", return_value=fake_transformer),
            patch("backend_gpu.pipeline_gpu._load_text_encoder", return_value=fake_te),
            patch("backend_gpu.pipeline_gpu._load_vae", return_value=fake_vae),
            patch("backend_gpu.pipeline_gpu._load_tokenizer", return_value=fake_tokenizer),
        ):
            from backend_gpu.server import app

            with TestClient(app) as client:
                yield client
    finally:
        if prev_dk is None:
            sys.modules.pop("backend_gpu.diffusion_klein", None)
        else:
            sys.modules["backend_gpu.diffusion_klein"] = prev_dk


class HealthzTest(unittest.TestCase):
    def test_healthz_no_auth(self) -> None:
        with _patched_app() as client:
            response = client.get("/healthz")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})


class AuthTest(unittest.TestCase):
    def test_generate_without_token_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post("/generate", json={"prompt": "x"})
            self.assertEqual(response.status_code, 401)

    def test_generate_with_wrong_token_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate",
                json={"prompt": "x"},
                headers={"Authorization": "Bearer not-the-right-token"},
            )
            self.assertEqual(response.status_code, 401)

    def test_compare_without_token_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post("/generate/compare", json={"prompt": "x"})
            self.assertEqual(response.status_code, 401)


class GenerateTest(unittest.TestCase):
    def test_generate_returns_png(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate",
                json={"prompt": "a cat", "width": 256, "height": 256},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], "image/png")
            self.assertTrue(response.content.startswith(_PNG_MAGIC))
            self.assertIn("X-Wall-Seconds", response.headers)
            self.assertIn("X-Peak-Memory-MB", response.headers)

    def test_generate_rejects_unknown_backend(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate",
                json={"prompt": "x", "backend": "bogus-backend"},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)

    def test_generate_rejects_zero_steps(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate",
                json={"prompt": "x", "steps": 0},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)

    def test_generate_rejects_empty_prompt(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate",
                json={"prompt": ""},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)


class CompareTest(unittest.TestCase):
    def test_compare_default_backends(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate/compare",
                json={"prompt": "a cat"},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(len(body["results"]), 2)
            self.assertEqual(
                [r["backend"] for r in body["results"]],
                ["bonsai-binary-gemlite", "bonsai-ternary-gemlite"],
            )
            for result in body["results"]:
                decoded = base64.b64decode(result["png_b64"])
                self.assertTrue(decoded.startswith(_PNG_MAGIC))
                self.assertGreaterEqual(result["wall_seconds"], 0.0)
                self.assertGreaterEqual(result["swap_seconds"], 0.0)

    def test_compare_empty_list_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate/compare",
                json={"prompt": "x", "backends": []},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)

    def test_compare_unknown_backend_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate/compare",
                json={"prompt": "x", "backends": ["bogus"]},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)

    def test_compare_duplicates_rejected(self) -> None:
        with _patched_app() as client:
            response = client.post(
                "/generate/compare",
                json={"prompt": "x", "backends": ["bonsai-ternary-gemlite", "bonsai-ternary-gemlite"]},
                headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
            )
            self.assertEqual(response.status_code, 422)


class NotReadyTest(unittest.TestCase):
    def test_generate_raises_when_pipeline_not_ready(self) -> None:
        # Lifespan prewarms by default; flip _ready off after entry to assert
        # the GpuPipeline.generate_png guard fires (raises RuntimeError) and
        # bubbles up. TestClient defaults to raise_server_exceptions=True so
        # the unhandled error surfaces here rather than as a 500.
        with _patched_app() as client:
            client.app.state.pipeline._ready = False
            with self.assertRaises(RuntimeError):
                client.post(
                    "/generate",
                    json={"prompt": "x"},
                    headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                )


class StartupTest(unittest.TestCase):
    def test_lifespan_requires_token(self) -> None:
        os.environ.pop("MFLUX_STUDIO_GPU_TOKEN", None)
        # Re-import to get a fresh app object whose lifespan has not yet run.
        import importlib

        import backend_gpu.server as server_module

        importlib.reload(server_module)
        with self.assertRaises(RuntimeError):
            with TestClient(server_module.app):
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
