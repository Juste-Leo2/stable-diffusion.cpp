from __future__ import annotations

# Unit tests for POST /generate/compare. FluxPipeline internals are stubbed out —
# no MLX model is constructed.
#
#   .venv/bin/python -m unittest backend.tests.test_generate_compare -v

import base64
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.pipeline import BACKENDS, LOCAL_BACKENDS
from backend.server import CompareRequest, app


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _fake_png(backend: str) -> bytes:
    return _PNG_MAGIC + f"fake-{backend}".encode()


class _FakeModel:
    # Stand-in for Flux2Klein inside FluxPipeline — only the surface used by
    # generate_png is faked out. generate_image returns a PIL-ish stub whose
    # save() writes a unique marker per backend.
    def __init__(self, *, backend: str) -> None:
        self.backend = backend
        self.tiling_config = None

    def generate_image(self, **_kwargs):
        backend = self.backend

        class _Image:
            def save(self, buf, format):  # noqa: A002 — FastAPI signature
                buf.write(_fake_png(backend))

        class _Generated:
            image = _Image()

        return _Generated()


@contextmanager
def _patched_app():
    # Replace _build_model so FluxPipeline.__init__ (invoked by the lifespan)
    # never constructs a real Flux2Klein — we get a lightweight stub instead.
    # _FakeModel lacks model_config; opt out of 4-bit-TE so _load works.
    with patch.dict("os.environ", {"MFLUX_STUDIO_TE_4BIT": "0"}, clear=False):
        with patch("backend.pipeline._build_model", side_effect=lambda *, backend, **_kw: _FakeModel(backend=backend)):
            with TestClient(app) as client:
                yield client, client.app.state.pipeline


class CompareRequestValidationTest(unittest.TestCase):
    def test_defaults_to_all_three_backends(self) -> None:
        req = CompareRequest(prompt="x")
        self.assertEqual(req.backends, list(LOCAL_BACKENDS))

    def test_empty_backends_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompareRequest(prompt="x", backends=[])

    def test_unknown_backend_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompareRequest(prompt="x", backends=["bogus"])  # type: ignore[list-item]

    def test_duplicates_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CompareRequest(prompt="x", backends=["bonsai-ternary-mlx", "bonsai-ternary-mlx"])

    def test_subset_accepted(self) -> None:
        req = CompareRequest(prompt="x", backends=["bonsai-ternary-mlx", "bonsai-binary-mlx"])
        self.assertEqual(req.backends, ["bonsai-ternary-mlx", "bonsai-binary-mlx"])

class CompareEndpointTest(unittest.TestCase):
    # End-to-end: POST /generate/compare with a stubbed _build_model, assert
    # serialized backend iteration + per-backend result shape.

    def test_default_runs_all_local_backends(self) -> None:
        with _patched_app() as (client, pipeline):
            response = client.post("/generate/compare", json={"prompt": "a cat"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(len(body["results"]), len(LOCAL_BACKENDS))
            self.assertEqual([r["backend"] for r in body["results"]], list(LOCAL_BACKENDS))
            # Final backend is the last one in the request order.
            self.assertEqual(pipeline.backend, LOCAL_BACKENDS[-1])

    def test_result_shape(self) -> None:
        with _patched_app() as (client, _):
            response = client.post(
                "/generate/compare",
                json={"prompt": "a cat", "backends": ["bonsai-ternary-mlx", "bonsai-binary-mlx"]},
            )
            self.assertEqual(response.status_code, 200, response.text)
            for result in response.json()["results"]:
                self.assertIn(result["backend"], BACKENDS)
                decoded = base64.b64decode(result["png_b64"])
                self.assertTrue(decoded.startswith(_PNG_MAGIC))
                self.assertTrue(decoded.endswith(f"fake-{result['backend']}".encode()))
                self.assertIsInstance(result["wall_seconds"], float)
                self.assertIsInstance(result["swap_seconds"], float)
                self.assertGreaterEqual(result["wall_seconds"], 0.0)
                self.assertGreaterEqual(result["swap_seconds"], 0.0)

    def test_preserves_order(self) -> None:
        with _patched_app() as (client, _):
            response = client.post(
                "/generate/compare",
                json={
                    "prompt": "x",
                    "backends": ["bonsai-binary-mlx", "bonsai-ternary-mlx"],
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                [r["backend"] for r in response.json()["results"]],
                ["bonsai-binary-mlx", "bonsai-ternary-mlx"],
            )

    def test_custom_params_reach_pipeline(self) -> None:
        with _patched_app() as (client, pipeline):
            captured: dict = {}
            original = pipeline._model.generate_image  # type: ignore[union-attr]

            def _spy(**kwargs):
                captured.update(kwargs)
                return original(**kwargs)

            # Replace every lazily-built _FakeModel's generate_image with a spy
            # by patching _build_model to wrap the produced model.
            with patch("backend.pipeline._build_model", side_effect=lambda *, backend, **_kw: _wrap_with_spy(_FakeModel(backend=backend), captured)):
                response = client.post(
                    "/generate/compare",
                    json={
                        "prompt": "detailed cat",
                        "seed": 123,
                        "steps": 8,
                        "guidance": 3.5,
                        "height": 1024,
                        "width": 768,
                        "backends": ["bonsai-binary-mlx"],
                    },
                )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(captured["seed"], 123)
            self.assertEqual(captured["num_inference_steps"], 8)
            self.assertEqual(captured["guidance"], 3.5)
            self.assertEqual(captured["height"], 1024)
            self.assertEqual(captured["width"], 768)
            self.assertEqual(captured["prompt"], "detailed cat")

    def test_empty_backends_rejected_by_api(self) -> None:
        with _patched_app() as (client, _):
            response = client.post(
                "/generate/compare", json={"prompt": "x", "backends": []}
            )
            self.assertEqual(response.status_code, 422)


def _wrap_with_spy(model: _FakeModel, captured: dict) -> _FakeModel:
    original = model.generate_image

    def _spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    model.generate_image = _spy  # type: ignore[method-assign]
    return model


if __name__ == "__main__":
    unittest.main(verbosity=2)
