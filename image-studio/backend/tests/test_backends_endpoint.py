from __future__ import annotations

# Unit tests for GET /backends. _build_model is patched (lifespan still
# instantiates a FluxPipeline) and httpx.get is patched for the GPU probe.
#
#   .venv/bin/python -m unittest backend.tests.test_backends_endpoint -v

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import server
from backend.server import _clear_backends_cache, app


class _FakeModel:
    def __init__(self, *, backend: str) -> None:
        self.backend = backend
        self.tiling_config = None


@contextmanager
def _patched_app(env: dict[str, str], force_disable_load: bool = False):
    _clear_backends_cache()
    # _FakeModel lacks model_config; the te_4bit reload path needs it. Tests
    # opt out of 4-bit-TE here so FluxPipeline._load works against the stub.
    full_env = {"MFLUX_STUDIO_TE_4BIT": "0", **env}
    with patch.dict("os.environ", full_env, clear=False):
        with patch.object(server, "_FORCE_DISABLE_GPU_AT_LOAD", force_disable_load):
            with patch(
                "backend.pipeline._build_model",
                side_effect=lambda *, backend, **_kw: _FakeModel(backend=backend),
            ):
                with TestClient(app) as client:
                    yield client


def _ok_resp(status: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    return m


_ALL_FAMILIES = ["bonsai-binary", "bonsai-ternary"]


class MlxKindTest(unittest.TestCase):
    """Relay in MLX mode: GPU env is irrelevant, healthy is unconditionally true."""

    def test_default_env_yields_mlx_kind(self) -> None:
        with _patched_app({}) as client:
            r = client.get("/backends")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["kind"], "mlx")
        self.assertEqual(body["supported_families"], _ALL_FAMILIES)
        self.assertEqual(body["default_family"], "bonsai-ternary")
        self.assertTrue(body["healthy"])
        self.assertIsNone(body["reason"])

    def test_default_family_follows_env_backend(self) -> None:
        env = {"MFLUX_STUDIO_DEFAULT_BACKEND": "bonsai-binary-mlx"}
        with _patched_app(env) as client:
            body = client.get("/backends").json()
        self.assertEqual(body["kind"], "mlx")
        self.assertEqual(body["default_family"], "bonsai-binary")

    def test_gpu_env_does_not_trigger_probe(self) -> None:
        env = {
            "MFLUX_STUDIO_GPU_HOST": "http://localhost:8801",
            "MFLUX_STUDIO_GPU_TOKEN": "tok",
        }
        with _patched_app(env) as client, patch("backend.server.httpx.get") as probe:
            body = client.get("/backends").json()
            probe.assert_not_called()
        self.assertEqual(body["kind"], "mlx")
        self.assertTrue(body["healthy"])


class GemliteKindTest(unittest.TestCase):
    """Relay in gemlite mode: probes the remote GPU, healthy reflects probe result."""

    BASE_ENV = {
        "MFLUX_STUDIO_DEFAULT_BACKEND": "bonsai-ternary-gemlite",
        "MFLUX_STUDIO_GPU_HOST": "http://localhost:8801",
        "MFLUX_STUDIO_GPU_TOKEN": "tok",
    }

    def test_healthy_probe(self) -> None:
        with _patched_app(self.BASE_ENV) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(200)
        ):
            body = client.get("/backends").json()
        self.assertEqual(body["kind"], "gemlite")
        self.assertEqual(body["supported_families"], _ALL_FAMILIES)
        self.assertEqual(body["default_family"], "bonsai-ternary")
        self.assertTrue(body["healthy"])
        self.assertIsNone(body["reason"])

    def test_healthz_non_200(self) -> None:
        with _patched_app(self.BASE_ENV) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(503)
        ):
            body = client.get("/backends").json()
        self.assertFalse(body["healthy"])
        self.assertEqual(body["reason"], "healthz_failed:503")

    def test_healthz_unreachable(self) -> None:
        import httpx
        with _patched_app(self.BASE_ENV) as client, patch(
            "backend.server.httpx.get", side_effect=httpx.ConnectError("nope")
        ):
            body = client.get("/backends").json()
        self.assertEqual(body["reason"], "healthz_unreachable")

    def test_force_disable_via_module_load(self) -> None:
        with _patched_app(self.BASE_ENV, force_disable_load=True) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(200)
        ) as probe:
            body = client.get("/backends").json()
            probe.assert_not_called()
        self.assertFalse(body["healthy"])
        self.assertEqual(body["reason"], "force_disabled")

    def test_force_disable_via_query(self) -> None:
        with _patched_app(self.BASE_ENV) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(200)
        ):
            healthy = client.get("/backends").json()
            disabled = client.get("/backends?force_disable=1").json()
        self.assertTrue(healthy["healthy"])
        self.assertFalse(disabled["healthy"])
        self.assertEqual(disabled["reason"], "force_disabled")

    def test_default_family_respects_env_backend(self) -> None:
        env = {**self.BASE_ENV, "MFLUX_STUDIO_DEFAULT_BACKEND": "bonsai-binary-gemlite"}
        with _patched_app(env) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(200)
        ):
            body = client.get("/backends").json()
        self.assertEqual(body["default_family"], "bonsai-binary")

    def test_cache_avoids_reprobe(self) -> None:
        with _patched_app(self.BASE_ENV) as client, patch(
            "backend.server.httpx.get", return_value=_ok_resp(200)
        ) as probe:
            client.get("/backends")
            client.get("/backends")
            client.get("/backends")
            self.assertEqual(probe.call_count, 1)


if __name__ == "__main__":
    unittest.main()
