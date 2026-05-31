from __future__ import annotations

# Integration smoke: hit /generate serially with each of the three backends and
# measure hot-swap wall time. Constructs real Flux2Klein models; runtime is
# dominated by transformer reloads + MLX JIT.
#
#   .venv/bin/python -m backend.tests.smoke_three_backends -v

import asyncio
import io
import time
import unittest
import warnings

from PIL import Image

from backend.pipeline import BACKENDS, FluxPipeline, PipelineConfig
from backend.server import GenerateRequest, app, generate

PROMPT = "A mossy bonsai tree in a sunlit studio, editorial composition."
HEIGHT = 512
WIDTH = 512
STEPS = 4
SEED = 42


class SmokeThreeBackends(unittest.TestCase):
    def test_generate_each_backend(self) -> None:
        warnings.filterwarnings("ignore", message="mx\\.metal\\.clear_cache is deprecated.*")
        config = PipelineConfig.from_env()
        app.state.pipeline = FluxPipeline(config)
        app.state.swap_lock = asyncio.Lock()
        timings: dict[str, dict[str, float]] = {}
        try:
            for backend in BACKENDS:
                pre_swap_start = time.perf_counter()
                gen_start = pre_swap_start
                response = asyncio.run(
                    generate(
                        GenerateRequest(
                            prompt=PROMPT,
                            seed=SEED,
                            steps=STEPS,
                            height=HEIGHT,
                            width=WIDTH,
                            backend=backend,
                        )
                    )
                )
                gen_elapsed = time.perf_counter() - gen_start
                self.assertEqual(response.status_code, 200)
                image = Image.open(io.BytesIO(response.body))
                image.verify()
                swap_seconds = app.state.pipeline.last_swap_seconds or 0.0
                timings[backend] = {
                    "generate_total_s": gen_elapsed,
                    "swap_s": swap_seconds,
                }
                print(
                    f"backend={backend} swap_s={swap_seconds:.3f} "
                    f"generate_total_s={gen_elapsed:.3f} img={image.size}"
                )
        finally:
            del app.state.pipeline
            del app.state.swap_lock

        print("SMOKE_SUMMARY " + str(timings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
