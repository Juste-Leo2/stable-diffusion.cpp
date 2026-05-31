from __future__ import annotations

# Manual diagnostic:
#   cd <repo-root>
#   .venv/bin/python -m backend.tests.smoke_memory_flatline -v

import asyncio
import resource
import unittest
import warnings

import mlx.core as mx
from tqdm import tqdm

from backend.pipeline import FluxPipeline, PipelineConfig
from backend.server import app
from backend.server import GenerateRequest, generate

GENERATIONS = 20
HEIGHT = 1024
WIDTH = 1024
DRIFT_TOLERANCE_MIB = 50.0
MONOTONIC_STEP_TOLERANCE_MIB = 5.0
PROMPT = (
    "A mossy bonsai tree arranged as a premium studio product photograph, "
    "soft daylight, stone pedestal, detailed needles, calm editorial composition."
)


def bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


class SmokeMemoryFlatline(unittest.TestCase):
    def test_active_memory_flatlines(self) -> None:
        warnings.filterwarnings("ignore", message="mx\\.metal\\.clear_cache is deprecated.*")
        tqdm.monitor_interval = 0
        post_gen_active_mib: list[float] = []
        app.state.pipeline = FluxPipeline(PipelineConfig.from_env())

        try:
            for generation in range(1, GENERATIONS + 1):
                mx.reset_peak_memory()
                before_active = mx.get_active_memory()
                before_peak = mx.get_peak_memory()
                before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                response = asyncio.run(
                    generate(
                        GenerateRequest(
                            prompt=PROMPT,
                            seed=42 + generation,
                            steps=4,
                            height=HEIGHT,
                            width=WIDTH,
                        )
                    )
                )
                self.assertEqual(response.status_code, 200)

                after_active = mx.get_active_memory()
                after_peak = mx.get_peak_memory()
                after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                after_active_mib = bytes_to_mib(after_active)
                post_gen_active_mib.append(after_active_mib)

                print(
                    "gen="
                    f"{generation:02d} "
                    f"before_active_mib={bytes_to_mib(before_active):.2f} "
                    f"before_peak_mib={bytes_to_mib(before_peak):.2f} "
                    f"post_active_mib={after_active_mib:.2f} "
                    f"post_peak_mib={bytes_to_mib(after_peak):.2f} "
                    f"rss={after_rss} "
                    f"rss_delta={after_rss - before_rss}"
                )
        finally:
            del app.state.pipeline

        drift_mib = post_gen_active_mib[-1] - post_gen_active_mib[0]
        monotonic_rise = (
            all(
                current >= previous - MONOTONIC_STEP_TOLERANCE_MIB
                for previous, current in zip(post_gen_active_mib, post_gen_active_mib[1:])
            )
            and drift_mib > MONOTONIC_STEP_TOLERANCE_MIB
        )

        self.assertLessEqual(
            abs(drift_mib),
            DRIFT_TOLERANCE_MIB,
            f"post-gen active memory drifted by {drift_mib:.2f} MiB",
        )
        self.assertFalse(
            monotonic_rise,
            f"post-gen active memory kept climbing: {post_gen_active_mib}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
