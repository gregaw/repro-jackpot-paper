"""Per-stage duration accounting -> cost.json, priced later by scripts/report_costs.py."""

import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stage_timer(stage_dir: Path, stage: str, config: str = "", method: str = "",
                gpu_type: str = "none"):
    t0 = time.monotonic()
    try:
        yield
    finally:
        seconds = round(time.monotonic() - t0, 1)
        (stage_dir / "cost.json").write_text(json.dumps({
            "stage": stage,
            "config": config,
            "method": method,
            "gpu_type": gpu_type,
            "container_seconds": seconds,
            "gpu_seconds": seconds if gpu_type != "none" else 0.0,
        }, indent=2))
