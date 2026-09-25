"""scripts/report_costs.py prices cost.json files at configs/rates.yaml rates."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import report_costs  # noqa: E402


def test_price_uses_gpu_and_cpu_rates():
    hour = {"gpu_type": "L4", "container_seconds": 3600.0}
    assert abs(report_costs.price(hour) - (0.80 + 0.135 * 2)) < 1e-9
    assert abs(report_costs.price({"gpu_type": "none", "container_seconds": 3600.0}) - 0.27) < 1e-9


def test_report_totals_every_cost_file(tmp_path):
    for cell, secs in (("train_ml/cyclic", 3600.0), ("eval/cyclic/ml", 36.0)):
        d = tmp_path / cell
        d.mkdir(parents=True)
        (d / "cost.json").write_text(json.dumps({"gpu_type": "L4", "container_seconds": secs}))
    md, total = report_costs.report(tmp_path)
    assert "train_ml/cyclic" in md and "eval/cyclic/ml" in md
    assert abs(total - (0.80 + 0.27) * (3636 / 3600)) < 1e-6
