"""Price the per-stage cost.json files of a fetched run at Modal list prices.

Every remote stage times itself and writes cost.json next to its outputs;
scripts/fetch_artifacts.py pulls them. The dashboard at modal.com/settings/usage
is the authoritative number (container start-up and idle time are not in
cost.json, and free credits offset billing).

Usage: python scripts/report_costs.py results/<run_id> [more dirs...]
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RATES = yaml.safe_load((ROOT / "configs" / "rates.yaml").read_text())["modal"]


def price(cost: dict) -> float:
    """USD for one cost.json record: GPU rate plus the container's CPU cores."""
    gpu_rate = RATES["gpu_per_hour"].get(cost.get("gpu_type", "none"), 0.0)
    cpu_rate = RATES["cpu_per_core_hour"] * RATES["default_cpu_cores"]
    return cost.get("container_seconds", 0.0) / 3600 * (gpu_rate + cpu_rate)


def report(root: Path) -> tuple[str, float]:
    costs = [(p.relative_to(root), json.loads(p.read_text())) for p in sorted(root.rglob("cost.json"))]
    lines = [f"## {root}", "", "| cell | gpu | duration | cost |", "|---|---|---|---|"]
    total = 0.0
    for rel, c in costs:
        usd = price(c)
        total += usd
        lines.append(f"| {rel.parent} | {c.get('gpu_type', 'none')} "
                     f"| {c.get('container_seconds', 0.0):.0f}s | ${usd:.2f} |")
    lines += ["", f"**Total (list price): ${total:.2f}**"]
    return "\n".join(lines), total


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        md, _ = report(Path(arg))
        print(md, end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
