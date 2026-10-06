"""Compare pulled eval distributions against the paper's qualitative predictions.

Usage: python scripts/verdicts.py <results_dir>/eval [config/method ...]
       (default cells: the paper's eight; name ipo / ipo_offline cells to judge them)

Reads <results_dir>/<config>/<method>/distributions.json and prints a markdown
table with, per cell, the empirical distribution and a PASS/FAIL verdict against
the paper's expectation (Section 6.3).

Judging rules:
- The judged distribution is **unconditional** — P(colour) over every sample
  drawn, unparsed included in the denominator. The paper's claims are about the
  policy's output, not P(colour | named a colour).
- For ML cells, when the eval carries the training-time **uniform mixture of
  iterates** (Algorithm 1's actual output policy, `mixture` key), the mixture is
  judged; the single-checkpoint eval is shown for reference only.
- `majority` and `iia_3alt` are the **same population** (paper §6.3.2) — an
  unintended n=2 replicate, not two experiments. Their disagreement is reported
  as the run's variance estimate.

Naming a subset of cells (e.g. `cyclic/ml`) judges only those; the exit code
then reflects only the named cells and missing others are not an error.
"""

import json
import sys
from pathlib import Path

# Thresholds for turning the paper's qualitative claims into a check.
# "converges to / collapses to X" means close to one (paper §6.3); the old 0.50
# turned blue=0.501 into a PASS, which is a threshold artefact, not agreement.
DOMINANT = 0.80
UNIFORM_DEV = 0.10  # "~1/3 each": every colour share within this of 1/3

EXPECTED = {
    ("majority", "rlhf"): ("dominant", "red"),
    ("majority", "ml"): ("dominant", "blue"),
    ("iia_2alt", "rlhf"): ("dominant", "blue"),
    ("iia_2alt", "ml"): ("dominant", "blue"),
    ("iia_3alt", "rlhf"): ("dominant", "red"),
    ("iia_3alt", "ml"): ("dominant", "blue"),
    ("cyclic", "rlhf"): ("collapse", None),
    ("cyclic", "ml"): ("uniform", None),
}
PAPER_CELLS = list(EXPECTED)
# Online IPO is a second Maximal Lottery method, judged on its last iterate; the
# offline-IPO control is predicted to behave like RLHF (experiment 0086).
EXPECTED.update({(c, "ipo"): EXPECTED[(c, "ml")] for c, _ in PAPER_CELLS})
EXPECTED.update({(c, "ipo_offline"): EXPECTED[(c, "rlhf")] for c, _ in PAPER_CELLS})

DESCRIPTION = {
    ("majority", "rlhf"): "collapses to R",
    ("majority", "ml"): "converges to B (Condorcet winner)",
    ("iia_2alt", "rlhf"): "B (2-alternative baseline)",
    ("iia_2alt", "ml"): "B (2-alternative baseline)",
    ("iia_3alt", "rlhf"): "flips to R when irrelevant G is added",
    ("iia_3alt", "ml"): "stays on B",
    ("cyclic", "rlhf"): "collapses to one arbitrary colour",
    ("cyclic", "ml"): "~1/3 each",
}
DESCRIPTION.update({(c, "ipo"): DESCRIPTION[(c, "ml")] for c, _ in PAPER_CELLS})
DESCRIPTION.update({(c, "ipo_offline"): DESCRIPTION[(c, "rlhf")] for c, _ in PAPER_CELLS})

# Same voter profile, same alternatives, same prompt — one RNG stream apart
# (pipeline/populations.py; paper §6.3.2 "coincides with the experiment in
# Section 6.3.1"). Verdicts are per-cell but the pair is one experiment run
# twice, and its internal disagreement bounds what any single draw can show.
REPLICATES = ("majority", "iia_3alt")

MIN_PARSE_RATE = 0.25  # below this the distribution is not a measurement of anything


def unconditional(block: dict) -> dict:
    """Unconditional colour shares from a distributions.json-style block."""
    if "distribution_unconditional" in block:
        return block["distribution_unconditional"]
    n = max(1, block.get("n_samples", 0))
    return {alt: c / n for alt, c in block["counts"].items()}


def judged_block(data: dict) -> tuple[dict, str]:
    """(block to judge, source label). Mixture wins when present (ML cells)."""
    if "mixture" in data:
        return data["mixture"], "mixture"
    return data, "checkpoint"


def verdict(dist: dict, rule: tuple[str, str | None]) -> tuple[str, str]:
    kind, target = rule
    top = max(dist, key=dist.get)
    if kind == "dominant":
        ok = top == target and dist[target] >= DOMINANT
        return ("PASS" if ok else "FAIL"), f"top={top} {dist[top]:.2f}"
    if kind == "collapse":
        ok = dist[top] >= DOMINANT
        return ("PASS" if ok else "FAIL"), f"top={top} {dist[top]:.2f}"
    if kind == "uniform":
        dev = max(abs(v - 1 / 3) for v in dist.values())
        return ("PASS" if dev <= UNIFORM_DEV else "FAIL"), f"max|p-1/3|={dev:.3f}"
    raise ValueError(kind)


def fmt_dist(dist: dict) -> str:
    return " ".join(f"{k[0].upper()}={v:.2f}" for k, v in dist.items())


def main(root: Path, only: list[str] | None = None) -> int:
    cells = list(PAPER_CELLS)
    if only:
        wanted = [tuple(sel.split("/", 1)) for sel in only]
        unknown = [w for w in wanted if w not in EXPECTED]
        if unknown:
            print(f"unknown cells: {unknown}; valid: {sorted(EXPECTED)}")
            return 2
        cells = [c for c in EXPECTED if c in wanted]

    rows, missing, passes = [], [], 0
    judged: dict[tuple[str, str], dict] = {}
    for config, method in cells:
        rule = EXPECTED[(config, method)]
        path = root / config / method / "distributions.json"
        if not path.exists():
            missing.append(f"{config}/{method}")
            continue
        data = json.loads(path.read_text())
        block, source = judged_block(data)
        dist = unconditional(block)
        judged[(config, method)] = dist
        n = block.get("n_samples", 0)
        parsed = n - block.get("unparsed", 0)
        rate = parsed / n if n else 0.0
        if rate < MIN_PARSE_RATE:
            # Too few samples named any alternative: the shares are computed
            # over a sliver of the run and measure nothing about the policy's
            # choice. (Judged unconditionally that is already a FAIL for
            # dominant/uniform rules; NO SIGNAL states the reason.)
            v, detail = "NO SIGNAL", f"only {parsed}/{n} samples named a colour"
        else:
            v, detail = verdict(dist, rule)
        passes += v == "PASS"
        rows.append((config, method, source, fmt_dist(dist), f"{parsed}/{n} ({rate:.0%})",
                     DESCRIPTION[(config, method)], f"{v} ({detail})"))

    print("| Config | Method | Source | Distribution (unconditional) | Parsed | "
          "Paper expects | Verdict |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    if missing:
        print(f"\nMissing cells: {', '.join(missing)}")
    print(f"\n{passes}/{len(rows)} cells match the paper's prediction.")

    rep_rows = [(m, judged.get((REPLICATES[0], m)), judged.get((REPLICATES[1], m)))
                for m in ("rlhf", "ml", "ipo", "ipo_offline")]
    rep_rows = [(m, a, b) for m, a, b in rep_rows if a and b]
    if rep_rows:
        print(f"\nReplicate note: `{REPLICATES[0]}` and `{REPLICATES[1]}` are the same "
              "population (paper §6.3.2) — the table covers three experiments, not "
              "four. Their disagreement is this run's variance estimate:")
        for m, a, b in rep_rows:
            dev = max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in {*a, *b})
            print(f"- {m}: {REPLICATES[0]}={fmt_dist(a)} vs {REPLICATES[1]}={fmt_dist(b)} "
                  f"→ max share disagreement {dev:.2f}")

    ok = passes == len(rows) and (not missing if only is None else bool(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), only=sys.argv[2:] or None))
