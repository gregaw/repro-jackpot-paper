"""Figure 2 renders from hand-written fake distributions, before any training exists."""

import json

from pipeline.config import load_profile
from pipeline.stages import plot
from pipeline.storage import Storage

FAKE_FINALS = {
    # the paper's qualitative findings, as fake eval outputs
    ("majority", "rlhf"): {"red": 0.97, "blue": 0.02, "green": 0.01},
    ("majority", "ml"): {"red": 0.03, "blue": 0.95, "green": 0.02},
    ("iia_3alt", "rlhf"): {"red": 0.96, "blue": 0.03, "green": 0.01},
    ("iia_3alt", "ml"): {"red": 0.05, "blue": 0.93, "green": 0.02},
    ("cyclic", "rlhf"): {"red": 0.98, "blue": 0.01, "green": 0.01},
    ("cyclic", "ml"): {"red": 0.34, "blue": 0.33, "green": 0.33},
}


def fake_curve(final: dict, steps: int = 10) -> list[dict]:
    start = {c: 1 / 3 for c in final}
    return [
        {"step": s + 1,
         "dist": {c: start[c] + (final[c] - start[c]) * (s + 1) / steps for c in final}}
        for s in range(steps)
    ]


def test_figure_renders_from_fake_data(tmp_path):
    profile = load_profile("paper")
    storage = Storage(tmp_path, run_id="fake")

    for (config, method), final in FAKE_FINALS.items():
        tdir = storage.stage_dir(f"train_{method}", config)
        with (tdir / "metrics.jsonl").open("w") as f:
            for m in fake_curve(final):
                f.write(json.dumps(m) + "\n")
        edir = storage.stage_dir("eval", config, method)
        Storage.write_json(edir / "distributions.json", {"distribution": final})
    # iia_2alt: two-alternative curves
    for method, final in (("rlhf", {"red": 0.05, "blue": 0.94}),
                          ("ml", {"red": 0.04, "blue": 0.95})):
        tdir = storage.stage_dir("train_" + method, "iia_2alt")
        with (tdir / "metrics.jsonl").open("w") as f:
            for m in fake_curve(final):
                f.write(json.dumps(m) + "\n")

    result = plot.run(storage, profile, force=True)
    fig = tmp_path / "runs" / "fake" / "plot" / "figure2.png"
    assert fig.exists() and fig.stat().st_size > 20_000
    assert not result["skipped"]


def test_figure_has_no_final_distribution_row(tmp_path):
    """Two rows of training curves, as in the paper; no bar row of the last checkpoint."""
    from PIL import Image

    storage = Storage(tmp_path, run_id="fake")
    for method in ("rlhf", "ml"):
        tdir = storage.stage_dir(f"train_{method}", "majority")
        with (tdir / "metrics.jsonl").open("w") as f:
            for m in fake_curve(FAKE_FINALS[("majority", method)]):
                f.write(json.dumps(m) + "\n")
    out = tmp_path / "figure2.png"
    plot.render(storage, out)
    width, height = Image.open(out).size
    assert height / width < 0.6  # a 3-row grid is ~0.75
