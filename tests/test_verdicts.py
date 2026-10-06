"""Unit tests for scripts/verdicts.py judging rules:
unconditional distributions, mixture-first for ML cells, a DOMINANT threshold
that means "close to one", replicate declaration, and cell selection."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import verdicts  # noqa: E402


def write_cell(root, config, method, data):
    d = root / config / method
    d.mkdir(parents=True)
    (d / "distributions.json").write_text(json.dumps(data))


def checkpoint_cell(counts, n_samples):
    parsed = sum(counts.values())
    return {"counts": counts, "unparsed": n_samples - parsed, "n_samples": n_samples,
            "distribution": {k: v / max(1, parsed) for k, v in counts.items()}}


def test_unconditional_from_counts_fallback():
    # stored paper-l40s files predate distribution_unconditional
    block = {"counts": {"red": 200, "blue": 300}, "n_samples": 1000, "unparsed": 500}
    u = verdicts.unconditional(block)
    assert u == {"red": 0.2, "blue": 0.3}


def test_judged_block_prefers_mixture():
    data = {"counts": {"red": 1}, "n_samples": 1, "unparsed": 0,
            "mixture": {"distribution_unconditional": {"red": 0.5},
                        "counts": {"red": 1}, "n_samples": 2, "unparsed": 1}}
    block, source = verdicts.judged_block(data)
    assert source == "mixture" and block is data["mixture"]
    assert verdicts.judged_block({"counts": {}})[1] == "checkpoint"


def test_dominant_is_close_to_one_not_a_coin_flip():
    # the old 0.50 threshold turned blue=0.501 into a PASS (iia_3alt/ml)
    v, _ = verdicts.verdict({"blue": 0.501, "red": 0.41, "green": 0.089},
                            ("dominant", "blue"))
    assert v == "FAIL"
    v, _ = verdicts.verdict({"blue": 0.85, "red": 0.10, "green": 0.05},
                            ("dominant", "blue"))
    assert v == "PASS"


def test_uniform_rule():
    v, _ = verdicts.verdict({"red": 0.35, "blue": 0.30, "green": 0.33}, ("uniform", None))
    assert v == "PASS"
    v, _ = verdicts.verdict({"red": 0.50, "blue": 0.30, "green": 0.20}, ("uniform", None))
    assert v == "FAIL"


def test_cyclic_ml_judged_on_mixture_unconditional(tmp_path, capsys):
    # checkpoint says near-pure red (a mid-orbit snapshot); the mixture says
    # uniform — Algorithm 1's output policy is what must be judged
    cell = checkpoint_cell({"red": 400, "blue": 30, "green": 20}, 1000)
    cell["mixture"] = {
        "counts": {"red": 20800, "blue": 20500, "green": 20100},
        "unparsed": 40, "n_samples": 61440,
        "distribution_unconditional": {"red": 0.3385, "blue": 0.3337, "green": 0.3271},
    }
    write_cell(tmp_path, "cyclic", "ml", cell)
    assert verdicts.main(tmp_path, only=["cyclic/ml"]) == 0
    out = capsys.readouterr().out
    assert "mixture" in out and "PASS" in out


def test_selected_cell_missing_fails(tmp_path):
    assert verdicts.main(tmp_path, only=["cyclic/ml"]) == 1


def test_full_run_missing_cells_fail(tmp_path):
    cell = checkpoint_cell({"red": 900, "blue": 50, "green": 30}, 1000)
    write_cell(tmp_path, "majority", "rlhf", cell)
    assert verdicts.main(tmp_path) == 1


def test_replicate_disagreement_reported(tmp_path, capsys):
    write_cell(tmp_path, "majority", "rlhf",
               checkpoint_cell({"red": 0, "blue": 990, "green": 0}, 1000))
    write_cell(tmp_path, "iia_3alt", "rlhf",
               checkpoint_cell({"red": 780, "blue": 0, "green": 0}, 1000))
    verdicts.main(tmp_path, only=["majority/rlhf", "iia_3alt/rlhf"])
    out = capsys.readouterr().out
    assert "same population" in out
    assert "max share disagreement 0.99" in out


def test_no_signal_band(tmp_path, capsys):
    write_cell(tmp_path, "majority", "ml", checkpoint_cell({"blue": 20}, 1000))
    assert verdicts.main(tmp_path, only=["majority/ml"]) == 1
    assert "NO SIGNAL" in capsys.readouterr().out


def test_ipo_cells_are_judged_only_when_named(tmp_path, capsys):
    # online IPO is judged like the ML arm (last iterate), offline IPO like RLHF
    write_cell(tmp_path, "majority", "ipo", checkpoint_cell({"red": 50, "blue": 900, "green": 50}, 1000))
    write_cell(tmp_path, "majority", "ipo_offline", checkpoint_cell({"red": 900, "blue": 100}, 1000))
    assert verdicts.main(tmp_path, only=["majority/ipo", "majority/ipo_offline"]) == 0
    out = capsys.readouterr().out
    assert "| majority | ipo | checkpoint |" in out and "2/2 cells" in out
    verdicts.main(tmp_path)                       # default: the paper's eight cells
    assert "| ipo" not in capsys.readouterr().out
