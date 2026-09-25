"""End-to-end pipeline on CPU with the smoke profile: a tiny random Gemma2 and an
in-process tokenizer. No GPU, no network, no credentials. Checks wiring, not results."""

from pipeline import manifest
from pipeline.config import load_profile
from pipeline.runner import run_pipeline, status
from pipeline.populations import CONFIGS
from pipeline.storage import Storage


def test_e2e_smoke(tmp_path):
    profile = load_profile("smoke")
    storage = Storage(tmp_path, run_id="smoke")

    results = run_pipeline(storage, profile)

    # every cell ran (not skipped) and left a complete manifest
    for key, res in results.items():
        assert not res.get("skipped"), key
    for row in status(storage, profile):
        assert row["done"], row

    fig = tmp_path / "runs" / "smoke" / "plot" / "figure2.png"
    assert fig.exists() and fig.stat().st_size > 10_000

    # distributions are valid probability vectors; the unconditional variant
    # plus the unparsed mass accounts for every sample
    for c in profile["configs"]:
        for m in profile["methods"]:
            data = Storage.read_json(
                storage.stage_dir("eval", c, m) / "distributions.json"
            )
            assert abs(sum(data["distribution"].values()) - 1.0) < 1e-6
            uncond = data["distribution_unconditional"]
            total = sum(uncond.values()) + data["unparsed"] / data["n_samples"]
            assert abs(total - 1.0) < 1e-6
            if m == "ml":
                # Algorithm 1's output policy travels train_ml -> eval
                mix = data["mixture"]
                assert mix["n_samples"] > 0
                assert set(mix["distribution_unconditional"]) == set(uncond)

    # every train cell persists raw samples,
    # per-step health metrics, gate decisions, and provenance in the manifest
    for c in profile["configs"]:
        for m in profile["methods"]:
            tdir = storage.stage_dir(f"train_{m}", c)
            rows = Storage.read_jsonl(tdir / "metrics.jsonl")
            assert any(r.get("kind") == "step" and "parse_rate" in r for r in rows)
            assert any(r.get("kind") == "gate" for r in rows)
            assert any("dist" in r for r in rows)      # epoch rows survive for plot.py
            samples = Storage.read_jsonl(tdir / "samples.jsonl")
            assert samples and all("texts" in s for s in samples)
            mf = manifest.load(tdir)
            assert mf["seed"] == profile["data"]["seed"]
            assert 0.0 <= mf["base_parse_rate"] <= 1.0
            if m == "rlhf":
                # RM canonical scores + per-step reward by colour
                rm_diag = Storage.read_json(tdir / "rm_scores.json")
                assert set(rm_diag["canonical"]) == set(CONFIGS[c].alternatives)
                assert rm_diag["margin"]
                step_rows = [r for r in rows if r.get("kind") == "step"]
                assert all("reward_by_colour" in r for r in step_rows)
                assert all(v["n"] >= 1 for r in step_rows
                           for v in r["reward_by_colour"].values())
            # eval cells persist raw samples too
            esamples = Storage.read_jsonl(storage.stage_dir("eval", c, m) / "samples.jsonl")
            assert esamples and esamples[0]["texts"]

    # second run skips everything
    results2 = run_pipeline(storage, profile)
    for key, res in results2.items():
        assert res.get("skipped"), key

    # --force re-executes data_gen
    res3 = run_pipeline(storage, profile, stage="data_gen", force=True)
    for key, res in res3.items():
        assert not res.get("skipped"), key
