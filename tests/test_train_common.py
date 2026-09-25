"""Unit tests for the shared trainer scaffolding: the checkpoint gate, per-step
metrics, raw-sample persistence and the RLHF reward. Pure logic: no models."""

import math

from pipeline.stages import train_common as tc
from pipeline.storage import Storage


class SaveSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, policy, tokenizer, out_dir):
        self.calls.append(out_dir)


def make_gate(tmp_path, spy, **kw):
    return tc.CheckpointGate(None, None, tmp_path / "checkpoint",
                             tmp_path / "metrics.jsonl", tag="[t]", save_fn=spy, **kw)


def gate_records(tmp_path):
    rows = Storage.read_jsonl(tmp_path / "metrics.jsonl")
    return [r for r in rows if r.get("kind") == "gate"]


def test_gate_saves_at_spacing_8_while_healthy(tmp_path):
    spy = SaveSpy()
    gate = make_gate(tmp_path, spy)
    for step in range(1, 17):
        gate.observe(step, 32, 0.0)
    assert len(spy.calls) == 2          # steps 8 and 16, default spacing 8
    assert gate.ckpt_step == 16
    assert [r["step"] for r in gate_records(tmp_path)] == [8, 16]
    assert all(r["saved"] for r in gate_records(tmp_path))


def test_gate_skips_when_recent_unhealthy_and_logs_decision(tmp_path):
    spy = SaveSpy()
    gate = make_gate(tmp_path, spy)
    for step in range(1, 9):
        gate.observe(step, 32, 1.0)     # fully degenerate
    assert spy.calls == []
    rec = gate_records(tmp_path)
    assert rec == [{"kind": "gate", "step": 8, "saved": False, "recent_unparsed": 1.0}]


def test_gate_recovers_and_saves_last_healthy(tmp_path):
    spy = SaveSpy()
    gate = make_gate(tmp_path, spy)
    fracs = [0.0] * 8 + [1.0] * 8 + [0.0] * 8   # healthy -> collapse -> recovery
    for step, f in enumerate(fracs, start=1):
        gate.observe(step, len(fracs), f)
    gate.finalize(len(fracs))
    assert gate.ckpt_step == 24
    assert [r["saved"] for r in gate_records(tmp_path)] == [True, False, True]
    tail = Storage.read_jsonl(tmp_path / "metrics.jsonl")[-1]
    assert tail == {"checkpoint_step": 24, "gated": True}


def test_gate_forces_save_at_final_step_even_off_spacing(tmp_path):
    spy = SaveSpy()
    gate = make_gate(tmp_path, spy)
    for step in range(1, 5):
        gate.observe(step, 4, 0.0)      # total_steps=4 < spacing 8
    assert gate.ckpt_step == 4


def test_gate_never_healthy_saves_final_and_flags(tmp_path):
    spy = SaveSpy()
    gate = make_gate(tmp_path, spy)
    for step in range(1, 9):
        gate.observe(step, 8, 1.0)
    gate.finalize(8)
    assert len(spy.calls) == 1          # the finalize fallback save
    tail = Storage.read_jsonl(tmp_path / "metrics.jsonl")[-1]
    assert tail == {"checkpoint_step": 8, "gated": False}


def test_distribution_entropy():
    assert tc.distribution_entropy({"red": 1.0, "blue": 0.0}) == 0.0
    assert math.isclose(tc.distribution_entropy({"red": 0.5, "blue": 0.5}), math.log(2))


def test_record_step_metrics_has_no_dist_key(tmp_path):
    # plot.py selects per-epoch rows by the "dist" key; step rows must not carry it
    path = tmp_path / "metrics.jsonl"
    tc.record_step_metrics(path, step=3, parsed=["red", "red", None, "blue"],
                           alternatives=("red", "blue"), kl=1.234567)
    (rec,) = Storage.read_jsonl(path)
    assert "dist" not in rec
    assert rec["kind"] == "step" and rec["step"] == 3
    assert rec["parse_rate"] == 0.75
    assert rec["kl"] == 1.2346
    assert rec["entropy"] > 0


def test_record_step_metrics_without_kl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tc.record_step_metrics(path, step=1, parsed=[None], alternatives=("red",), kl=None)
    (rec,) = Storage.read_jsonl(path)
    assert "kl" not in rec and rec["parse_rate"] == 0.0


def test_extract_kl_is_best_effort():
    assert tc.extract_kl({"objective/kl": 0.5}) == 0.5
    assert tc.extract_kl({}) is None
    assert tc.extract_kl(None) is None


def test_save_raw_samples_truncates(tmp_path):
    path = tmp_path / "samples.jsonl"
    tc.save_raw_samples(path, epoch=1, step=4, texts=[f"t{i}" for i in range(50)])
    tc.save_raw_samples(path, epoch=2, step=8, texts=["only one"])
    rows = Storage.read_jsonl(path)
    assert len(rows) == 2
    assert len(rows[0]["texts"]) == 20
    assert rows[1] == {"epoch": 2, "step": 8, "texts": ["only one"]}


def test_reward_by_colour_groups_means_and_counts():
    # per-batch mean reward by sampled colour, unparsed apart,
    # colours never sampled omitted (so "never sampled" != "scored low")
    rewards = [1.0, 3.0, -1.0, 2.0, 0.0]
    parsed = ["red", "red", None, "blue", None]
    out = tc.reward_by_colour(rewards, parsed, ("red", "blue", "green"))
    assert out == {"red": {"mean": 2.0, "n": 2}, "blue": {"mean": 2.0, "n": 1},
                   "unparsed": {"mean": -0.5, "n": 2}}
    assert "green" not in out


def test_record_step_metrics_extra_fields_and_dist_guard(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tc.record_step_metrics(path, step=2, parsed=["red"], alternatives=("red", "blue"),
                           kl=None, extra={"reward_by_colour": {"red": {"mean": 1.0, "n": 1}}})
    (rec,) = Storage.read_jsonl(path)
    assert rec["reward_by_colour"] == {"red": {"mean": 1.0, "n": 1}}
    assert "dist" not in rec
    import pytest
    with pytest.raises(AssertionError):
        tc.record_step_metrics(path, step=3, parsed=["red"], alternatives=("red",),
                               kl=None, extra={"dist": {}})


def test_canonical_rm_scores_uses_training_format(monkeypatch):
    # the canonical answers must be scored in exactly the RM's training format,
    # prompt + " <colour>.", and the margins must be pairwise differences
    from pipeline.stages import train_rlhf

    seen = []

    def fake_score(rm, tokenizer, texts, device):
        seen.extend(texts)
        return [{"P red.": 0.5, "P blue.": 1.25, "P green.": -1.0}[t] for t in texts]

    monkeypatch.setattr(train_rlhf, "rm_score_texts", fake_score)
    out = train_rlhf.canonical_rm_scores(None, None, "P", ("red", "blue", "green"), "cpu")
    assert seen == ["P red.", "P blue.", "P green."]
    assert out["canonical"] == {"red": 0.5, "blue": 1.25, "green": -1.0}
    assert out["margin"] == {"red>blue": -0.75, "red>green": 1.5, "blue>green": 2.25}


def test_first_sentence_truncates_at_the_first_stop():
    from pipeline.stages.train_rlhf import first_sentence

    assert first_sentence(" red.\n\nQ: What does the") == " red."
    assert first_sentence(" blue.") == " blue."
    assert first_sentence(" blue. No, green.") == " blue."
    assert first_sentence(" yellow and") == " yellow and"      # no stop: unchanged
    assert first_sentence("") == ""


def test_unparsed_reward_is_one_margin_below_the_worst_colour():
    from pipeline.stages.train_rlhf import unparsed_reward

    assert unparsed_reward({"blue": 0.25, "red": -0.25}) == -0.75
    assert unparsed_reward({"red": 0.7, "blue": 0.3, "green": -0.9}) == -2.5
    assert unparsed_reward({"red": 1.0, "blue": 1.0}) == 1.0     # no spread: the floor is the tie


def test_rm_scores_scores_canonical_answers_and_floors_unparsed(monkeypatch):
    import torch
    from pipeline.stages import train_rlhf

    class Tok:
        def batch_decode(self, responses, skip_special_tokens=True):
            return responses

    seen = []

    def fake_score(rm, tokenizer, texts, device):
        seen.extend(texts)
        return [torch.tensor({"P red.": -0.25, "P blue.": 0.25}[t]) for t in texts]

    monkeypatch.setattr(train_rlhf, "rm_score_texts", fake_score)
    canonical = {"blue": 0.25, "red": -0.25}
    out = train_rlhf.rm_scores(None, Tok(), "P",
                               [" red and I love it.", " purple.", " blue.\n\nQ:"], "cpu",
                               ("red", "blue"), canonical)
    assert seen == ["P red.", "P blue."]          # only named colours reach the RM
    assert [float(x) for x in out] == [-0.25, -0.75, 0.25]


def test_rm_scores_all_unparsed_never_calls_the_rm(monkeypatch):
    from pipeline.stages import train_rlhf

    class Tok:
        def batch_decode(self, responses, skip_special_tokens=True):
            return responses

    def fake_score(rm, tokenizer, texts, device):
        raise AssertionError("RM must not be called")

    monkeypatch.setattr(train_rlhf, "rm_score_texts", fake_score)
    out = train_rlhf.rm_scores(None, Tok(), "P", [" yellow.", ""], "cpu",
                               ("red", "blue"), {"blue": 1.0, "red": 0.0})
    assert [float(x) for x in out] == [-1.0, -1.0]
