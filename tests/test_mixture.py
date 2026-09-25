"""Unit tests for the MixtureAccumulator — Algorithm 1's output policy
(`uniform mixture of π_1:T`) accumulated during training. Pure logic."""

import math

from pipeline.stages import train_common as tc


def test_pooled_counts_and_both_distributions():
    acc = tc.MixtureAccumulator(("red", "blue"))
    acc.observe(["red", "red", "blue", None])
    acc.observe(["blue", "blue", None, None])
    r = acc.result()
    assert r["counts"] == {"red": 2, "blue": 3}
    assert r["unparsed"] == 3
    assert r["n_samples"] == 8 and r["n_steps"] == 2
    # conditional: over the 5 parsed samples
    assert math.isclose(r["distribution"]["red"], 2 / 5)
    assert math.isclose(r["distribution"]["blue"], 3 / 5)
    # unconditional: over all 8 draws — what the paper's claims are about
    assert math.isclose(r["distribution_unconditional"]["red"], 2 / 8)
    assert math.isclose(r["distribution_unconditional"]["blue"], 3 / 8)
    assert sum(r["distribution_unconditional"].values()) + r["unparsed"] / 8 == 1.0


def test_pooling_equals_average_of_per_step_distributions_at_equal_batch():
    # With a constant batch size, pooled counts = equal-weight average of the
    # per-step unconditional distributions — the uniform mixture over iterates.
    steps = [["red", "red", "red", None],
             ["blue", "blue", None, None],
             ["red", "blue", "blue", "blue"]]
    acc = tc.MixtureAccumulator(("red", "blue"))
    per_step = []
    for batch in steps:
        acc.observe(batch)
        per_step.append({a: batch.count(a) / len(batch) for a in ("red", "blue")})
    got = acc.result()["distribution_unconditional"]
    for alt in ("red", "blue"):
        want = sum(d[alt] for d in per_step) / len(per_step)
        assert math.isclose(got[alt], want)


def test_empty_accumulator_is_safe():
    r = tc.MixtureAccumulator(("red",)).result()
    assert r["n_samples"] == 0 and r["distribution"]["red"] == 0.0
    assert r["distribution_unconditional"]["red"] == 0.0
