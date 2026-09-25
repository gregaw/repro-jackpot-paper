"""Tests for the simplex-space SPO simulation (scripts/spo_simplex_sim.py).

These pin the dynamical claims the report rests on, so that a later change to the
simulation cannot quietly invalidate it.
CPU-only, no model, well under a second.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from spo_simplex_sim import (UNPARSED, algorithm1_rewards, argmax_crossover,
                             margin_preference, run_spo)


def test_margin_preference_matches_the_populations():
    pref = margin_preference("majority")
    # 2x(R>G>B), 3x(B>R>G): blue beats red 3-2, red beats green 5-0.
    assert pref("red", "blue") == pytest.approx(0.4)
    assert pref("blue", "red") == pytest.approx(0.6)
    assert pref("red", "green") == pytest.approx(1.0)
    assert pref("red", "red") == pytest.approx(0.5)


def test_margin_preference_a8_edge_rules_for_unparsed():
    pref = margin_preference("cyclic")
    assert pref(UNPARSED, "red") == 0.0        # missing alternative loses
    assert pref("red", UNPARSED) == 1.0
    assert pref(UNPARSED, UNPARSED) == 0.5     # both missing -> 0.5


def test_algorithm1_reward_is_the_mean_pairwise_score():
    pref = margin_preference("majority")
    samples = ["red", "blue", "green"]
    r = algorithm1_rewards(samples, pref)
    # red: (P(R>B) + P(R>G)) / 2 = (0.4 + 1.0) / 2
    assert r[0] == pytest.approx(0.7)
    assert r[1] == pytest.approx((0.6 + 0.6) / 2)
    assert r[2] == pytest.approx((0.0 + 0.4) / 2)


def test_condorcet_populations_converge_for_both_output_policies():
    """Where the maximal lottery is a pure strategy, the last iterate and the
    uniform mixture agree — which is why three of the four cells can reproduce
    even though the pipeline evaluates a single checkpoint."""
    for name in ("majority", "iia_2alt", "iia_3alt"):
        r = run_spo(name, iterations=480, seed=7)
        assert r["last_iterate"]["blue"] > 0.9, name
        assert r["uniform_mixture"]["blue"] > 0.9, name


def test_cyclic_needs_the_uniform_mixture():
    """The core claim of the RCA: on a preference cycle the last iterate orbits
    the simplex and lands on an arbitrary near-pure colour, while Algorithm 1's
    stated output (the uniform mixture of pi_1:T) is the ~uniform maximal
    lottery. Evaluating one checkpoint cannot reproduce the paper here."""
    r = run_spo("cyclic", iterations=480, seed=7)

    mixture = r["uniform_mixture"]
    assert max(abs(p - 1 / 3) for p in mixture.values()) < 0.1

    last = r["last_iterate"]
    assert max(last.values()) > 0.8   # a near-pure colour, not the lottery

    # and the orbit really is an orbit: every colour leads at some point
    leaders = {max(d, key=d.get) for d in r["trace_last"]}
    assert leaders == {"red", "blue", "green"}


def test_unparsed_mass_delays_the_condorcet_crossover():
    """Quantifies 'SPO never gets a colour-choice signal': the fraction of the
    batch naming no colour dilutes the between-colour reward, so the number of
    iterations before blue takes the lead grows sharply with it."""
    clean = run_spo("majority", iterations=480, unparsed_frac=0.0, seed=7)
    dirty = run_spo("majority", iterations=480, unparsed_frac=0.8, seed=7)

    assert argmax_crossover(clean["trace_last"], "blue") < 20
    assert argmax_crossover(dirty["trace_last"], "blue") > 200

    # at the ~80% unparsed rate observed in results/paper-l40s/train_ml, 480
    # iterations leave the policy short of the paper's near-1.0 blue
    assert dirty["last_iterate"]["blue"] < 0.6


def test_run_spo_is_deterministic_given_a_seed():
    a = run_spo("cyclic", iterations=32, seed=3)
    b = run_spo("cyclic", iterations=32, seed=3)
    assert a["last_iterate"] == b["last_iterate"]
    assert run_spo("cyclic", iterations=32, seed=4)["last_iterate"] != a["last_iterate"]


def test_argmax_crossover_semantics():
    trace = [{"a": 0.9, "b": 0.1}, {"a": 0.4, "b": 0.6}, {"a": 0.3, "b": 0.7}]
    assert argmax_crossover(trace, "b") == 2
    assert argmax_crossover(trace, "a") is None
    assert argmax_crossover([{"a": 1.0, "b": 0.0}], "a") == 1


# --- base prior, KL penalty, credit modes, fixed point ---

from spo_simplex_sim import (QWEN_BASE_PRIOR, empirical_table_preference,
                             kl_anneal, kl_fixed_point, kl_penalties,
                             parse_prior)


def test_kl_anneal_schedule():
    assert kl_anneal(1, 480, 0.2, None) == 0.2
    assert kl_anneal(480, 480, 0.2, None) == 0.2
    assert kl_anneal(1, 480, 0.2, 0.02) == pytest.approx(0.2)
    assert kl_anneal(480, 480, 0.2, 0.02) == pytest.approx(0.02)
    mid = kl_anneal(240, 480, 0.2, 0.02)
    assert 0.02 < mid < 0.2


def test_kl_penalties_sign_and_edges():
    pi = {"red": 0.6, "blue": 0.3, "green": 0.1}
    base = {"red": 0.35, "blue": 0.43, "green": 0.22}
    pens = kl_penalties(["red", "blue", UNPARSED], pi, base, 0.2)
    assert pens[0] < 0      # red above base: penalised
    assert pens[1] > 0      # blue below base: rewarded back toward base
    assert pens[2] == 0.0   # unparsed is not a policy action here
    assert kl_penalties(["red"], pi, base, 0.0) == [0.0]


def test_base_prior_seeds_the_start():
    r = run_spo("majority", iterations=1, learning_rate=0.0,
                base_prior=QWEN_BASE_PRIOR, seed=7)
    for a, p in QWEN_BASE_PRIOR.items():
        assert r["last_iterate"][a] == pytest.approx(p, abs=1e-6)


def test_severed_credit_stays_at_base_prior():
    """gamma=0 (tests/test_gamma_credit.py): only the KL penalty reaches the
    colour position, so the policy has nothing pulling it anywhere but base."""
    r = run_spo("majority", iterations=480, base_prior=QWEN_BASE_PRIOR,
                kl_coef=0.2, credit="severed", seed=7,
                pref=empirical_table_preference("majority"))
    for a, p in QWEN_BASE_PRIOR.items():
        assert abs(r["uniform_mixture"][a] - p) < 0.06, a


def test_full_credit_low_kl_reaches_dominant_blue():
    r = run_spo("majority", iterations=480, base_prior=QWEN_BASE_PRIOR,
                kl_coef=0.02, credit="full", seed=7,
                pref=empirical_table_preference("majority"))
    assert r["uniform_mixture"]["blue"] > 0.85
    assert r["last_iterate"]["blue"] > 0.9


def test_fixed_point_tracks_the_kl_coefficient():
    pref = empirical_table_preference("majority")
    blues = [kl_fixed_point("majority", base_prior=QWEN_BASE_PRIOR, kl_coef=b,
                            pref=pref)["blue"] for b in (0.2, 0.05, 0.02, 0.0)]
    assert blues == sorted(blues)            # lower KL -> more dominant blue
    assert blues[0] < 0.60                   # 0.2: pinned well under the bar
    assert blues[2] > 0.90 and blues[3] > 0.97


def test_fixed_point_strong_anchor_returns_base():
    fp = kl_fixed_point("majority", base_prior=QWEN_BASE_PRIOR, kl_coef=50.0)
    for a, p in QWEN_BASE_PRIOR.items():
        assert fp[a] == pytest.approx(p, abs=0.02)


def test_empirical_table_matches_the_seed7_triplets():
    pref = empirical_table_preference("majority")
    assert pref("blue", "red") == pytest.approx(0.5826, abs=1e-3)
    assert pref("blue", "green") == pytest.approx(0.5504, abs=1e-3)
    assert pref("red", "green") == pytest.approx(1.0)
    assert pref(UNPARSED, "red") == 0.0
    assert pref("red", UNPARSED) == 1.0


def test_parse_prior_roundtrip():
    assert parse_prior("red=0.35,blue=0.43,green=0.22") == {
        "red": 0.35, "blue": 0.43, "green": 0.22}


def test_unknown_credit_mode_rejected():
    with pytest.raises(ValueError):
        run_spo("majority", iterations=1, credit="bogus")
