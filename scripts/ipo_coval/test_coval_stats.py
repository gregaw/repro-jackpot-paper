"""Pins coval_stats.py's social-choice helpers on hand-built elections, including
the Jackpot colour populations relabelled as three responses A-C."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coval_stats as cs  # noqa: E402


def abc(*spec):
    return cs.pairwise(votes(*spec), letters="ABC")


def votes(*spec):
    """votes((2, "A>C>B"), (3, "B>A>C")) -> parsed rankings, weights expanded."""
    return [cs.parse_ranking(r) for w, r in spec for _ in range(w)]


def test_parse_ranking_tiers():
    assert cs.parse_ranking("A>B>C=D") == [["A"], ["B"], ["C", "D"]]
    assert cs.parse_ranking("A = B > C") == [["A", "B"], ["C"]]


def test_pairwise_ties_count_half_and_unranked_pairs_are_half():
    p = cs.pairwise(votes((1, "A>B=C"), (1, "B>A>C")))
    assert p[0, 1] == 0.5 and p[1, 2] == 0.75 and p[0, 2] == 1.0
    assert np.allclose(p + p.T, 1.0)
    assert p[0, 3] == 0.5                       # D never ranked


def test_majority_population_condorcet_vs_borda():
    # Jackpot "majority": 2 voters red>green>blue, 3 voters blue>red>green
    # with red=A, blue=B, green=C: Condorcet winner B, Borda winner A
    p = abc((2, "A>C>B"), (3, "B>A>C"))
    c = cs.classify(p)
    assert c["condorcet"] == "B" and c["borda"] == "A" and not c["cw_is_borda"]
    assert cs.bucket(c) == "cw_ne_borda"
    assert np.allclose(cs.maximal_lottery(p), [0, 1, 0])


def test_cyclic_population_has_uniform_lottery_and_no_winner():
    p = abc((1, "A>C>B"), (1, "C>B>A"), (1, "B>A>C"))
    c = cs.classify(p)
    assert c["condorcet"] is None and not c["weak_condorcet"]
    ml = cs.maximal_lottery(p)
    assert np.allclose(ml, [1 / 3] * 3, atol=1e-6)


def test_iia_flip_when_an_irrelevant_option_is_added():
    # iia_3alt: with green (C) present Borda picks red (A); drop C and B wins
    p = abc((2, "A>C>B"), (3, "B>A>C"))
    assert cs.iia_flip(p)
    unanimous = abc((5, "A>B>C"))
    assert not cs.iia_flip(unanimous)


def test_bootstrap_of_a_unanimous_election_is_fully_stable():
    s = cs.bootstrap_stability(votes((7, "A>B>C>D")), n_boot=20, rng=random.Random(0))
    assert s == {"bucket": 1.0, "condorcet": 1.0, "iia_flip": 1.0}


def test_prompt_report_on_the_majority_population():
    # majority relabelled: A = red (Borda), B = blue (Condorcet), C = green;
    # D, last on every ballot, plays no part
    ranks = votes((2, "A>C>B>D"), (3, "B>A>C>D"))
    rep = cs.prompt_report(ranks, n_boot=50, seed=1)
    assert rep["n_voters"] == 5
    assert rep["condorcet"] == "B" and rep["borda"] == "A"
    assert rep["maximal_lottery"][1] == 1.0
    assert rep["iia_removals"] == ["C"]              # dropping green hands Borda to B
    assert sum(rep["bootstrap"]["borda"].values()) == 50
    assert len(rep["leave_one_out_borda"]) == 5
    assert rep["borda_scores"][0] > rep["borda_scores"][1]
