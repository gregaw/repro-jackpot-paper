"""data_gen unit test: sampled triplet frequencies match the population spec."""

import math

from pipeline.populations import CONFIGS, empirical_preference
from pipeline.stages.data_gen import sample_triplets


def expected_preference(pop, a: str, b: str) -> float:
    total = sum(w for w, _ in pop.voters)
    wins = sum(w for w, ranking in pop.voters if ranking.index(a) < ranking.index(b))
    return wins / total


def test_frequencies_match_population_spec():
    n = 20000
    for name, pop in CONFIGS.items():
        rows = sample_triplets(name, n, seed=1, prompt="p")
        pref = empirical_preference(rows, pop.alternatives)
        for i, a in enumerate(pop.alternatives):
            for b in pop.alternatives[i + 1:]:
                exp = expected_preference(pop, a, b)
                # 3-sigma tolerance for a binomial with the pair-count sample size
                pairs = sum(1 for r in rows if {r["chosen"], r["rejected"]} == {a, b})
                tol = 3 * math.sqrt(exp * (1 - exp) / max(1, pairs)) + 1e-9
                assert abs(pref(a, b) - exp) < tol, (name, a, b, pref(a, b), exp)


def test_pairs_uniform_and_distinct():
    rows = sample_triplets("majority", 9000, seed=2, prompt="p")
    from collections import Counter

    pair_counts = Counter(frozenset((r["chosen"], r["rejected"])) for r in rows)
    assert all(len(p) == 2 for p in pair_counts)  # alternatives always distinct
    assert len(pair_counts) == 3                  # all 3 unordered pairs occur
    for count in pair_counts.values():            # roughly uniform pair sampling
        assert abs(count - 3000) < 3 * (9000 * (1 / 3) * (2 / 3)) ** 0.5


def test_edge_rules():
    pref = empirical_preference([{"chosen": "red", "rejected": "blue"}], ("red", "blue", "green"))
    assert pref("red", "blue") == 1.0
    assert pref("blue", "red") == 0.0
    assert pref("red", "green") == 1.0   # green absent -> present alternative wins
    assert pref("green", "red") == 0.0
    assert pref("green", "green") == 0.5
