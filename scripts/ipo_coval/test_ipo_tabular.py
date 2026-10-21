"""Pins the claims results/experiments_ipo_coval.md makes from ipo_tabular.py
(Table 2 and section 3)."""

import numpy as np
from ipo_tabular import POPULATIONS, ipo, preference_matrix, regularised_nash

RED, BLUE, GREEN = 0, 1, 2


def m(name):
    return preference_matrix(*POPULATIONS[name])


def test_preference_matrix_majority():
    p = m("majority")
    assert np.isclose(p[BLUE, RED], 0.6) and np.isclose(p[BLUE, GREEN], 0.6)
    assert np.isclose(p[RED, GREEN], 1.0)
    assert np.allclose(p + p.T, 1.0)


def test_small_tau_nash_is_the_maximal_lottery():
    assert regularised_nash(m("majority"), 0.01)[BLUE] > 0.99
    assert np.allclose(regularised_nash(m("cyclic"), 0.01), 1 / 3, atol=1e-3)


def test_large_tau_nash_leans_to_the_borda_winner():
    pi = regularised_nash(m("majority"), 1.0)
    assert pi[RED] > pi[BLUE]


def test_online_ipo_last_iterate_matches_the_nash_equilibrium():
    for tau in (0.1, 0.02):
        assert np.allclose(ipo(m("majority"), tau, online=True),
                           regularised_nash(m("majority"), tau), atol=2e-3)


def test_offline_ipo_on_uniform_pairs_picks_the_borda_winner_and_flips():
    assert ipo(m("majority"), 0.02, online=False)[RED] > 0.9
    assert ipo(m("iia_2alt"), 0.02, online=False)[1] > 0.9      # blue with two options
    assert ipo(m("iia_3alt"), 0.02, online=False)[RED] > 0.9    # red once green is added


def test_online_ipo_last_iterate_converges_on_the_cycle():
    pi = ipo(m("cyclic"), 0.005, online=True, init=(0.8, 0.15, 0.05), steps=40_000)
    assert np.allclose(pi, 1 / 3, atol=1e-3)
