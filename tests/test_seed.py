"""--seed as a first-class cell dimension: profile override and hash separation."""

from pipeline import manifest
from pipeline.config import load_profile


def test_load_profile_seed_override():
    assert load_profile("smoke")["data"]["seed"] == 7          # yaml default
    assert load_profile("smoke", seed=123)["data"]["seed"] == 123


def test_seed_separates_params_hashes():
    base = {"config": "cyclic", "method": "ml", "epochs": 3}
    h1 = manifest.params_hash({**base, "seed": 7})
    h2 = manifest.params_hash({**base, "seed": 8})
    assert h1 != h2
