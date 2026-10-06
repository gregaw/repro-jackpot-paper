"""Hyperparameter resolution: A.8 values, profile overrides, per-cell overrides."""

import pytest

from pipeline.config import load_profile, train_params


def test_paper_values_are_the_base():
    profile = load_profile("paper")
    profile["train"] = {}
    ml, rlhf = train_params(profile, "ml"), train_params(profile, "rlhf")
    assert ml["epochs"] == 30 and ml["batch_size"] == 128 and ml["mini_batch_size"] == 32
    assert ml["gamma"] == 0.0 and ml["vf_coef"] == 0.0          # A.8 as extracted
    assert rlhf["epochs"] == 4 and rlhf["learning_rate"] == 5e-4
    assert rlhf["rm_epochs"] == 3 and rlhf["center_rewards_coefficient"] == 0.01
    assert ml["lora"] == rlhf["lora"] == {"r": 8, "alpha": 32, "dropout": 0.1}


def test_profile_overrides_apply_per_method():
    profile = load_profile("paper")
    ml, rlhf = train_params(profile, "ml"), train_params(profile, "rlhf")
    assert ml["gamma"] == 1.0 and ml["vf_coef"] == 0.01
    assert ml["mini_batch_size"] == 8 and ml["gradient_accumulation_steps"] == 4
    assert rlhf["epochs"] == 2 and rlhf["init_kl_coef"] == 0.2
    assert rlhf["gamma"] == 1.0                                 # untouched trl default


def test_cell_overrides_apply_only_to_their_cell():
    profile = load_profile("paper")
    assert train_params(profile, "ml", "majority")["init_kl_coef"] == 0.01
    assert train_params(profile, "ml", "cyclic")["init_kl_coef"] == 0.02
    assert train_params(profile, "rlhf", "iia_2alt")["init_kl_coef"] == 0.1
    assert train_params(profile, "rlhf", "cyclic")["init_kl_coef"] == 0.05
    assert train_params(profile, "rlhf", "majority")["init_kl_coef"] == 0.2
    assert train_params(profile, "rlhf")["init_kl_coef"] == 0.2  # no cell: method default


def test_profiles_load_and_agree_on_the_recipe():
    for name in ("paper", "qwen"):
        profile = load_profile(name)
        assert set(profile["gpu"]) == {"train", "eval"}
        assert train_params(profile, "ml")["init_kl_coef"] == 0.02
    smoke = load_profile("smoke")
    assert train_params(smoke, "ml")["epochs"] == 1
    assert train_params(smoke, "rlhf")["rm_epochs"] == 1


def test_unknown_method():
    with pytest.raises(ValueError):
        train_params(load_profile("paper"), "dpo")


def test_ipo_offline_inherits_the_online_arm_settings():
    profile = load_profile("paper")
    profile["train"] = {
        "ipo": {"tau": 0.05, "batch_size": 64},
        "ipo_offline": {"batch_size": 32},
        "cells": {"ipo": {"cyclic": {"tau": 0.01}}},
    }
    online, offline = train_params(profile, "ipo"), train_params(profile, "ipo_offline")
    assert online["tau"] == offline["tau"] == 0.05
    assert online["batch_size"] == 64 and offline["batch_size"] == 32
    assert online["epochs"] == train_params(profile, "ml")["epochs"]   # SPO's sample budget
    assert train_params(profile, "ipo_offline", "cyclic")["tau"] == 0.01
    assert train_params(profile, "ipo", "majority")["tau"] == 0.05


def test_train_overrides_merge_over_the_profile():
    profile = load_profile("qwen", train_overrides={"ipo": {"tau": 0.05},
                                                    "ml": {"ppo_epochs": 3}})
    assert train_params(profile, "ipo")["tau"] == 0.05
    ml = train_params(profile, "ml")
    assert ml["ppo_epochs"] == 3 and ml["gamma"] == 1.0          # rest of train.ml kept
