"""Profile loading: a profile yaml (experiments, qwen or smoke) plus paper_hparams.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"


def load_profile(name: str = "paper", seed: int | None = None) -> dict:
    """`name` is "paper" for configs/experiments.yaml or the stem of another profile
    yaml. `seed` overrides the profile's data/training seed."""
    fname = "experiments.yaml" if name == "paper" else f"{name}.yaml"
    profile = yaml.safe_load((CONFIGS_DIR / fname).read_text())
    profile["paper"] = yaml.safe_load((CONFIGS_DIR / "paper_hparams.yaml").read_text())
    if seed is not None:
        profile["data"]["seed"] = seed
    return profile


def train_params(profile: dict, method: str, config_name: str | None = None) -> dict:
    """Training hyperparameters for one cell: Appendix A.8 values, then the
    profile's `train.<method>` overrides, then `train.cells.<method>.<config>`."""
    paper = profile["paper"]
    if method == "ml":
        spo = paper["spo"]
        base = {
            "epochs": spo["epochs"],
            "batch_size": spo["batch_size"],
            "mini_batch_size": spo["mini_batch_size"],
            "gradient_accumulation_steps": 1,
            "ppo_epochs": 4,                      # trl 0.10.1 default; A.8 does not say
            "learning_rate": spo["learning_rate"],
            "vf_coef": spo["vf_coef"],
            "init_kl_coef": spo["init_kl_coef"],
            "gamma": spo["gamma"],
            "uniform_colour_fraction": spo["uniform_colour_fraction"],
        }
    elif method == "rlhf":
        ppo, rm = paper["rlhf_ppo"], paper["reward_model"]
        base = {
            "epochs": ppo["epochs"],
            "batch_size": ppo["batch_size"],
            "mini_batch_size": ppo["batch_size"],
            "gradient_accumulation_steps": 1,
            "learning_rate": ppo["learning_rate"],
            "vf_coef": ppo["vf_coef"],
            "init_kl_coef": ppo["init_kl_coef"],
            "gamma": 1.0,                         # trl default
            "rm_epochs": rm["epochs"],
            "center_rewards_coefficient": rm["center_rewards_coefficient"],
        }
    else:
        raise ValueError(f"unknown method {method!r}")
    train = profile.get("train") or {}
    base.update(train.get(method) or {})
    if config_name is not None:
        base.update(((train.get("cells") or {}).get(method) or {}).get(config_name) or {})
    base["lora"] = paper["lora"]
    return base


def prompt_for(profile: dict, prompt_key: str) -> str:
    return profile["paper"]["prompts"][prompt_key]
