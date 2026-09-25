"""Credit assignment from the preference score to the colour token.

trl 0.10.1's PPOTrainer adds the score to the last response token only
(compute_rewards) and propagates it backward through gamma * lam recursion
(compute_advantages). At gamma 0, A.8's value as extracted, the advantage at
the colour position (token 0) is independent of the score, so SPO cannot learn
the preference. These tests pin that on trl's own code and check the profile
runs with gamma 1.0 and a trained value head.
"""

import torch

from pipeline.config import load_profile, train_params
from pipeline.stages import train_common as tc


def make_trainer(gamma: float):
    from trl import PPOConfig

    from trl import PPOTrainer

    t = PPOTrainer.__new__(PPOTrainer)   # compute_rewards/advantages touch only config + kl_ctl
    t.config = PPOConfig(batch_size=2, mini_batch_size=2, gamma=gamma)
    from trl.trainer.ppo_trainer import FixedKLController

    t.kl_ctl = FixedKLController(0.2)
    return t


def colour_position_advantages(gamma: float, scores: list[float],
                               seq_len: int = 4) -> torch.Tensor:
    """Advantage at token 0 (the colour token) for two responses that are
    identical in every way except their preference score."""
    t = make_trainer(gamma)
    n = len(scores)
    logprobs = torch.zeros(n, seq_len)       # zero KL: policy == reference
    ref_logprobs = torch.zeros(n, seq_len)
    masks = torch.ones(n, seq_len, dtype=torch.long)
    rewards, _, _ = t.compute_rewards(torch.tensor(scores), logprobs,
                                      ref_logprobs, masks)
    values = torch.zeros(n, seq_len)         # vf_coef=0: value head never learns
    _, advantages, _ = t.compute_advantages(values, rewards, masks)
    return advantages[:, 0]


def test_score_lands_on_last_token_only():
    t = make_trainer(0.0)
    logprobs = torch.zeros(1, 4)
    rewards, _, _ = t.compute_rewards(torch.tensor([1.0]), logprobs,
                                      torch.zeros(1, 4),
                                      torch.ones(1, 4, dtype=torch.long))
    assert rewards[0, -1] == 1.0
    assert rewards[0, :-1].abs().sum() == 0.0


def test_gamma_zero_severs_score_from_colour_token():
    adv = colour_position_advantages(0.0, [0.0, 1.0])
    # the two samples differ only in score; at gamma=0 the colour position
    # cannot see it
    assert torch.allclose(adv[0], adv[1])


def test_gamma_one_delivers_score_to_colour_token():
    adv = colour_position_advantages(1.0, [0.0, 1.0])
    assert (adv[1] - adv[0]).item() > 0.1


def test_profile_restores_gamma_and_trains_the_value_head():
    for name in ("paper", "qwen"):
        profile = load_profile(name)
        assert profile["paper"]["spo"]["gamma"] == 0.0          # A.8 as extracted
        assert profile["paper"]["spo"]["vf_coef"] == 0.0
        assert train_params(profile, "ml")["gamma"] == 1.0      # what runs
        assert train_params(profile, "ml")["vf_coef"] == 0.01
        assert train_params(profile, "rlhf")["gamma"] == 1.0


def test_trained_value_head_removes_bootstrap_noise_from_advantages():
    """With gamma=1 the colour-position
    advantage is score + telescoped value-head terms. A perfect value head
    (V_t = expected return) cancels to near-zero mean noise; a random one
    injects bias on the scale of its own outputs."""
    t = make_trainer(1.0)
    seq = 4
    logprobs = torch.zeros(2, seq)
    rewards, _, _ = t.compute_rewards(torch.tensor([0.0, 1.0]), logprobs,
                                      torch.zeros(2, seq),
                                      torch.ones(2, seq, dtype=torch.long))
    # random ("untrained") values shift the colour-position advantage gap away
    # from what the scores alone give
    torch.manual_seed(0)
    noisy = torch.randn(2, seq)
    _, adv_clean, _ = t.compute_advantages(torch.zeros(2, seq), rewards.clone(),
                                           torch.ones(2, seq, dtype=torch.long))
    _, adv_noisy, _ = t.compute_advantages(noisy, rewards.clone(),
                                           torch.ones(2, seq, dtype=torch.long))
    clean_gap = (adv_clean[1, 0] - adv_clean[0, 0]).item()
    noisy_gap = (adv_noisy[1, 0] - adv_noisy[0, 0]).item()
    assert abs(noisy_gap - clean_gap) > 0.05
