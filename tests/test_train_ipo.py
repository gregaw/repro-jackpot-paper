"""Unit tests for the Online IPO stage (pipeline/stages/train_ipo.py): the loss
on hand-worked pairs, the pair targets and their edge rules, the response mask,
the two-pass mini-batch gradient, and that the loss drives a one-token toy
policy to the regularised Nash equilibrium online and to the Borda winner
offline."""

import pytest
import torch

from pipeline.populations import CONFIGS, empirical_preference
from pipeline.stages import train_ipo as ti


def table_pref(table):
    return lambda a, b: 0.5 if a == b else table[(a, b)]


def test_online_loss_hand_worked():
    d = torch.tensor([1.0, 0.0])
    targets = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])
    # one pair (0, 1): h = 1, target 2 -> (1 - 2)^2
    assert ti.online_ipo_loss(d, targets).item() == pytest.approx(1.0)


def test_online_loss_averages_pairs_i_lt_j():
    d = torch.tensor([0.0, 1.0, 3.0])
    targets = torch.zeros(3, 3)
    # h01 = -1, h02 = -3, h12 = -2 -> (1 + 9 + 4) / 3
    assert ti.online_ipo_loss(d, targets).item() == pytest.approx(14 / 3)


def test_offline_loss_hand_worked():
    tau = 0.25                                       # target 1/(2 tau) = 2
    loss = ti.offline_ipo_loss(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 0.0]), tau)
    assert loss.item() == pytest.approx(((3 - 2) ** 2 + (0 - 2) ** 2) / 2)


def test_pair_targets_edge_rules_and_antisymmetry():
    pref = table_pref({("red", "blue"): 0.4, ("blue", "red"): 0.6})
    t = ti.pair_targets(["red", "blue", None, None, "red"], pref, tau=0.1)
    assert t[0, 1].item() == pytest.approx(-1.0)     # (0.4 - 0.5) / 0.1
    assert t[1, 0].item() == pytest.approx(1.0)
    assert t[0, 2].item() == pytest.approx(5.0)      # a named colour beats unparsed
    assert t[2, 0].item() == pytest.approx(-5.0)
    assert t[2, 3].item() == 0.0                     # two unparsed tie
    assert t[0, 4].item() == 0.0                     # same colour
    assert torch.allclose(t, -t.T)


def test_soft_target_is_hard_label_expectation():
    # the soft target (P - 1/2)/tau equals the mean of the hard IPO targets
    # +-1/(2 tau) over the dataset's orientations of the pair
    rows = [{"chosen": "blue", "rejected": "red"}] * 3 + [{"chosen": "red", "rejected": "blue"}] * 2
    pref = empirical_preference(rows, ("red", "blue"))
    tau = 0.1
    hard = [(1 if r["chosen"] == "blue" else -1) / (2 * tau) for r in rows]
    soft = ti.pair_targets(["blue", "red"], pref, tau)[0, 1].item()
    assert soft == pytest.approx(sum(hard) / len(hard))


def test_response_mask_stops_after_first_eos():
    eos = 1
    resp = torch.tensor([[5, 6, 1, 1, 1],     # pad == eos
                         [5, 1, 0, 0, 0],     # pad != eos
                         [5, 6, 7, 8, 9]])    # never stops
    m = ti.response_mask(resp, eos)
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]


def _tiny_policy():
    from pipeline.models import build_tokenizer

    tok = build_tokenizer("tiny-random")
    torch.manual_seed(0)
    lora = {"r": 4, "alpha": 8, "dropout": 0.0}      # no dropout: passes must agree
    policy = ti.build_policy("tiny-random", "float32", lora, tok, "cpu")
    # non-zero LoRA B so the policy differs from the reference
    for n, p in policy.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)
    return policy, tok


def _grads_after_step(policy, query, resp, mask, loss_fn, mini_batch_size):
    opt = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.0)
    stats = ti.ipo_step(policy, opt, query, resp, mask, loss_fn,
                        mini_batch_size=mini_batch_size, max_grad_norm=1e9)
    return stats, {n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad}


def test_mini_batch_two_pass_matches_full_batch_gradient():
    policy, tok = _tiny_policy()
    query = tok("the colour is", return_tensors="pt")["input_ids"][0]
    resp = torch.tensor([[3, 14, 1, 1], [4, 1, 1, 1], [5, 14, 3, 1], [3, 3, 3, 3]])
    mask = ti.response_mask(resp, tok.eos_token_id)
    targets = ti.pair_targets(["red", "green", "blue", None],
                              table_pref({(a, b): 0.7 if a < b else 0.3
                                          for a in ("red", "green", "blue")
                                          for b in ("red", "green", "blue")}), tau=0.1)

    def loss_fn(d):
        return ti.online_ipo_loss(d, targets)

    full_stats, full = _grads_after_step(policy, query, resp, mask, loss_fn, 4)
    mb_stats, mb = _grads_after_step(policy, query, resp, mask, loss_fn, 2)
    assert full_stats["loss"] == pytest.approx(mb_stats["loss"], rel=1e-5)
    assert full_stats["grad_norm"] > 0
    for n in full:
        assert torch.allclose(full[n], mb[n], atol=1e-6, rtol=1e-4), n


def test_reference_is_adapter_off():
    policy, tok = _tiny_policy()
    query = tok("the colour is", return_tensors="pt")["input_ids"][0]
    resp = torch.tensor([[3, 1], [4, 1]])
    mask = ti.response_mask(resp, tok.eos_token_id)
    d = ti.log_ratios(policy, query, resp, mask, with_grad=False)
    assert d.abs().max() > 0                          # adapter on: differs from ref
    for n, p in policy.named_parameters():
        if "lora_B" in n:
            torch.nn.init.zeros_(p)                   # adapter is now the identity
    d0 = ti.log_ratios(policy, query, resp, mask, with_grad=False)
    assert torch.allclose(d0, torch.zeros_like(d0), atol=1e-5)


# --- a one-token toy policy: the stage's loss reaches the regularised Nash
#     equilibrium online and the Borda winner offline (cf. 0086 report §2.3)

def _preference_matrix(config):
    pop = CONFIGS[config]
    alts = pop.alternatives
    total = sum(w for w, _ in pop.voters)
    m = torch.full((len(alts), len(alts)), 0.5)
    for i, a in enumerate(alts):
        for j, b in enumerate(alts):
            if i != j:
                m[i, j] = sum(w for w, r in pop.voters if r.index(a) < r.index(b)) / total
    return alts, m


def _regularised_nash(m, tau, iters=20_000):
    """pi ∝ ref exp(P(y > pi) / tau), uniform ref, by damped fixed point."""
    pi = torch.full((m.shape[0],), 1 / m.shape[0], dtype=torch.float64)
    m = m.double()
    for _ in range(iters):
        pi = 0.99 * pi + 0.01 * torch.softmax(m @ pi / tau, 0)
    return pi.float()


def _toy_ipo(m, tau, online, steps=3000, init=None, betas=(0.9, 0.999)):
    n = m.shape[0]
    log_ref = torch.full((n,), -torch.log(torch.tensor(float(n))))
    theta = (torch.log(torch.tensor(init)) if init else log_ref.clone()).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=0.02, betas=betas)
    targets = (m - 0.5) / tau
    for _ in range(steps):
        log_pi = torch.log_softmax(theta, 0)
        mu = log_pi.exp().detach() if online else log_ref.exp()
        loss = ti.online_ipo_loss(log_pi - log_ref, targets, weights=torch.outer(mu, mu))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(theta, 0).detach()


def test_online_reaches_regularised_nash_on_majority():
    alts, m = _preference_matrix("majority")
    pi = _toy_ipo(m, tau=0.1, online=True)
    assert torch.allclose(pi, _regularised_nash(m, 0.1), atol=0.02)
    assert alts[int(pi.argmax())] == "blue"           # the Condorcet winner


def test_offline_uniform_pairs_pick_borda_winner():
    alts, m = _preference_matrix("majority")
    pi = _toy_ipo(m, tau=0.1, online=False)
    assert alts[int(pi.argmax())] == "red"            # Borda, like RLHF


def test_online_last_iterate_converges_on_cycle_without_momentum():
    # Plain gradient steps spiral in to the uniform equilibrium. With Adam's
    # default momentum (beta1 = 0.9) at this step size the last iterate
    # orbits instead, which is why train.ipo exposes adam_beta1.
    _, m = _preference_matrix("cyclic")
    uniform = torch.full((3,), 1 / 3)
    pi = _toy_ipo(m, tau=0.05, online=True, init=[0.8, 0.15, 0.05], steps=4000,
                  betas=(0.0, 0.999))
    assert torch.allclose(pi, uniform, atol=0.03)
    orbit = _toy_ipo(m, tau=0.05, online=True, init=[0.8, 0.15, 0.05], steps=4000)
    assert not torch.allclose(orbit, uniform, atol=0.03)


class _FakeAdapterPolicy:
    """Fixed next-token logits, one set with the adapter on and one with it off."""

    def __init__(self, pol_logits, ref_logits):
        self.pol, self.ref, self.adapter_on = pol_logits, ref_logits, True

    def eval(self):
        return self

    def __call__(self, input_ids):
        from types import SimpleNamespace

        row = self.pol if self.adapter_on else self.ref
        b, t = input_ids.shape
        return SimpleNamespace(logits=row.expand(b, t, -1))

    def disable_adapter(self):
        from contextlib import contextmanager

        @contextmanager
        def off():
            self.adapter_on = False
            try:
                yield
            finally:
                self.adapter_on = True
        return off()


def test_geometric_mixture_sampler_matches_its_target_distribution():
    pol = torch.log(torch.tensor([0.90, 0.05, 0.05]))
    ref = torch.log(torch.tensor([0.10, 0.45, 0.45]))
    fake = _FakeAdapterPolicy(pol, ref)
    query = torch.tensor([0, 0])
    torch.manual_seed(1)
    for beta in (0.0, 0.5, 1.0):
        target = torch.softmax((1 - beta) * pol + beta * ref, -1)
        out = ti.generate_geometric_mixture(fake, query, 4000, 1, pad_token_id=9,
                                            eos_token_id=8, beta=beta, chunk=1000)
        freq = torch.bincount(out[:, 0], minlength=3).float() / 4000
        assert (freq - target).abs().max() < 0.03, beta


def test_geometric_mixture_pads_after_eos():
    policy, tok = _tiny_policy()
    query = tok("the colour is", return_tensors="pt")["input_ids"][0]
    out = ti.generate_geometric_mixture(policy, query, 64, 6, tok.pad_token_id,
                                        tok.eos_token_id, 0.3)
    assert out.shape == (64, 6)
    for row in out.tolist():
        if tok.eos_token_id in row:
            k = row.index(tok.eos_token_id)
            assert all(t == tok.pad_token_id for t in row[k + 1:])


def test_lr_multiplier_decays_linearly_over_the_final_fraction():
    assert [ti.lr_multiplier(s, 100, 0.0) for s in (0, 99, 100)] == [1.0, 1.0, 1.0]
    assert ti.lr_multiplier(79, 100, 0.2) == 1.0
    assert ti.lr_multiplier(80, 100, 0.2) == 1.0
    assert ti.lr_multiplier(90, 100, 0.2) == pytest.approx(0.5)
    assert ti.lr_multiplier(100, 100, 0.2) == 0.0
