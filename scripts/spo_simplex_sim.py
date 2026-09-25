"""Algorithm 1 (SPO) run in the probability simplex, with no LLM in the loop.

Claims about the dynamics of Algorithm 1 cost GPU-hours per data point to test on
a language model. Replacing the LLM with a 3-way softmax over the colour words keeps the parts
under test — the preference function, the Algorithm 1 reward, the PPO-style
advantage-whitened policy-gradient step, the forced-uniform 10%, and the choice of
output policy — and drops only the language model. It runs in under a second on CPU.

The two properties it demonstrates:

1. `last iterate != uniform mixture`. On the cyclic population the maximal lottery is
   genuinely mixed (1/3 each). Algorithm 1 reaches it only through its stated output,
   `Return: uniform mixture of pi_1:T` (A.8, Algorithm 1). The last iterate orbits the
   simplex forever and never settles — which is what no-regret dynamics on a zero-sum
   cycle do. Evaluating a single checkpoint therefore cannot reproduce the paper's
   right-hand column no matter how well the LLM is trained.

2. `the transient is Borda, the limit is Condorcet`. On the majority population
   (= the paper's left column, and identical to `iia_3alt`), red has the highest mean
   pairwise score against a uniform opponent (0.70 vs blue 0.60), so the first phase of
   training pushes toward red. Blue only overtakes once green has been squeezed out of
   the policy's support. A run that stops — or is checkpointed — before that crossover
   reports red, or a red/blue mixture, rather than the paper's near-1.0 blue.

Exogenous unparsed mass `q` (the fraction of the batch that names no colour) dilutes
the between-colour reward signal and pushes the crossover later.

Options for screening recipes before paying for GPU runs:

- `base_prior`: start the policy at a measured base-model colour distribution and
  penalise KL against it, the way trl's PPOTrainer penalises divergence from the
  reference model.
- `kl_coef` / `kl_coef_final`: per-sample reward penalty -beta * log(pi(a)/base(a)),
  trl's compute_rewards at the alternative level; a final value != initial gives a
  linear anneal across iterations.
- `credit`: "full" models gamma=1 (score + bonus + KL all reach the colour
  decision); "severed" models the transcribed gamma=0, where trl provably delivers
  only the per-token KL penalty to the colour position (tests/test_gamma_credit.py):
  the score never arrives. (The real severed run also carries an arbitrary
  per-colour bias from the untrained value head; the sim leaves that out, so
  severed runs are the *drift-free* best case for gamma=0.)
- `table="empirical"`: score with the empirical preference function over the
  2048 seed-7 triplets (P(B>R)=0.583, P(B>G)=0.550) instead of the population
  margins — the harsher table training actually sees.
- `kl_fixed_point()`: the self-consistent optimum p ∝ base * exp(r(q)/beta),
  q = (1-forced)*p + forced*uniform — where training would settle with credit
  restored, independent of trajectory/learning-rate questions.

Usage:
    python scripts/spo_simplex_sim.py            # table for all four populations
    python scripts/spo_simplex_sim.py --unparsed 0.8
    python scripts/spo_simplex_sim.py --screen   # majority/ml recipe screen
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from pipeline.populations import CONFIGS

UNPARSED = None  # sentinel: a sample that names no colour, as in train_spo.py


def empirical_table_preference(config_name: str, n: int = 2048, seed: int = 7):
    """The preference function over the actual generated triplets (the one
    training sees), via the pipeline's own data_gen + empirical_preference."""
    from pipeline.populations import empirical_preference
    from pipeline.stages.data_gen import sample_triplets

    pop = CONFIGS[config_name]
    rows = sample_triplets(config_name, n, seed, "")
    table = empirical_preference(rows, pop.alternatives)

    def pref(a, b) -> float:
        if a is UNPARSED and b is UNPARSED:
            return 0.5
        if a is UNPARSED:
            return 0.0
        if b is UNPARSED:
            return 1.0
        return table(a, b)

    return pref


def margin_preference(config_name: str):
    """P(a > b) straight from the population, i.e. the infinite-data limit of
    `pipeline.populations.empirical_preference`. Using the exact value instead of a
    2048-row sample keeps the simulation's noise sources down to the batch."""
    pop = CONFIGS[config_name]
    voters = [ranking for weight, ranking in pop.voters for _ in range(weight)]
    table = {
        (a, b): (0.5 if a == b
                 else sum(1 for v in voters if v.index(a) < v.index(b)) / len(voters))
        for a in pop.alternatives for b in pop.alternatives
    }

    def pref(a, b) -> float:
        # A.8 edge rules, restricted to what this simulation can produce: an
        # unparsed sample is an alternative absent from the dataset.
        if a is UNPARSED and b is UNPARSED:
            return 0.5
        if a is UNPARSED:
            return 0.0
        if b is UNPARSED:
            return 1.0
        return table[(a, b)]

    return pref


def algorithm1_rewards(samples: list, pref) -> list[float]:
    """r_i = 1/(k-1) * sum_{j != i} P(a_i > a_j)  (Algorithm 1, A.8)."""
    k = len(samples)
    return [
        sum(pref(samples[i], samples[j]) for j in range(k) if j != i) / max(1, k - 1)
        for i in range(k)
    ]


def rarity_bonus(samples: list, beta: float) -> list[float]:
    """Alternative-level exploration bonus (not used by the pipeline):
    b_i = beta * (-log q_hat(colour_i)) with q_hat the batch's parsed-colour
    frequency. Unparsed samples get no bonus — the bonus pays for rare
    *colours*, never for gibberish."""
    if not beta:
        return [0.0] * len(samples)
    parsed = [s for s in samples if s is not UNPARSED]
    if not parsed:
        return [0.0] * len(samples)
    n = len(parsed)
    freq = {a: parsed.count(a) / n for a in set(parsed)}
    return [0.0 if s is UNPARSED else beta * -math.log(freq[s]) for s in samples]


def _softmax(logits: dict) -> dict:
    hi = max(logits.values())
    exp = {a: math.exp(v - hi) for a, v in logits.items()}
    total = sum(exp.values())
    return {a: v / total for a, v in exp.items()}


def kl_penalties(samples: list, pi: dict, base_prior: dict, coef: float) -> list[float]:
    """trl's compute_rewards at the alternative level: every sample's reward
    carries -coef * (log pi(a) - log base(a)). Forced-uniform samples carry it
    too (trl computes the penalty on the substituted response's logprobs).
    Unparsed samples are not policy actions in this model and get none."""
    if not coef:
        return [0.0] * len(samples)
    return [0.0 if a is UNPARSED
            else -coef * (math.log(max(pi[a], 1e-12)) - math.log(max(base_prior[a], 1e-12)))
            for a in samples]


def kl_anneal(t: int, iterations: int, start: float, final: float | None) -> float:
    """Linear schedule from start to final across iterations; final=None holds
    start constant."""
    if final is None or iterations <= 1:
        return start
    frac = (t - 1) / (iterations - 1)
    return start + (final - start) * frac


def kl_fixed_point(config_name: str, *, base_prior: dict, kl_coef: float,
                   pref=None, uniform_colour_fraction: float = 0.1,
                   iterations: int = 2000, damping: float = 0.9) -> dict:
    """Self-consistent optimum of E[r(a, q)] - kl_coef * KL(pi || base):
    pi ∝ base * exp(r(·, q)/kl_coef) with q = (1-f)*pi + f*uniform.
    kl_coef=0 sends the optimum to the argmax vertex of the limiting rewards."""
    pop = CONFIGS[config_name]
    alts = list(pop.alternatives)
    pref = pref or margin_preference(config_name)
    pi = dict(base_prior)
    for _ in range(iterations):
        q = {a: (1 - uniform_colour_fraction) * pi[a]
                + uniform_colour_fraction / len(alts) for a in alts}
        r = {a: sum(q[b] * pref(a, b) for b in alts) for a in alts}
        if kl_coef:
            target = _softmax({a: math.log(max(base_prior[a], 1e-12)) + r[a] / kl_coef
                               for a in alts})
        else:
            top = max(r, key=r.get)
            target = {a: 1.0 if a == top else 0.0 for a in alts}
        pi = {a: damping * pi[a] + (1 - damping) * target[a] for a in alts}
    return pi


def _whiten(xs: list[float]) -> list[float]:
    """trl's PPOTrainer whitens advantages before the policy loss; the whitening is
    what makes the parsed/unparsed gap crowd out the between-colour signal, so it
    has to be in the model."""
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / sd for x in xs]


def run_spo(config_name: str, *, iterations: int = 480, batch_size: int = 128,
            learning_rate: float = 0.5, uniform_colour_fraction: float = 0.1,
            unparsed_frac: float = 0.0, rarity_beta: float = 0.0,
            seed: int = 7, base_prior: dict | None = None, kl_coef: float = 0.0,
            kl_coef_final: float | None = None, credit: str = "full",
            pref=None) -> dict:
    """One SPO run. Returns the last iterate, the uniform mixture of pi_1:T, and the
    per-iteration trace of both.

    `learning_rate` is a simplex-space step size, not the LLM's 1e-4 — the map between
    them depends on the LoRA parameterisation. Conclusions here are about the *shape* of
    the trajectory (does it settle? in what order?), which is what A.8 pins down; they
    are not a prediction of wall-clock convergence step counts.

    `base_prior` seeds the starting logits and anchors the KL penalty (None keeps the
    original uniform start). `credit="severed"` models the transcribed gamma=0: only
    the per-token KL penalty reaches the colour decision, never the score/bonus.
    """
    if credit not in ("full", "severed"):
        raise ValueError(f"unknown credit mode {credit!r}")
    pop = CONFIGS[config_name]
    alts = list(pop.alternatives)
    pref = pref or margin_preference(config_name)
    rng = random.Random(f"sim-{seed}-{config_name}")

    if base_prior is None:
        base_prior = {a: 1.0 / len(alts) for a in alts}
        logits = {a: 0.0 for a in alts}
    else:
        total = sum(base_prior[a] for a in alts)
        base_prior = {a: base_prior[a] / total for a in alts}
        logits = {a: math.log(max(base_prior[a], 1e-12)) for a in alts}
    mixture_sum = {a: 0.0 for a in alts}
    trace_last, trace_mix = [], []

    for t in range(1, iterations + 1):
        beta = kl_anneal(t, iterations, kl_coef, kl_coef_final)
        pi = _softmax(logits)
        population = list(pi)
        weights = [pi[a] for a in population]
        samples = rng.choices(population, weights=weights, k=batch_size)

        # A.8: "we enforce that 10% of the batch is a uniform sample of the three
        # colour words". Same substitution as train_spo.py.
        n_forced = int(uniform_colour_fraction * batch_size)
        for idx in rng.sample(range(batch_size), n_forced):
            samples[idx] = rng.choice(alts)

        # Exogenous degeneracy: the LLM emitting text that names no colour. Modelled
        # as noise on the batch rather than as a policy action, because the observed
        # collapse is an optimiser/entropy pathology, not something SPO's reward
        # rewards (an unparsed sample loses to every parsed one).
        n_unparsed = int(unparsed_frac * batch_size)
        for idx in rng.sample(range(batch_size), n_unparsed):
            samples[idx] = UNPARSED

        rewards = algorithm1_rewards(samples, pref)
        bonuses = rarity_bonus(samples, rarity_beta)
        kls = kl_penalties(samples, pi, base_prior, beta)
        if credit == "full":
            # gamma=1: score, bonus and KL penalty all reach the colour token
            advantages = _whiten([r + b + k for r, b, k in
                                  zip(rewards, bonuses, kls)])
        else:
            # gamma=0 (tests/test_gamma_credit.py): the colour position's
            # advantage carries only its own token's KL penalty
            advantages = _whiten(kls)

        # PPO's policy loss, to first order, is advantage-weighted score matching:
        #   grad_z log pi(a) = e_a - pi
        grad = {a: 0.0 for a in alts}
        for adv, a in zip(advantages, samples):
            if a is UNPARSED:
                continue  # not a policy action in this model
            for b in alts:
                grad[b] += adv * ((1.0 if b == a else 0.0) - pi[b])
        for a in alts:
            logits[a] += learning_rate * grad[a] / batch_size

        pi_next = _softmax(logits)
        for a in alts:
            mixture_sum[a] += pi_next[a]
        trace_last.append(dict(pi_next))
        trace_mix.append({a: mixture_sum[a] / t for a in alts})

    return {
        "config": config_name,
        "last_iterate": trace_last[-1],
        "uniform_mixture": trace_mix[-1],
        "trace_last": trace_last,
        "trace_mixture": trace_mix,
    }


def argmax_crossover(trace: list[dict], winner: str) -> int | None:
    """First iteration after which `winner` is the argmax and stays so. None if never."""
    last_bad = None
    for i, dist in enumerate(trace, start=1):
        if max(dist, key=dist.get) != winner:
            last_bad = i
    if last_bad is None:
        return 1
    return None if last_bad == len(trace) else last_bad + 1


def _fmt(dist: dict) -> str:
    return " ".join(f"{a[0].upper()}={dist[a]:.2f}" for a in sorted(dist))


# Measured base colour prior of Qwen2.5-0.5B on the three-option prompt (n=1000,
# temperature 1.0, max_new_tokens 8, conditional on parsing; parse rate 0.965).
QWEN_BASE_PRIOR = {"red": 0.348, "blue": 0.429, "green": 0.223}


def parse_prior(text: str) -> dict:
    """"red=0.35,blue=0.43,green=0.22" -> dict."""
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def screen(args) -> int:
    """Recipe screen for majority/ml: fixed points per KL coefficient,
    a severed-credit validation run, and full-credit trajectory sweeps."""
    pref = (empirical_table_preference("majority") if args.table == "empirical"
            else margin_preference("majority"))
    prior = parse_prior(args.base_prior) if args.base_prior else QWEN_BASE_PRIOR
    u = args.unparsed
    print(f"majority/ml recipe screen — base prior "
          f"{_fmt(prior)} (measured), table={args.table}, unparsed={u:.0%}, "
          f"judged number = mixture blue x (1-unparsed); bar >= 0.80\n")

    print("KL-anchored fixed points (equilibria, trajectory-independent):")
    for beta in (0.2, 0.1, 0.05, 0.02, 0.0):
        fp = kl_fixed_point("majority", base_prior=prior, kl_coef=beta, pref=pref)
        print(f"  kl={beta:<5} equilibrium {_fmt(fp)}  "
              f"blue_unconditional={(fp['blue'] * (1 - u)):.3f}")

    print("\nsevered credit (gamma=0) at kl=0.2:")
    for lr in args.lrs:
        r = run_spo("majority", iterations=args.iterations, base_prior=prior,
                    kl_coef=0.2, credit="severed", learning_rate=lr,
                    unparsed_frac=u, seed=args.seed, pref=pref)
        print(f"  lr={lr:<5} mixture {_fmt(r['uniform_mixture'])}  "
              f"last {_fmt(r['last_iterate'])}")

    print("\nfull credit (gamma=1) trajectories:")
    print(f"  {'kl':<12} {'lr':<6} {'mixture pi_1:T':<24} {'judged B':<9} "
          f"{'last iterate':<24} {'B-argmax from':<14} verdict")
    results = []
    for kl, kl_final, label in [(0.2, None, "0.2"), (0.1, None, "0.1"),
                                (0.05, None, "0.05"), (0.02, None, "0.02"),
                                (0.0, None, "0"), (0.2, 0.02, "0.2->0.02")]:
        for lr in args.lrs:
            r = run_spo("majority", iterations=args.iterations, base_prior=prior,
                        kl_coef=kl, kl_coef_final=kl_final, credit="full",
                        learning_rate=lr, unparsed_frac=u, seed=args.seed,
                        pref=pref)
            judged = r["uniform_mixture"]["blue"] * (1 - u)
            cross = argmax_crossover(r["trace_last"], "blue")
            verdict = "PASS" if judged >= 0.80 else "fail"
            results.append({"kl": label, "lr": lr, "judged_blue": round(judged, 3),
                            "mixture": r["uniform_mixture"],
                            "last": r["last_iterate"], "crossover": cross,
                            "verdict": verdict})
            print(f"  {label:<12} {lr:<6} {_fmt(r['uniform_mixture']):<24} "
                  f"{judged:<9.3f} {_fmt(r['last_iterate']):<24} "
                  f"{str(cross):<14} {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "base_prior": prior, "table": args.table, "unparsed": u,
            "iterations": args.iterations, "runs": results}, indent=1))
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=480,
                    help="SPO iterations (480 = the pipeline's 30 epochs x 16 steps)")
    ap.add_argument("--unparsed", type=float, default=0.0,
                    help="exogenous fraction of the batch naming no colour")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="rarity-bonus coefficient (0 = off, as in the pipeline)")
    ap.add_argument("--sweep-beta", action="store_true",
                    help="table over beta in {0, 0.05, 0.1, 0.2} x populations")
    ap.add_argument("--screen", action="store_true",
                    help="majority/ml recipe screen (KL x credit x lr)")
    ap.add_argument("--table", choices=("exact", "empirical"), default="empirical",
                    help="preference table for --screen (default empirical)")
    ap.add_argument("--base-prior", default=None,
                    help='e.g. "red=0.35,blue=0.43,green=0.22" (default: measured '
                         "Qwen prior)")
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                    help="simplex-space learning rates for --screen sweeps")
    ap.add_argument("--json", default=None,
                    help="write --screen results to this JSON path")
    args = ap.parse_args()

    if args.screen:
        return screen(args)

    if args.sweep_beta:
        print(f"Rarity-bonus sweep — {args.iterations} iterations, "
              f"unparsed={args.unparsed:.0%}, seed={args.seed}\n")
        print(f"{'population':<10} {'beta':<6} {'uniform mixture pi_1:T':<26} "
              f"{'max|p - 1/3| (cyclic target <= 0.10)'}")
        print("-" * 86)
        for name in CONFIGS:
            for beta in (0.0, 0.05, 0.1, 0.2):
                r = run_spo(name, iterations=args.iterations,
                            unparsed_frac=args.unparsed, rarity_beta=beta,
                            seed=args.seed)
                mix = r["uniform_mixture"]
                dev = max(abs(v - 1 / len(mix)) for v in mix.values())
                print(f"{name:<10} {beta:<6} {_fmt(mix):<26} {dev:.3f}")
        return 0

    print(f"Algorithm 1 in the simplex — {args.iterations} iterations, "
          f"unparsed={args.unparsed:.0%}, beta={args.beta}, seed={args.seed}\n")
    print(f"{'population':<10} {'last iterate pi_T':<24} {'uniform mixture pi_1:T':<24} "
          f"{'|last - mixture|'}")
    print("-" * 82)
    for name in CONFIGS:
        r = run_spo(name, iterations=args.iterations, unparsed_frac=args.unparsed,
                    rarity_beta=args.beta, seed=args.seed)
        gap = max(abs(r["last_iterate"][a] - r["uniform_mixture"][a])
                  for a in r["last_iterate"])
        print(f"{name:<10} {_fmt(r['last_iterate']):<24} "
              f"{_fmt(r['uniform_mixture']):<24} {gap:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
