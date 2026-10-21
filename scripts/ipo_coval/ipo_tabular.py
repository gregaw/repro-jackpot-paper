"""Tabular online vs offline IPO on the four Jackpot colour populations.

Reproduces Table 2 of results/experiments_ipo_coval.md. The policy is a softmax over the colours (no
language model), the reference is uniform, and the gradient is the exact
expectation of the IPO loss over pairs, so there is no sampling noise:

    L(pi) = sum_{y,y'} mu(y) mu(y') [h(y,y') - (P(y > y') - 1/2) / tau]^2
    h(y,y') = log pi(y)/ref(y) - log pi(y')/ref(y')

mu is stop-gradient. Online IPO sets mu = pi (the current policy generates the
pairs); offline IPO fixes mu (here: uniform pairs, as in our data_gen stage).

    python ipo_tabular.py
"""

import numpy as np

POPULATIONS = {
    "majority": (("red", "blue", "green"), ((2, "red green blue"), (3, "blue red green"))),
    "iia_2alt": (("red", "blue"), ((2, "red blue"), (3, "blue red"))),
    "iia_3alt": (("red", "blue", "green"), ((2, "red green blue"), (3, "blue red green"))),
    "cyclic": (("red", "blue", "green"),
               ((1, "red green blue"), (1, "green blue red"), (1, "blue red green"))),
}


def preference_matrix(alternatives, voters) -> np.ndarray:
    """M[i, j] = fraction of voters ranking alternatives[i] above alternatives[j]."""
    n = len(alternatives)
    total = sum(w for w, _ in voters)
    m = np.full((n, n), 0.5)
    for i, a in enumerate(alternatives):
        for j, b in enumerate(alternatives):
            if i != j:
                m[i, j] = sum(w for w, r in voters
                              if r.split().index(a) < r.split().index(b)) / total
    return m


def softmax(theta: np.ndarray) -> np.ndarray:
    e = np.exp(theta - theta.max())
    return e / e.sum()


def regularised_nash(m: np.ndarray, tau: float, ref=None, iters: int = 200_000) -> np.ndarray:
    """Solve pi(y) ~ ref(y) exp(P(y > pi) / tau) by damped fixed-point iteration."""
    ref = np.full(len(m), 1 / len(m)) if ref is None else np.asarray(ref, float)
    pi = ref.copy()
    for _ in range(iters):
        pi = 0.999 * pi + 0.001 * softmax(np.log(ref) + m @ pi / tau)
    return pi


def ipo(m: np.ndarray, tau: float, *, online: bool, init=None, ref=None,
        steps: int = 20_000, lr: float = 0.5, return_trajectory: bool = False):
    """Gradient descent on the expected IPO loss. Returns the last iterate."""
    n = len(m)
    ref = np.full(n, 1 / n) if ref is None else np.asarray(ref, float)
    theta = np.log(ref if init is None else np.asarray(init, float))
    target = (m - 0.5) / tau
    trajectory = []
    for _ in range(steps):
        pi = softmax(theta)
        trajectory.append(pi)
        mu = pi if online else ref
        log_ratio = np.log(pi) - np.log(ref)
        h = log_ratio[:, None] - log_ratio[None, :]
        r = np.outer(mu, mu) * (h - target)
        g = r.sum(1) - r.sum(0)          # dL/dlog pi(y), up to a factor 2
        g = g - pi * g.sum()             # chain rule through the softmax
        theta -= lr * tau * g            # lr scaled by tau: the target grows as 1/tau
    pi = softmax(theta)
    return (pi, np.array(trajectory)) if return_trajectory else pi


def fmt(p) -> str:
    return " / ".join(f"{x:.2f}" for x in p)


def main() -> None:
    taus = (1.0, 0.1, 0.05, 0.02, 0.01)
    print("Regularised Nash equilibrium, uniform reference")
    for name, (alts, voters) in POPULATIONS.items():
        m = preference_matrix(alts, voters)
        print(f"  {name} ({'/'.join(alts)})")
        for tau in taus:
            print(f"    tau={tau:<5} {fmt(regularised_nash(m, tau))}")

    print("\nLast iterate of IPO: online (self-play) vs offline (uniform pairs)")
    for name, (alts, voters) in POPULATIONS.items():
        m = preference_matrix(alts, voters)
        for tau in (1.0, 0.1, 0.02):
            on, off = ipo(m, tau, online=True), ipo(m, tau, online=False)
            print(f"  {name:9s} tau={tau:<4} online {fmt(on)}   offline {fmt(off)}")

    print("\nCyclic, online IPO from a skewed start (0.80 / 0.15 / 0.05)")
    m = preference_matrix(*POPULATIONS["cyclic"])
    for tau in (0.1, 0.02, 0.005):
        _, traj = ipo(m, tau, online=True, init=(0.8, 0.15, 0.05), steps=40_000,
                      return_trajectory=True)
        print(f"  tau={tau:<5} step 1000 {fmt(traj[1000])}  step 10000 {fmt(traj[10000])}"
              f"  last {fmt(traj[-1])}")


if __name__ == "__main__":
    main()
