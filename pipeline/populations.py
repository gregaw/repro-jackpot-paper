"""The four experiment populations from the paper, and the empirical preference function.

Preference profiles (Section 6.3 / Figure 2):
  majority : 2x(R>G>B), 3x(B>R>G)         -- B is majority choice and Condorcet winner
  iia_2alt : 2x(R>B),   3x(B>R)           -- two alternatives only
  iia_3alt : 2x(R>G>B), 3x(B>R>G)         -- iia_2alt plus irrelevant alternative G
  cyclic   : 1x(R>G>B), 1x(G>B>R), 1x(B>R>G)
"""

from dataclasses import dataclass

RED, BLUE, GREEN = "red", "blue", "green"


@dataclass(frozen=True)
class Population:
    name: str
    alternatives: tuple[str, ...]
    # (weight, ranking best-first)
    voters: tuple[tuple[int, tuple[str, ...]], ...]
    prompt_key: str  # key into paper_hparams.yaml prompts


CONFIGS: dict[str, Population] = {
    "majority": Population(
        name="majority",
        alternatives=(RED, BLUE, GREEN),
        voters=((2, (RED, GREEN, BLUE)), (3, (BLUE, RED, GREEN))),
        prompt_key="three_options",
    ),
    "iia_2alt": Population(
        name="iia_2alt",
        alternatives=(RED, BLUE),
        voters=((2, (RED, BLUE)), (3, (BLUE, RED))),
        prompt_key="two_options",
    ),
    "iia_3alt": Population(
        name="iia_3alt",
        alternatives=(RED, BLUE, GREEN),
        voters=((2, (RED, GREEN, BLUE)), (3, (BLUE, RED, GREEN))),
        prompt_key="three_options",
    ),
    "cyclic": Population(
        name="cyclic",
        alternatives=(RED, BLUE, GREEN),
        voters=(
            (1, (RED, GREEN, BLUE)),
            (1, (GREEN, BLUE, RED)),
            (1, (BLUE, RED, GREEN)),
        ),
        prompt_key="three_options",
    ),
}


def empirical_preference(rows: list[dict], alternatives: tuple[str, ...]):
    """Return P(a, b) = fraction of dataset rows preferring a over b (A.8 edge rules).

    rows: [{"chosen": colour, "rejected": colour}, ...]
    Edge rules from A.8: ties -> 0.5; only one present -> present wins (1.0);
    neither present -> 0.5.
    """
    wins: dict[tuple[str, str], int] = {}
    present: set[str] = set()
    for row in rows:
        wins[(row["chosen"], row["rejected"])] = wins.get((row["chosen"], row["rejected"]), 0) + 1
        present.add(row["chosen"])
        present.add(row["rejected"])

    def pref(a: str, b: str) -> float:
        if a == b:
            return 0.5
        if a not in present and b not in present:
            return 0.5
        if b not in present:
            return 1.0
        if a not in present:
            return 0.0
        ab, ba = wins.get((a, b), 0), wins.get((b, a), 0)
        if ab == 0 and ba == 0:
            return 0.5
        return ab / (ab + ba)

    return pref
