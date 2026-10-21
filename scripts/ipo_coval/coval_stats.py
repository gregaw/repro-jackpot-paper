"""Social-choice statistics of the CoVal dataset (openai/coval), per prompt.

Each prompt is a small election: about 17 annotators rank four responses A-D.
For the `world` and `personal` rankings separately this computes, per prompt,
the pairwise preference P(a > b) (ties count 1/2, pairs a voter left unranked
are skipped), the Condorcet winner, the Borda winner (sum of P against the
others), the Maximal Lottery, and whether removing one losing response changes
the Borda winner (an IIA failure). A bootstrap over annotators then keeps only
prompts whose classification is stable, and sorts them into the buckets of
Table 7 in results/experiments_ipo_coval.md.

    python coval_stats.py <dir with comparisons.jsonl> [--boot 200] [--out stats.json]
    python coval_stats.py <dir> --example <prompt_id>   # one prompt in detail (section 6.2)

Needs numpy only. The data is not committed; download it with
    curl -LO https://huggingface.co/datasets/openai/coval/resolve/main/comparisons.jsonl
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

LETTERS = "ABCD"


def parse_ranking(text: str) -> list[list[str]]:
    """"A>B>C=D" -> [["A"], ["B"], ["C", "D"]] (tiers, best first)."""
    return [[x.strip() for x in tier.split("=") if x.strip()]
            for tier in text.replace(" ", "").split(">") if tier]


def pairwise(rankings: list[list[list[str]]], letters=LETTERS) -> np.ndarray:
    """P[i, j] = share of voters ranking both i and j who put i above j, ties
    counting 1/2; 0.5 where no voter ranked both."""
    n = len(letters)
    wins, seen = np.zeros((n, n)), np.zeros((n, n))
    idx = {c: k for k, c in enumerate(letters)}
    for tiers in rankings:
        tier_of = {c: t for t, tier in enumerate(tiers) for c in tier if c in idx}
        for a, b in itertools.permutations(tier_of, 2):
            i, j = idx[a], idx[b]
            seen[i, j] += 1
            if tier_of[a] < tier_of[b]:
                wins[i, j] += 1.0
            elif tier_of[a] == tier_of[b]:
                wins[i, j] += 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(seen > 0, wins / np.maximum(seen, 1), 0.5)
    np.fill_diagonal(p, 0.5)
    return p


def condorcet_winner(p: np.ndarray, alive=None):
    """Index beating every other alive option strictly (P > 1/2), else None."""
    alive = list(range(len(p))) if alive is None else alive
    for i in alive:
        if all(p[i, j] > 0.5 for j in alive if j != i):
            return i
    return None


def borda_winners(p: np.ndarray, alive=None) -> set:
    alive = list(range(len(p))) if alive is None else alive
    score = {i: sum(p[i, j] for j in alive if j != i) for i in alive}
    best = max(score.values())
    return {i for i, s in score.items() if s > best - 1e-9}


def maximal_lottery(p: np.ndarray) -> np.ndarray:
    """A mixed strategy x with x^T (P - 1/2) >= 0 in every column, found by
    enumerating supports (n = 4, so at most 15). With ties the lottery need not
    be unique; this returns the first feasible support, smallest first."""
    m = p - 0.5
    n = len(p)
    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            s = list(support)
            # x_S (M_SS) = 0 on the support, sum x = 1
            a = np.vstack([m[np.ix_(s, s)].T, np.ones(size)])
            b = np.zeros(size + 1)
            b[-1] = 1.0
            x_s, *_ = np.linalg.lstsq(a, b, rcond=None)
            if np.abs(a @ x_s - b).max() > 1e-7 or (x_s < -1e-9).any():
                continue
            x = np.zeros(n)
            x[s] = np.clip(x_s, 0, None)
            x /= x.sum()
            if (x @ m >= -1e-9).all():
                return x
    raise RuntimeError("no maximal lottery found")      # cannot happen: the game has a value


def iia_flip(p: np.ndarray) -> bool:
    """Does removing some option outside the Borda winners change who wins?
    Only prompts with a unique Borda winner count."""
    full = borda_winners(p)
    if len(full) != 1:
        return False
    (w,) = full
    for drop in range(len(p)):
        if drop == w:
            continue
        alive = [i for i in range(len(p)) if i != drop]
        if borda_winners(p, alive) != full:
            return True
    return False


def classify(p: np.ndarray) -> dict:
    cw = condorcet_winner(p)
    bw = borda_winners(p)
    weak = any(all(p[i, j] >= 0.5 for j in range(len(p))) for i in range(len(p)))
    return {
        "condorcet": None if cw is None else LETTERS[cw],
        "weak_condorcet": weak,       # some option ties-or-beats every other
        "borda": "".join(sorted(LETTERS[i] for i in bw)),
        "cw_is_borda": cw is not None and bw == {cw},
        "iia_flip": iia_flip(p),
    }


def bucket(c: dict) -> str:
    if c["condorcet"] is None:
        return "no_condorcet"
    return "cw_eq_borda" if c["cw_is_borda"] else "cw_ne_borda"


def prompt_elections(rows: list[dict], block: str) -> dict:
    """prompt_id -> list of rankings (one per annotator who gave this block)."""
    out = {}
    for r in rows:
        ranks = []
        for a in r["metadata"]["assessments"]:
            for entry in a["ranking_blocks"].get(block) or []:
                if entry.get("ranking"):
                    ranks.append(parse_ranking(entry["ranking"]))
        if ranks:
            out[r["prompt_id"]] = ranks
    return out


def bootstrap_stability(ranks: list, n_boot: int, rng: random.Random) -> dict:
    """Share of annotator resamples reproducing the full-sample bucket, its
    Condorcet winner, and the IIA flip."""
    full = classify(pairwise(ranks))
    same_bucket = same_cw = same_iia = 0
    for _ in range(n_boot):
        c = classify(pairwise([rng.choice(ranks) for _ in ranks]))
        same_bucket += bucket(c) == bucket(full)
        same_cw += c["condorcet"] == full["condorcet"]
        same_iia += c["iia_flip"] == full["iia_flip"]
    return {"bucket": same_bucket / n_boot, "condorcet": same_cw / n_boot,
            "iia_flip": same_iia / n_boot}


def summarise(elections: dict, n_boot: int, seed: int, stable_at: float = 0.9) -> dict:
    rng = random.Random(seed)
    per_prompt, counts = {}, {}
    for pid, ranks in elections.items():
        p = pairwise(ranks)
        c = classify(p)
        c["n_voters"] = len(ranks)
        c["maximal_lottery"] = [round(float(x), 4) for x in maximal_lottery(p)]
        c["ml_is_pure"] = max(c["maximal_lottery"]) > 1 - 1e-6
        c["stability"] = bootstrap_stability(ranks, n_boot, rng) if n_boot else None
        c["bucket"] = bucket(c)
        per_prompt[pid] = c
    def tally(pred):
        return sum(1 for c in per_prompt.values() if pred(c))

    def stable(c):
        return c["stability"] is None or c["stability"]["bucket"] >= stable_at

    counts = {
        "prompts": len(per_prompt),
        "voters_median": float(np.median([c["n_voters"] for c in per_prompt.values()])),
        "condorcet_winner": tally(lambda c: c["condorcet"] is not None),
        "no_condorcet_winner": tally(lambda c: c["condorcet"] is None),
        "no_condorcet_but_weak_winner": tally(
            lambda c: c["condorcet"] is None and c["weak_condorcet"]),
        "mixed_maximal_lottery": tally(lambda c: not c["ml_is_pure"]),
        "cw_eq_borda": tally(lambda c: c["cw_is_borda"]),
        "cw_ne_borda": tally(lambda c: c["condorcet"] is not None and not c["cw_is_borda"]),
        "iia_flip": tally(lambda c: c["iia_flip"]),
        f"stable_buckets_at_{stable_at}": {
            b: tally(lambda c, b=b: c["bucket"] == b and stable(c))
            for b in ("cw_eq_borda", "cw_ne_borda", "no_condorcet")},
        f"stable_iia_flip_at_{stable_at}": tally(
            lambda c: c["iia_flip"] and (c["stability"] is None
                                         or c["stability"]["iia_flip"] >= stable_at)),
    }
    return {"counts": counts, "per_prompt": per_prompt}


def prompt_report(ranks: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Everything the paper's worked example shows for one prompt: the
    pairwise matrix, Borda scores, the rules' answers, the IIA removals that
    change the Borda winner, a bootstrap tally of winners, and which single
    voters' removal changes the Borda winner."""
    p = pairwise(ranks)
    full = classify(p)
    w = sorted(borda_winners(p))
    iia = [LETTERS[d] for d in range(len(p)) if d not in w
           and borda_winners(p, [i for i in range(len(p)) if i != d]) != set(w)]
    rng = random.Random(seed)
    cw, bw, same = {}, {}, 0
    for _ in range(n_boot):
        c = classify(pairwise([rng.choice(ranks) for _ in ranks]))
        cw[str(c["condorcet"])] = cw.get(str(c["condorcet"]), 0) + 1
        bw[c["borda"]] = bw.get(c["borda"], 0) + 1
        same += bucket(c) == bucket(full)
    leave_one_out = [classify(pairwise(ranks[:i] + ranks[i + 1:]))["borda"]
                     for i in range(len(ranks))]
    return {
        "n_voters": len(ranks),
        "pairwise": [[round(float(x), 3) for x in row] for row in p],
        "borda_scores": [round(float(p[i].sum() - 0.5), 3) for i in range(len(p))],
        **full,
        "maximal_lottery": [round(float(x), 4) for x in maximal_lottery(p)],
        "iia_removals": iia,
        "bootstrap": {"n": n_boot, "condorcet": cw, "borda": bw,
                      "bucket_reproduced": same / n_boot},
        "leave_one_out_borda": leave_one_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--example", help="print the worked-example report for this prompt_id")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.data_dir / "comparisons.jsonl")]
    if args.example:
        ranks = prompt_elections(rows, "world")[args.example]
        print(json.dumps(prompt_report(ranks, max(args.boot, 1), args.seed), indent=1))
        return
    result = {}
    for block in ("world", "personal"):
        result[block] = summarise(prompt_elections(rows, block), args.boot, args.seed)
        print(f"{block}: {json.dumps(result[block]['counts'], indent=1)}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
