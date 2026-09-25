"""Stage: synthetic preference-triplet generation (paper Section 6.1).

Per datapoint: sample two distinct alternatives uniformly without replacement,
sample a voter from the population, orient (chosen, rejected) by that voter's
ranking.
"""

import json
import random

from pipeline import manifest
from pipeline.config import prompt_for
from pipeline.costs import stage_timer
from pipeline.populations import CONFIGS
from pipeline.storage import Storage

STAGE = "data_gen"


def sample_triplets(config_name: str, n: int, seed: int, prompt: str) -> list[dict]:
    pop = CONFIGS[config_name]
    rng = random.Random(f"{seed}-{config_name}")
    voter_rankings = [r for w, r in pop.voters for _ in range(w)]
    rows = []
    for _ in range(n):
        a, b = rng.sample(pop.alternatives, 2)
        ranking = rng.choice(voter_rankings)
        if ranking.index(a) < ranking.index(b):
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def run(storage: Storage, profile: dict, config_name: str, force: bool = False) -> dict:
    params = {
        "config": config_name,
        "n": profile["data"]["n_datapoints"],
        "seed": profile["data"]["seed"],
    }
    phash = manifest.params_hash(params)
    out = storage.stage_dir(STAGE, config_name)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    with stage_timer(out, STAGE, config=config_name):
        prompt = prompt_for(profile, CONFIGS[config_name].prompt_key)
        rows = sample_triplets(config_name, params["n"], params["seed"], prompt)
        triplets = out / "triplets.jsonl"
        triplets.write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest.write_manifest(out, phash, ["triplets.jsonl"])
    return {"skipped": False, "dir": str(out), "n": len(rows)}
