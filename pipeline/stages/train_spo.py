"""Stage: Maximal Lottery policy via SPO (paper Algorithm 1).

Each iteration samples a batch of completions for the fixed prompt (10% replaced
by uniformly random colour words, per A.8), scores sample i with
r_i = mean_j P(colour_i > colour_j) under the empirical preference function of
the generated dataset, and takes a PPO step. The output policy is the uniform
mixture of the iterates, accumulated during training into mixture.json.
"""

import random

from pipeline import manifest
from pipeline.config import prompt_for, train_params
from pipeline.costs import stage_timer
from pipeline.populations import CONFIGS, empirical_preference
from pipeline.stages import train_common as tc
from pipeline.storage import Storage

STAGE = "train_ml"


def preference_rewards(parsed: list, pref) -> list:
    """r_i = (1/(k-1)) sum_{j != i} P(a_i > a_j). None (no colour named) counts
    as an alternative absent from the dataset, so it loses to any named one."""
    k = len(parsed)
    rewards = []
    for i in range(k):
        total = 0.0
        for j in range(k):
            if i == j:
                continue
            a, b = parsed[i], parsed[j]
            if a is None and b is None:
                total += 0.5
            elif a is None:
                total += 0.0
            elif b is None:
                total += 1.0
            else:
                total += pref(a, b)
        rewards.append(total / max(1, k - 1))
    return rewards


def run(storage: Storage, profile: dict, config_name: str, force: bool = False) -> dict:
    import torch

    tp = train_params(profile, "ml", config_name)
    seed = profile["data"]["seed"]
    params = {"config": config_name, "method": "ml", "model": profile["model"]["name"],
              "seed": seed, **tp}
    phash = manifest.params_hash(params)
    out = storage.stage_dir(STAGE, config_name)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    pop = CONFIGS[config_name]
    device = profile["model"]["device"]
    rng = random.Random(f"spo-{seed}-{config_name}")
    rows = Storage.read_jsonl(storage.stage_dir("data_gen", config_name) / "triplets.jsonl")
    pref = empirical_preference(rows, pop.alternatives)

    with stage_timer(out, STAGE, config=config_name, method="ml",
                     gpu_type=profile["gpu"]["train"]):
        tokenizer = tc.build_tokenizer(profile["model"]["name"])
        policy = tc.build_ppo_policy(
            profile["model"]["name"], profile["model"]["dtype"], tp["lora"], tokenizer
        )
        trainer = tc.make_ppo_trainer(policy, tokenizer, tp, seed)
        prompt = prompt_for(profile, pop.prompt_key)
        query = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
        forced_ids = {
            c: tokenizer(" " + c, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
            for c in pop.alternatives
        }
        max_new = profile["eval"]["max_new_tokens"]
        n_forced = int(tp["uniform_colour_fraction"] * tp["batch_size"])

        loop = tc.TrainingLoop(out, policy, tokenizer, pop.alternatives, tag=f"[spo/{config_name}]",
                               epochs=tp["epochs"],
                               steps_per_epoch=max(1, profile["data"]["n_datapoints"] // tp["batch_size"]))
        mixture = tc.MixtureAccumulator(pop.alternatives)
        for epoch, it in loop.epochs_and_steps():
            queries, responses = tc.generate_responses(
                trainer, query, tp["batch_size"], max_new, tokenizer.pad_token_id
            )
            # The mixture and the metrics describe the policy's own outputs,
            # parsed before the forced-uniform substitution.
            texts = tc.decode_batch(tokenizer, responses)
            policy_parsed = tc.parse_texts(texts, pop.alternatives)
            mixture.observe(policy_parsed)
            train_parsed = list(policy_parsed)
            for idx in rng.sample(range(len(responses)), n_forced):
                colour = rng.choice(pop.alternatives)
                responses[idx] = forced_ids[colour]
                train_parsed[idx] = colour
            rewards = [torch.tensor(r) for r in preference_rewards(train_parsed, pref)]
            stats = trainer.step(queries, responses, rewards)
            loop.after_step(epoch=epoch, it=it, parsed=policy_parsed, texts=texts, stats=stats)
        Storage.write_json(out / "mixture.json", mixture.result())
    manifest.write_manifest(out, phash,
                            ["checkpoint/", "metrics.jsonl", "samples.jsonl", "mixture.json"],
                            extra={"seed": seed, "base_parse_rate": loop.base_parse_rate})
    return {"skipped": False, "dir": str(out)}
