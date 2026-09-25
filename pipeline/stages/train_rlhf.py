"""Stage: RLHF baseline. A reward model is fine-tuned on the preference
triplets, then PPO runs against it (trl 0.10.1)."""

from pathlib import Path

from pipeline import manifest
from pipeline.config import prompt_for, train_params
from pipeline.costs import stage_timer
from pipeline.models import build_seq_cls, lora_config
from pipeline.populations import CONFIGS
from pipeline.stages.evaluate import parse_colour
from pipeline.stages import train_common as tc
from pipeline.storage import Storage

STAGE = "train_rlhf"


def train_reward_model(rows: list[dict], model_name: str, dtype: str, lora: dict,
                       tokenizer, *, epochs: int, center_coef: float,
                       device: str, workdir: Path):
    from datasets import Dataset
    from peft import get_peft_model
    from trl import RewardConfig, RewardTrainer

    def fmt(row):
        chosen = tokenizer(row["prompt"] + " " + row["chosen"] + ".", truncation=True, max_length=256)
        rejected = tokenizer(row["prompt"] + " " + row["rejected"] + ".", truncation=True, max_length=256)
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    ds = Dataset.from_list([fmt(r) for r in rows])
    rm = build_seq_cls(model_name, dtype, tokenizer)
    rm = get_peft_model(rm, lora_config(lora, task_type="SEQ_CLS"))
    rm.to(device)

    cfg = RewardConfig(
        output_dir=str(workdir / "rm_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        center_rewards_coefficient=center_coef,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        max_length=256,
        bf16=(dtype == "bfloat16"),
        use_cpu=(device == "cpu"),
    )
    trainer = RewardTrainer(model=rm, args=cfg, tokenizer=tokenizer, train_dataset=ds)
    trainer.train()
    rm.eval()
    return rm


def rm_score_texts(rm, tokenizer, texts: list[str], device: str) -> list:
    """Reward-model logit per full text (prompt included)."""
    import torch

    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(device)
    with torch.no_grad():
        logits = rm(**batch).logits.squeeze(-1)
    return [logits[i].float().cpu() for i in range(len(texts))]


def first_sentence(response: str) -> str:
    """The response up to and including its first full stop."""
    idx = response.find(".")
    return response if idx < 0 else response[: idx + 1]


def canonical_rm_scores(rm, tokenizer, prompt: str, alternatives, device: str) -> dict:
    """The reward model's score for each canonical answer `prompt + " <colour>."`,
    the format it was trained on, and the pairwise margins. Written to
    rm_scores.json: it shows which colour the reward model ranks first."""
    texts = [prompt + " " + a + "." for a in alternatives]
    scores = [float(x) for x in rm_score_texts(rm, tokenizer, texts, device)]
    canonical = dict(zip(alternatives, scores))
    margin = {f"{a}>{b}": round(canonical[a] - canonical[b], 4)
              for i, a in enumerate(alternatives) for b in alternatives[i + 1:]}
    return {"canonical": {a: round(v, 4) for a, v in canonical.items()}, "margin": margin}


def unparsed_reward(canonical: dict) -> float:
    """Reward for a response naming no colour: the worst canonical score minus
    the canonical spread, so every named colour beats it by at least the
    preference gap. Set by the reward model, not a knob."""
    lo, hi = min(canonical.values()), max(canonical.values())
    return float(lo - (hi - lo))


def rm_scores(rm, tokenizer, prompt: str, responses, device: str, alternatives,
              canonical: dict) -> list:
    """Reward per sampled response: the reward model's score of the canonical
    answer for the colour the response names in its first sentence, or
    `unparsed_reward` when it names none.

    The reward model only ever saw canonical answers. Scored on the sampled
    text itself (" red.\\n\\nQ: What does the") its reward followed the
    continuation rather than the colour, and PPO optimised that. Reducing a
    response to its colour leaves PPO one lever, the colour token, which is the
    same information the SPO arm's reward uses."""
    import torch

    decoded = tokenizer.batch_decode(responses, skip_special_tokens=True)
    colours = [parse_colour(first_sentence(t), alternatives) for t in decoded]
    floor = unparsed_reward(canonical)
    texts = [prompt + " " + c + "." for c in colours if c is not None]
    scored = iter(rm_score_texts(rm, tokenizer, texts, device)) if texts else iter(())
    return [torch.tensor(floor) if c is None else next(scored) for c in colours]


def run(storage: Storage, profile: dict, config_name: str, force: bool = False) -> dict:
    tp = train_params(profile, "rlhf", config_name)
    seed = profile["data"]["seed"]
    params = {"config": config_name, "method": "rlhf", "model": profile["model"]["name"],
              "seed": seed, **tp}
    phash = manifest.params_hash(params)
    out = storage.stage_dir(STAGE, config_name)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    pop = CONFIGS[config_name]
    device = profile["model"]["device"]
    rows = Storage.read_jsonl(storage.stage_dir("data_gen", config_name) / "triplets.jsonl")

    with stage_timer(out, STAGE, config=config_name, method="rlhf",
                     gpu_type=profile["gpu"]["train"]):
        tokenizer = tc.build_tokenizer(profile["model"]["name"])
        rm = train_reward_model(
            rows, profile["model"]["name"], profile["model"]["dtype"], tp["lora"],
            tokenizer, epochs=tp["rm_epochs"], center_coef=tp["center_rewards_coefficient"],
            device=device, workdir=out,
        )
        prompt = prompt_for(profile, pop.prompt_key)
        rm_diag = canonical_rm_scores(rm, tokenizer, prompt, pop.alternatives, device)
        Storage.write_json(out / "rm_scores.json", rm_diag)
        print(f"[rlhf/{config_name}] RM canonical scores {rm_diag['canonical']} "
              f"margins {rm_diag['margin']}", flush=True)

        policy = tc.build_ppo_policy(
            profile["model"]["name"], profile["model"]["dtype"], tp["lora"], tokenizer
        )
        trainer = tc.make_ppo_trainer(policy, tokenizer, tp, seed)
        query = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
        max_new = profile["eval"]["max_new_tokens"]

        loop = tc.TrainingLoop(out, policy, tokenizer, pop.alternatives, tag=f"[rlhf/{config_name}]",
                               epochs=tp["epochs"],
                               steps_per_epoch=max(1, profile["data"]["n_datapoints"] // tp["batch_size"]))
        for epoch, it in loop.epochs_and_steps():
            queries, responses = tc.generate_responses(
                trainer, query, tp["batch_size"], max_new, tokenizer.pad_token_id
            )
            rewards = rm_scores(rm, tokenizer, prompt, responses, device,
                                pop.alternatives, rm_diag["canonical"])
            stats = trainer.step(queries, responses, rewards)
            texts = tc.decode_batch(tokenizer, responses)
            parsed = tc.parse_texts(texts, pop.alternatives)
            loop.after_step(epoch=epoch, it=it, parsed=parsed, texts=texts, stats=stats,
                            extra={"reward_by_colour": tc.reward_by_colour(
                                [float(r) for r in rewards], parsed, pop.alternatives)})
    manifest.write_manifest(out, phash, ["checkpoint/", "metrics.jsonl", "samples.jsonl",
                                         "rm_scores.json"],
                            extra={"seed": seed, "base_parse_rate": loop.base_parse_rate})
    return {"skipped": False, "dir": str(out)}
