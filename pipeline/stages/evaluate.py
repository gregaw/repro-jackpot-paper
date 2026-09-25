"""Stage: sample a trained policy, parse the chosen colour, output the empirical
distribution over alternatives -> distributions.json."""

import json
from pathlib import Path

from pipeline import manifest
from pipeline.config import prompt_for
from pipeline.costs import stage_timer
from pipeline.populations import CONFIGS
from pipeline.storage import Storage

STAGE = "eval"


def parse_colour(completion: str, alternatives: tuple[str, ...]) -> str | None:
    """First alternative word appearing in the completion, else None."""
    text = completion.lower()
    hits = {alt: text.find(alt) for alt in alternatives if alt in text}
    if not hits:
        return None
    return min(hits, key=hits.get)


def sample_distribution(model, tokenizer, prompt: str, alternatives: tuple[str, ...],
                        n_samples: int, batch_size: int, max_new_tokens: int,
                        seed: int, device: str, tag: str = "eval",
                        n_raw_samples: int = 20) -> dict:
    import time

    import torch

    torch.manual_seed(seed)
    model.eval()
    counts = {alt: 0 for alt in alternatives}
    unparsed = 0
    raw_samples: list[str] = []
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    done = 0
    t0 = time.monotonic()
    with torch.no_grad():
        while done < n_samples:
            bsz = min(batch_size, n_samples - done)
            out = model.generate(
                **{k: v.expand(bsz, -1) for k, v in inputs.items()},
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            completions = tokenizer.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            for c in completions:
                if len(raw_samples) < n_raw_samples:
                    raw_samples.append(c)
                colour = parse_colour(c, alternatives)
                if colour is None:
                    unparsed += 1
                else:
                    counts[colour] += 1
            done += bsz
            # heartbeat: eval cells used to emit no log lines at all, leaving
            # "app stopped" as the only terminal signal on a detached run
            print(f"[{tag}] {done}/{n_samples} sampled "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
    total = max(1, sum(counts.values()))
    return {
        # conditional on parsing: P(colour | named a colour) — kept for
        # comparability with earlier runs, but not what the paper claims
        "distribution": {alt: counts[alt] / total for alt in alternatives},
        # unconditional: P(colour) over every sample drawn, unparsed included
        # in the denominator — the quantity the paper's claims are about
        "distribution_unconditional": {
            alt: counts[alt] / max(1, n_samples) for alt in alternatives
        },
        "counts": counts,
        "unparsed": unparsed,
        "n_samples": n_samples,
        "raw_samples": raw_samples,
    }


def run(storage: Storage, profile: dict, config_name: str, method: str,
        load_policy, force: bool = False) -> dict:
    """load_policy: callable (config_name, method) -> (model, tokenizer, device).

    Injected so the same stage code serves smoke (tiny local model) and Modal
    (base Gemma + LoRA adapter from the Volume).
    """
    ev = profile["eval"]
    params = {"config": config_name, "method": method, **ev}
    phash = manifest.params_hash(params)
    out = storage.stage_dir(STAGE, config_name, method)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    pop = CONFIGS[config_name]
    with stage_timer(out, STAGE, config=config_name, method=method,
                     gpu_type=profile["gpu"]["eval"]):
        model, tokenizer, device = load_policy(config_name, method)
        result = sample_distribution(
            model, tokenizer,
            prompt=prompt_for(profile, pop.prompt_key),
            alternatives=pop.alternatives,
            n_samples=ev["n_samples"],
            batch_size=ev["batch_size"],
            max_new_tokens=ev["max_new_tokens"],
            seed=ev["seed"],
            device=device,
            tag=f"eval/{config_name}/{method}",
        )
        raw_samples = result.pop("raw_samples")
        (out / "samples.jsonl").unlink(missing_ok=True)   # fresh on --force re-runs
        Storage.append_jsonl(out / "samples.jsonl", {"texts": raw_samples})
        if method == "ml":
            # Algorithm 1's output policy is the uniform mixture of the
            # training iterates; train_spo accumulates its output distribution
            # into mixture.json. Carry it here so verdicts.py can judge the
            # mixture rather than the single sampled checkpoint.
            mixture_path = storage.stage_dir("train_ml", config_name) / "mixture.json"
            if mixture_path.exists():
                mixture = json.loads(mixture_path.read_text())
                mixture.pop("n_steps", None)
                result["mixture"] = mixture
        Storage.write_json(out / "distributions.json", result)
    manifest.write_manifest(out, phash, ["distributions.json", "samples.jsonl"])
    return {"skipped": False, "dir": str(out), **result}
