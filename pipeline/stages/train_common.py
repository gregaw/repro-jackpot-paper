"""Shared scaffolding for both trainers (trl 0.10.1 PPOTrainer, the version the paper used)."""

import math
from pathlib import Path

from pipeline.models import TINY_SENTINEL, build_tokenizer, lora_config, tiny_gemma2_config
from pipeline.stages.evaluate import parse_colour
from pipeline.storage import Storage


def build_ppo_policy(model_name: str, dtype: str, lora: dict, tokenizer):
    """Policy with value head + LoRA adapter. With ref_model=None the trainer
    computes reference logprobs by disabling the adapter."""
    import torch
    from trl import AutoModelForCausalLMWithValueHead

    peft_cfg = lora_config(lora, task_type="CAUSAL_LM")
    if model_name == TINY_SENTINEL:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_config(tiny_gemma2_config(len(tokenizer)))
        return AutoModelForCausalLMWithValueHead.from_pretrained(base, peft_config=peft_cfg)
    return AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, peft_config=peft_cfg, torch_dtype=getattr(torch, dtype),
        attn_implementation="eager",   # Gemma 2: sdpa skips logit softcapping and NaNs
    )


def make_ppo_trainer(policy, tokenizer, tp: dict, seed: int):
    from trl import PPOConfig, PPOTrainer

    cfg = PPOConfig(
        batch_size=tp["batch_size"],
        mini_batch_size=tp["mini_batch_size"],
        gradient_accumulation_steps=tp["gradient_accumulation_steps"],
        ppo_epochs=tp.get("ppo_epochs", 4),
        learning_rate=tp["learning_rate"],
        init_kl_coef=tp["init_kl_coef"],
        adap_kl_ctrl=False,            # fixed anchor, not adaptive
        vf_coef=tp["vf_coef"],
        gamma=tp["gamma"],
        seed=seed,
    )
    return PPOTrainer(cfg, policy, ref_model=None, tokenizer=tokenizer)


GEN_KWARGS = dict(do_sample=True, temperature=1.0, top_k=0.0, top_p=1.0)


def generate_responses(ppo_trainer, query_tensor, batch_size: int, max_new_tokens: int,
                       pad_token_id: int) -> tuple[list, list]:
    queries = [query_tensor] * batch_size
    responses = ppo_trainer.generate(
        queries, return_prompt=False, batch_size=min(64, batch_size),
        max_new_tokens=max_new_tokens, pad_token_id=pad_token_id, **GEN_KWARGS,
    )
    return queries, responses


def decode_batch(tokenizer, responses) -> list[str]:
    return tokenizer.batch_decode(responses, skip_special_tokens=True)


def parse_texts(texts: list[str], alternatives) -> list:
    return [parse_colour(t, alternatives) for t in texts]


def epoch_distribution(parsed: list, alternatives) -> dict:
    valid = [c for c in parsed if c is not None]
    total = max(1, len(valid))
    return {alt: valid.count(alt) / total for alt in alternatives}


def distribution_entropy(dist: dict) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def save_checkpoint(policy, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out_dir))       # LoRA adapter + value head
    tokenizer.save_pretrained(str(out_dir))


def extract_kl(stats) -> float | None:
    """trl's PPOTrainer.step stats; key and type vary across versions."""
    try:
        kl = stats.get("objective/kl")
        return None if kl is None else float(kl)
    except Exception:
        return None


def reward_by_colour(rewards: list[float], parsed: list, alternatives) -> dict:
    """Mean reward per sampled colour in a batch (plus "unparsed"), with counts.
    Colours absent from the batch are omitted."""
    out: dict = {}
    for key in list(alternatives) + [None]:
        vals = [r for r, c in zip(rewards, parsed) if c == key]
        if vals:
            out["unparsed" if key is None else key] = {
                "mean": round(sum(vals) / len(vals), 4), "n": len(vals)}
    return out


def record_step_metrics(metrics_path: Path, *, step: int, parsed: list,
                        alternatives, kl: float | None, extra: dict | None = None) -> None:
    """Per-step health record. Never carries a "dist" key: plot.py reads "dist"
    rows as the per-epoch series."""
    dist = epoch_distribution(parsed, alternatives)
    rec = {
        "kind": "step",
        "step": step,
        "parse_rate": round(1.0 - parsed.count(None) / max(1, len(parsed)), 4),
        "entropy": round(distribution_entropy(dist), 4),
    }
    if kl is not None:
        rec["kl"] = round(kl, 4)
    if extra:
        assert "dist" not in extra, "'dist' is reserved for per-epoch rows"
        rec.update(extra)
    Storage.append_jsonl(metrics_path, rec)


class MixtureAccumulator:
    """Algorithm 1's output policy: the uniform mixture of the iterates.

    For a fixed prompt the mixture's output distribution is the average of the
    per-iterate output distributions, so pooling every step's sampled batch
    (constant batch size) gives it without storing T adapters. Feed it the
    batch parsed before the forced-uniform substitution: substituted samples
    are not draws from the policy."""

    def __init__(self, alternatives):
        self.alternatives = tuple(alternatives)
        self.counts = {alt: 0 for alt in self.alternatives}
        self.unparsed = 0
        self.n_steps = 0

    def observe(self, parsed: list) -> None:
        for c in parsed:
            if c is None:
                self.unparsed += 1
            else:
                self.counts[c] += 1
        self.n_steps += 1

    def result(self) -> dict:
        total = self.unparsed + sum(self.counts.values())
        parsed_total = max(1, sum(self.counts.values()))
        return {
            "distribution": {a: self.counts[a] / parsed_total for a in self.alternatives},
            "distribution_unconditional": {
                a: self.counts[a] / max(1, total) for a in self.alternatives
            },
            "counts": dict(self.counts),
            "unparsed": self.unparsed,
            "n_samples": total,
            "n_steps": self.n_steps,
        }


def save_raw_samples(samples_path: Path, *, epoch: int, step: int, texts: list[str],
                     n: int = 20) -> None:
    """A few raw completions per epoch: the cheapest diagnostic there is."""
    Storage.append_jsonl(samples_path, {"epoch": epoch, "step": step, "texts": texts[:n]})


class CheckpointGate:
    """Keep-last-healthy checkpoint gate.

    PPO cells drift into degenerate generation (unparsed near 1.0) at
    unpredictable points, so an end-of-training save is a lottery. The gate
    overwrites the checkpoint every `every` steps, but only while the mean
    unparsed fraction over the last `window` steps is at most
    `healthy_max_unparsed`. Checkpoint selection only; training is untouched.
    Every decision is recorded in metrics.jsonl (kind="gate")."""

    def __init__(self, policy, tokenizer, ckpt_dir: Path, metrics_path: Path, *,
                 tag: str = "", every: int = 8, healthy_max_unparsed: float = 0.3,
                 window: int = 4, save_fn=None):
        self.policy, self.tokenizer = policy, tokenizer
        self.ckpt_dir, self.metrics_path, self.tag = ckpt_dir, metrics_path, tag
        self.every, self.healthy_max_unparsed, self.window = every, healthy_max_unparsed, window
        self.save_fn = save_fn or save_checkpoint
        self.recent: list[float] = []
        self.ckpt_step: int | None = None

    def observe(self, step_no: int, total_steps: int, unparsed_frac: float) -> bool:
        """Call once per training step; saves and logs when a gate is due."""
        self.recent = (self.recent + [unparsed_frac])[-self.window:]
        if step_no % self.every != 0 and step_no != total_steps:
            return False
        recent_mean = sum(self.recent) / len(self.recent)
        saved = recent_mean <= self.healthy_max_unparsed
        if saved:
            self.save_fn(self.policy, self.tokenizer, self.ckpt_dir)
            self.ckpt_step = step_no
            print(f"{self.tag} checkpoint saved at step {step_no} (healthy)", flush=True)
        Storage.append_jsonl(self.metrics_path, {
            "kind": "gate", "step": step_no, "saved": saved,
            "recent_unparsed": round(recent_mean, 4),
        })
        return saved

    def finalize(self, total_steps: int) -> None:
        if self.ckpt_step is None:
            # never healthy at a gate: save the final policy so the cell can be
            # inspected, and say so
            self.save_fn(self.policy, self.tokenizer, self.ckpt_dir)
            print(f"{self.tag} WARNING: no healthy gate hit; "
                  f"saved final (degenerate) policy", flush=True)
        Storage.append_jsonl(self.metrics_path, {
            "checkpoint_step": self.ckpt_step or total_steps,
            "gated": self.ckpt_step is not None,
        })


class TrainingLoop:
    """The bookkeeping both trainers share: step counting, per-step metrics,
    raw samples at the end of each epoch, the checkpoint gate, and per-epoch
    distribution rows for plot.py."""

    def __init__(self, out_dir: Path, policy, tokenizer, alternatives, *,
                 tag: str, epochs: int, steps_per_epoch: int):
        import time

        self.out, self.alternatives, self.tag = out_dir, alternatives, tag
        self.epochs, self.steps_per_epoch = epochs, steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.metrics_path = out_dir / "metrics.jsonl"
        self.samples_path = out_dir / "samples.jsonl"
        self.metrics_path.unlink(missing_ok=True)
        self.samples_path.unlink(missing_ok=True)
        self.gate = CheckpointGate(policy, tokenizer, out_dir / "checkpoint",
                                   self.metrics_path, tag=tag)
        self.step_no = 0
        self.base_parse_rate: float | None = None   # step 1 = fresh LoRA = base model
        self._epoch_parsed: list = []
        self._t0 = time.monotonic()

    def epochs_and_steps(self):
        for epoch in range(self.epochs):
            self._epoch_parsed = []
            for it in range(self.steps_per_epoch):
                yield epoch, it
            Storage.append_jsonl(self.metrics_path, {
                "step": epoch + 1,
                "dist": epoch_distribution(self._epoch_parsed, self.alternatives),
                "unparsed_frac": self._epoch_parsed.count(None) / max(1, len(self._epoch_parsed)),
            })
        self.gate.finalize(self.total_steps)

    def after_step(self, *, epoch: int, it: int, parsed: list, texts: list[str],
                   stats, extra: dict | None = None) -> None:
        import time

        self.step_no += 1
        self._epoch_parsed.extend(parsed)
        unparsed = parsed.count(None) / max(1, len(parsed))
        if self.base_parse_rate is None:
            self.base_parse_rate = round(1.0 - unparsed, 4)
        record_step_metrics(self.metrics_path, step=self.step_no, parsed=parsed,
                            alternatives=self.alternatives, kl=extract_kl(stats), extra=extra)
        if it == self.steps_per_epoch - 1:
            save_raw_samples(self.samples_path, epoch=epoch + 1, step=self.step_no, texts=texts)
        elapsed = time.monotonic() - self._t0
        print(f"{self.tag} step {self.step_no}/{self.total_steps} "
              f"({elapsed:.0f}s, {elapsed / self.step_no:.1f}s/step) "
              f"unparsed={unparsed:.2f} dist={epoch_distribution(parsed, self.alternatives)}",
              flush=True)
        self.gate.observe(self.step_no, self.total_steps, unparsed)
