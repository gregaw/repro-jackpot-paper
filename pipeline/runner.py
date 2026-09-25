"""Stage wiring: run the pipeline (or a subset) against a Storage root.

Pure local execution — no Modal imports. modal_app.py calls these same
functions inside containers; the e2e test calls them directly on CPU.
"""

from pipeline.models import build_causal_lm, build_tokenizer
from pipeline.stages import data_gen, evaluate, plot, train_rlhf, train_spo
from pipeline.storage import Storage

STAGES = ("data_gen", "train", "eval", "plot")


def make_policy_loader(storage: Storage, profile: dict):
    """(config, method) -> (model, tokenizer, device): base model + trained LoRA adapter."""

    def load_policy(config_name: str, method: str):
        from peft import PeftModel

        ckpt = storage.stage_dir(f"train_{method}", config_name) / "checkpoint"
        tokenizer = build_tokenizer(profile["model"]["name"])
        base = build_causal_lm(profile["model"]["name"], profile["model"]["dtype"], tokenizer)
        model = PeftModel.from_pretrained(base, str(ckpt))
        device = profile["model"]["device"]
        model.to(device)
        return model, tokenizer, device

    return load_policy


def train_one(storage: Storage, profile: dict, config_name: str, method: str,
              force: bool = False) -> dict:
    mod = train_spo if method == "ml" else train_rlhf
    return mod.run(storage, profile, config_name, force=force)


def run_pipeline(storage: Storage, profile: dict, *, stage: str = "all",
                 config: str = "all", method: str = "all", force: bool = False) -> dict:
    configs = profile["configs"] if config == "all" else [config]
    methods = profile["methods"] if method == "all" else [method]
    results: dict = {}

    def want(s: str) -> bool:
        return stage in ("all", s)

    if want("data_gen"):
        for c in configs:
            results[f"data_gen/{c}"] = data_gen.run(storage, profile, c, force=force)
    if want("train"):
        for c in configs:
            for m in methods:
                results[f"train_{m}/{c}"] = train_one(storage, profile, c, m, force=force)
    if want("eval"):
        loader = make_policy_loader(storage, profile)
        for c in configs:
            for m in methods:
                results[f"eval/{c}/{m}"] = evaluate.run(
                    storage, profile, c, m, loader, force=force
                )
    if want("plot"):
        results["plot"] = plot.run(storage, profile, force=force)
    return results


def status(storage: Storage, profile: dict) -> list[dict]:
    """Completion table across all cells, for polling detached runs."""
    from pipeline import manifest

    table = []
    for c in profile["configs"]:
        table.append({"stage": "data_gen", "config": c, "method": "-",
                      "done": manifest.load(storage.stage_dir("data_gen", c)) is not None})
        for m in profile["methods"]:
            table.append({"stage": f"train_{m}", "config": c, "method": m,
                          "done": manifest.load(storage.stage_dir(f"train_{m}", c)) is not None})
            table.append({"stage": "eval", "config": c, "method": m,
                          "done": manifest.load(storage.stage_dir("eval", c, m)) is not None})
    table.append({"stage": "plot", "config": "-", "method": "-",
                  "done": manifest.load(storage.stage_dir("plot")) is not None})
    return table


def format_status(table: list[dict]) -> str:
    """Checkbox table for humans and log-greps alike; printed by the client
    status command and by the orchestrator after every completed cell."""
    done = sum(r["done"] for r in table)
    lines = [f"  [{'x' if r['done'] else ' '}] {r['stage']:<12} {r['config']:<10} {r['method']}"
             for r in table]
    lines.append(f"{done}/{len(table)} complete")
    return "\n".join(lines)
