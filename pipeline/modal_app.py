"""Modal wrapper: the stage functions as GPU jobs plus a remote orchestrator.

`modal run --detach` dispatches orchestrate() and returns at once; the run then
lives on Modal and survives the client disconnecting. Poll with `--stage status`.

  modal run --detach pipeline/modal_app.py                            # full pipeline
  modal run pipeline/modal_app.py --stage status                      # completion table
  modal run --detach pipeline/modal_app.py --stage train --config cyclic --method ml
  modal run --detach pipeline/modal_app.py --stage eval --force
  modal run pipeline/modal_app.py --stage chat --config cyclic --method ml

JACKPOT_PROFILE=qwen selects configs/qwen.yaml instead of experiments.yaml.
"""

import os
from pathlib import Path

import modal

from pipeline import manifest as _manifest
from pipeline.config import load_profile

ROOT = Path(__file__).resolve().parent.parent

app = modal.App("jackpot-paper")

# Resolved on the client (containers have no git checkout) and baked into the
# image, so every manifest records the sha that produced it.
GIT_SHA = _manifest.git_sha()
PROFILE_NAME = os.environ.get("JACKPOT_PROFILE", "paper")
# GPU types are fixed when the function decorators run, so they come from the
# profile at import time, not from CLI flags.
_GPU = load_profile(PROFILE_NAME)["gpu"]
GPU_TRAIN, GPU_EVAL = _GPU["train"], _GPU["eval"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "configs" / "requirements-gpu.txt"))
    .env({"HF_HOME": "/artifacts/hf_cache",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "JACKPOT_PROFILE": PROFILE_NAME,
          "JACKPOT_GIT_SHA": GIT_SHA})
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs")
    .add_local_python_source("pipeline")
)

vol = modal.Volume.from_name("jackpot-artifacts", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
VOLUME = {"/artifacts": vol}


def _storage(run_id: str):
    from pipeline.storage import Storage

    return Storage("/artifacts", run_id)


def _profile(seed: int | None = None):
    return load_profile(os.environ.get("JACKPOT_PROFILE", "paper"), seed=seed)


@app.function(image=image, volumes=VOLUME, secrets=[hf_secret], timeout=3600)
def warm_cache() -> str:
    """Download the model into the Volume once, before any GPU fan-out, so
    parallel containers never race to write the same cache paths."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(_profile()["model"]["name"])
    vol.commit()
    return path


@app.function(image=image, volumes=VOLUME, timeout=600)
def gen_data(config: str, run_id: str, force: bool, seed: int | None = None) -> dict:
    from pipeline.stages import data_gen

    res = data_gen.run(_storage(run_id), _profile(seed), config, force=force)
    vol.commit()
    return res


@app.function(image=image, volumes=VOLUME, secrets=[hf_secret], gpu=GPU_TRAIN,
              timeout=14400, max_containers=4)
def train(config: str, method: str, run_id: str, force: bool, seed: int | None = None) -> dict:
    from pipeline.runner import train_one

    res = train_one(_storage(run_id), _profile(seed), config, method, force=force)
    vol.commit()
    return res


@app.function(image=image, volumes=VOLUME, secrets=[hf_secret], gpu=GPU_EVAL,
              timeout=3600, max_containers=4)
def evaluate(config: str, method: str, run_id: str, force: bool, seed: int | None = None) -> dict:
    from pipeline.runner import make_policy_loader
    from pipeline.stages import evaluate as evaluate_stage

    storage, profile = _storage(run_id), _profile(seed)
    res = evaluate_stage.run(storage, profile, config, method,
                             make_policy_loader(storage, profile), force=force)
    vol.commit()
    return res


@app.function(image=image, volumes=VOLUME, timeout=600)
def plot(run_id: str, force: bool, seed: int | None = None) -> dict:
    from pipeline.stages import plot as plot_stage

    res = plot_stage.run(_storage(run_id), _profile(seed), force=force)
    vol.commit()
    return res


@app.function(image=image, volumes=VOLUME, timeout=86400)
def orchestrate(stage: str, config: str, method: str, run_id: str, force: bool,
                seed: int | None = None) -> dict:
    """Runs on Modal and survives the client disconnecting. A failed cell does
    not stop the run (starmap with return_exceptions), so every failure is
    printed and a status table follows every completed cell."""
    from pipeline.runner import format_status, status

    profile = _profile(seed)
    configs = profile["configs"] if config == "all" else [config]
    methods = profile["methods"] if method == "all" else [method]
    results: dict = {}

    def want(s: str) -> bool:
        return stage in ("all", s)

    def note(stage_name: str, r) -> None:
        if isinstance(r, Exception):
            print(f"[orchestrate] CELL FAILED ({stage_name}): {r!r}", flush=True)
        vol.reload()
        print(format_status(status(_storage(run_id), profile)), flush=True)

    if want("train") or want("eval"):
        results["warm_cache"] = warm_cache.remote()
    if want("data_gen") or want("train"):
        results["data_gen"] = list(gen_data.starmap([(c, run_id, force, seed) for c in configs]))
    for stage_name, fn in (("train", train), ("eval", evaluate)):
        if want(stage_name):
            pairs = [(c, m, run_id, force, seed) for c in configs for m in methods]
            for r in fn.starmap(pairs, return_exceptions=True):
                results.setdefault(stage_name, []).append(repr(r) if isinstance(r, Exception) else r)
                note(stage_name, r)
    if want("plot"):
        results["plot"] = plot.remote(run_id, force, seed)
    return results


@app.function(image=image, volumes=VOLUME, timeout=300)
def status_table(run_id: str) -> list:
    from pipeline.runner import status

    vol.reload()
    return status(_storage(run_id), _profile())


@app.cls(image=image, volumes=VOLUME, secrets=[hf_secret], gpu=GPU_EVAL,
         scaledown_window=300)
class Chat:
    """A trained policy loaded once per container, for interactive sampling."""

    config: str = modal.parameter()
    method: str = modal.parameter()
    run_id: str = modal.parameter(default="default")

    @modal.enter()
    def load(self):
        from pipeline.runner import make_policy_loader

        loader = make_policy_loader(_storage(self.run_id), _profile())
        self.model, self.tokenizer, self.device = loader(self.config, self.method)

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=True, temperature=1.0,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)


@app.local_entrypoint()
def main(stage: str = "all", config: str = "all", method: str = "all",
         run_id: str = "default", force: bool = False, seed: int = -1):
    # --seed overrides the profile seed and suffixes the run id, so seeds land
    # in separate run trees.
    seed_opt = None if seed < 0 else seed
    if seed_opt is not None:
        run_id = f"{run_id}-s{seed_opt}"
    if stage == "status":
        from pipeline.runner import format_status

        print(format_status(status_table.remote(run_id)))
        return
    if stage == "chat":
        from pipeline.config import prompt_for
        from pipeline.populations import CONFIGS

        chat = Chat(config=config, method=method, run_id=run_id)
        default_prompt = prompt_for(load_profile(PROFILE_NAME), CONFIGS[config].prompt_key)
        print("Enter prompts (blank line = the paper's prompt, Ctrl-D to quit):")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            print(chat.generate.remote(line or default_prompt))
        return

    call = orchestrate.spawn(stage, config, method, run_id, force, seed_opt)
    print(f"dispatched: {call.object_id} (run_id={run_id}, profile={PROFILE_NAME}, "
          f"gpu={GPU_TRAIN}/{GPU_EVAL}, seed={'profile default' if seed_opt is None else seed_opt}, "
          f"git_sha={GIT_SHA})")
    print(f"poll with: modal run pipeline/modal_app.py --stage status --run-id {run_id}")
