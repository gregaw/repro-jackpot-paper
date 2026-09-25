"""Pull artifacts off the Modal Volume through a remote function.

Collects the small per-cell files (figure, distributions, metrics, costs,
manifests) from runs/<run_id>/ on the Volume, preserving relative paths. Unlike
`modal volume get` it needs no access to Modal's storage domain, only the API.

Usage: modal run scripts/fetch_artifacts.py --run-id <run_id> --dest results/<run_id>
"""

import modal

vol = modal.Volume.from_name("jackpot-artifacts")
app = modal.App("jackpot-fetch")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/artifacts": vol}, timeout=900)
def collect(run_id: str) -> dict:
    """Return {relative_path: bytes} for every wanted file under the run."""
    from pathlib import Path

    vol.reload()
    root = Path("/artifacts") / "runs" / run_id
    wanted = ["figure2.png", "distributions.json", "cost.json", "manifest.json",
              "metrics.jsonl", "samples.jsonl", "mixture.json", "rm_scores.json"]
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in wanted:
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


@app.local_entrypoint()
def main(run_id: str = "default", dest: str = "results"):
    from pathlib import Path

    files = collect.remote(run_id)
    for rel, data in files.items():
        target = Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"{len(data):>9} B  {target}")
    print(f"{len(files)} files -> {dest}/")
