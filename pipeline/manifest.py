"""Stage completion manifests and skip logic.

A stage directory is complete iff it contains manifest.json with status=complete
and a params_hash matching the current parameters. manifest.json is written last,
on success, so a crashed stage is simply re-run.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

MANIFEST = "manifest.json"


def params_hash(params: dict) -> str:
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def git_sha() -> str:
    """JACKPOT_GIT_SHA wins: Modal containers have no checkout, so the client
    resolves the sha at dispatch time and bakes it into the image env."""
    env_sha = os.environ.get("JACKPOT_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(stage_dir: Path, phash: str, outputs: list[str],
                   extra: dict | None = None) -> None:
    (stage_dir / MANIFEST).write_text(json.dumps({
        "status": "complete",
        "params_hash": phash,
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "outputs": outputs,
        **(extra or {}),
    }, indent=2))


def is_complete(stage_dir: Path, phash: str) -> bool:
    mf = stage_dir / MANIFEST
    if not mf.exists():
        return False
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return False
    return m.get("status") == "complete" and m.get("params_hash") == phash


def load(stage_dir: Path) -> dict | None:
    mf = stage_dir / MANIFEST
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text())
    except json.JSONDecodeError:
        return None
