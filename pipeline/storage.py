"""Path-rooted artifact storage.

The same class serves local runs (rooted at any directory, e.g. a pytest tmp_path)
and Modal runs (rooted at the Volume mount /artifacts). Stage code never builds
absolute paths itself; it asks Storage for stage directories.
"""

import json
from pathlib import Path


class Storage:
    def __init__(self, root: str | Path, run_id: str = "default"):
        self.root = Path(root)
        self.run_id = run_id

    def stage_dir(self, stage: str, *parts: str) -> Path:
        d = self.root / "runs" / self.run_id / stage
        for p in parts:
            d = d / p
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def write_json(path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True))

    @staticmethod
    def read_json(path: Path):
        return json.loads(path.read_text())

    @staticmethod
    def append_jsonl(path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")

    @staticmethod
    def read_jsonl(path: Path) -> list:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
