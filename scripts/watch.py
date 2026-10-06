"""Watcher for detached Modal runs. It parses only `--json` output, keys on app
ids rather than names (which `modal app list` truncates), and strips ANSI codes
before matching log text.

Each poll prints the latest trainer step line; it alerts on degeneration
(unparsed >= threshold for 3 consecutive logged steps past step 5) and on error
lines, and exits when the app ends — code 0 if the optional --pattern status
target completed, 2 if the app stopped without completing.

Usage:
  .venv/bin/python scripts/watch.py APP_ID [--interval 120] [--max-polls 40]
      [--run-id default] [--pattern 'train_ml +cyclic']

All parsing lives in pure functions, unit-tested by tests/test_watch.py.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"\[(?P<stage>spo|rlhf|ipo|ipo_offline)/(?P<config>[a-z0-9_]+)\] "
    r"step (?P<step>\d+)/(?P<total>\d+).*?unparsed=(?P<unparsed>[0-9.]+)"
)
ERROR_RE = re.compile(
    r"Traceback|out of memory|OutOfMemory|CUDA error|FAILED|Killed|CELL FAILED"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_step_lines(log_text: str) -> list[dict]:
    """Trainer progress lines, in log order."""
    steps = []
    for m in STEP_RE.finditer(strip_ansi(log_text)):
        steps.append({
            "stage": m["stage"], "config": m["config"],
            "step": int(m["step"]), "total": int(m["total"]),
            "unparsed": float(m["unparsed"]),
        })
    return steps


def degeneration(steps: list[dict], *, threshold: float = 0.4,
                 consecutive: int = 3, min_step: int = 5) -> bool:
    """Kill-candidate signal: the last `consecutive` logged steps all unparsed
    >= threshold, and training is past its noisy warm-up."""
    if len(steps) < consecutive:
        return False
    tail = steps[-consecutive:]
    return tail[-1]["step"] >= min_step and all(s["unparsed"] >= threshold for s in tail)


def find_errors(log_text: str) -> list[str]:
    return [line for line in strip_ansi(log_text).splitlines() if ERROR_RE.search(line)]


def app_state(app_list_json: str, app_id: str) -> str | None:
    """State of `app_id` from `modal app list --json`, keyed on the id.
    Tolerant of key naming across modal versions."""
    try:
        rows = json.loads(app_list_json)
    except json.JSONDecodeError:
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if app_id in (str(v) for v in row.values()):
            for key, val in row.items():
                if "state" in key.lower():
                    return str(val).strip().lower()
    return None


def is_live(state: str) -> bool:
    """Modal decorates states across versions ("ephemeral (detached)"), so
    match on the leading word, not the exact string."""
    return state.split()[0] in ("ephemeral", "running", "deploying")


def status_target_done(status_output: str, pattern: str) -> bool | None:
    """Match the status-table line for `pattern` (a regex over the row text);
    True if its checkbox is [x], None if no row matches."""
    for line in strip_ansi(status_output).splitlines():
        if re.search(pattern, line):
            return "[x]" in line
    return None


def _modal(args: list[str], timeout: int) -> str:
    try:
        res = subprocess.run([str(ROOT / ".venv" / "bin" / "modal"), *args],
                             capture_output=True, text=True, timeout=timeout)
        return res.stdout
    except subprocess.TimeoutExpired:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("app_id")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--max-polls", type=int, default=40)
    ap.add_argument("--run-id", default="default")
    ap.add_argument("--pattern", default=None,
                    help="status-row regex; exit 0 as soon as its cell is [x]")
    args = ap.parse_args()

    for poll in range(1, args.max_polls + 1):
        ts = time.strftime("%H:%MZ", time.gmtime())
        logs = _modal(["app", "logs", args.app_id], timeout=150)
        steps = parse_step_lines(logs)
        last = steps[-1] if steps else None
        last_txt = (f"[{last['stage']}/{last['config']}] step {last['step']}/{last['total']} "
                    f"unparsed={last['unparsed']:.2f}" if last else "<no step output yet>")
        print(f"[poll {poll} {ts}] {last_txt}", flush=True)

        for err in find_errors(logs)[-3:]:
            print(f"ERROR ({ts}): {err}", flush=True)
        if degeneration(steps):
            print(f"DEGENERATION_ALERT ({ts}): last steps unparsed>=0.4 — kill candidate",
                  flush=True)

        if args.pattern:
            status = _modal(["run", "pipeline/modal_app.py", "--stage", "status",
                             "--run-id", args.run_id], timeout=150)
            if status_target_done(status, args.pattern):
                print("TARGET COMPLETE", flush=True)
                return 0

        state = app_state(_modal(["app", "list", "--json"], timeout=90), args.app_id)
        if state is not None and not is_live(state):
            print(f"APP_ENDED state={state} ({ts})", flush=True)
            return 0 if args.pattern is None else 2
        time.sleep(args.interval)
    print("TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
