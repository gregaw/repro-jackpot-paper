"""Unit tests for scripts/watch.py parsing — the pure functions whose shell
predecessors misfired on ANSI codes and truncated app names."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import watch  # noqa: E402

ANSI_LOG = (
    "\x1b[36m[spo/cyclic] step 12/480 (105s, 8.8s/step) unparsed=0.02 "
    "dist={'red': 0.4}\x1b[0m\n"
    "\x1b[36m[rlhf/majority] step 13/256 (110s, 8.5s/step) unparsed=1.00 "
    "dist={'red': 0.0}\x1b[0m\n"
)


def test_strip_ansi():
    assert "\x1b" not in watch.strip_ansi(ANSI_LOG)


def test_parse_step_lines_through_ansi():
    steps = watch.parse_step_lines(ANSI_LOG)
    assert steps == [
        {"stage": "spo", "config": "cyclic", "step": 12, "total": 480, "unparsed": 0.02},
        {"stage": "rlhf", "config": "majority", "step": 13, "total": 256, "unparsed": 1.0},
    ]


def test_degeneration_needs_three_consecutive_past_warmup():
    healthy = [{"step": s, "unparsed": 0.02} for s in range(1, 10)]
    assert not watch.degeneration(healthy)
    collapse = healthy + [{"step": s, "unparsed": 0.9} for s in (10, 11, 12)]
    assert watch.degeneration(collapse)
    early = [{"step": s, "unparsed": 0.9} for s in (1, 2, 3)]   # warm-up noise
    assert not watch.degeneration(early)
    blip = healthy + [{"step": 10, "unparsed": 0.9}]            # single blip
    assert not watch.degeneration(blip)


def test_find_errors():
    log = "ok line\nTraceback (most recent call last):\n  boom\nCUDA error: OOM\n"
    errs = watch.find_errors(log)
    assert len(errs) == 2 and "Traceback" in errs[0]


def test_app_state_keyed_on_id_not_name():
    rows = [
        {"App ID": "ap-xyz", "Description": "jackpot-paper truncated…", "State": "stopped"},
        {"App ID": "ap-abc", "Description": "jackpot-paper", "State": "ephemeral"},
    ]
    text = json.dumps(rows)
    assert watch.app_state(text, "ap-abc") == "ephemeral"
    assert watch.app_state(text, "ap-xyz") == "stopped"
    assert watch.app_state(text, "ap-missing") is None
    assert watch.app_state("not json", "ap-abc") is None


def test_status_target_done():
    status = "  [x] train_ml     cyclic     ml\n  [ ] eval         cyclic     ml\n"
    assert watch.status_target_done(status, r"train_ml +cyclic") is True
    assert watch.status_target_done(status, r"eval +cyclic") is False
    assert watch.status_target_done(status, r"nonexistent") is None


def test_is_live_tolerates_decorated_states():
    # modal 1.x reports "ephemeral (detached)"; the exact-string check made the
    # watcher declare APP_ENDED on poll 1 of a live detached run
    assert watch.is_live("ephemeral (detached)")
    assert watch.is_live("ephemeral")
    assert watch.is_live("running")
    assert not watch.is_live("stopped")
    assert not watch.is_live("stopped (detached)")
