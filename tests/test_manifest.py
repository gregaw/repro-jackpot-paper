"""Provenance: JACKPOT_GIT_SHA env override (Modal containers have no checkout)
and extra manifest fields (seed, base_parse_rate)."""

import json

from pipeline import manifest


def test_git_sha_env_override(monkeypatch):
    monkeypatch.setenv("JACKPOT_GIT_SHA", "abc1234")
    assert manifest.git_sha() == "abc1234"


def test_git_sha_falls_back_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("JACKPOT_GIT_SHA", raising=False)
    monkeypatch.chdir(tmp_path)          # not a git checkout
    assert manifest.git_sha() == "unknown"


def test_write_manifest_extra_fields(tmp_path):
    manifest.write_manifest(tmp_path, "hash123", ["metrics.jsonl"],
                            extra={"seed": 7, "base_parse_rate": 0.95})
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["seed"] == 7 and m["base_parse_rate"] == 0.95
    assert m["status"] == "complete" and m["params_hash"] == "hash123"
    assert manifest.is_complete(tmp_path, "hash123")


def test_write_manifest_without_extra_unchanged(tmp_path):
    manifest.write_manifest(tmp_path, "h", ["x"])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert "seed" not in m and m["outputs"] == ["x"]
