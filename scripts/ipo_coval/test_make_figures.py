"""The paper's figures render from the committed per-epoch data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figures  # noqa: E402


def test_figures_render(tmp_path):
    make_figures.main(tmp_path)
    for name in ("fig_training.png", "fig_cyclic.png"):
        assert (tmp_path / name).stat().st_size > 50_000
