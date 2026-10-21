"""Figures 1 and 2 of results/experiments_ipo_coval.md, from the per-epoch colour
shares in data/epoch_shares.json.

    python make_figures.py        # writes fig_training.png and fig_cyclic.png next to itself

Colours and markers match the Figure 2 reproduction (pipeline/stages/plot.py):
the hue is the answer's own colour, and the marker shape repeats it for readers who
cannot tell the hues apart.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
PALETTE = {"red": "#b2182b", "blue": "#1f77b4", "green": "#2e9e44"}
MARKERS = {"red": "o", "blue": "s", "green": "^"}
INK, GRID = "#333333", "#dddddd"


def curves(ax, epochs: list[dict], style: str = "-") -> None:
    x = range(1, len(epochs) + 1)
    for colour in ("red", "blue", "green"):
        if colour in epochs[0]:
            ax.plot(x, [e[colour] for e in epochs], style, color=PALETTE[colour],
                    marker=MARKERS[colour], markersize=3.5, linewidth=1.6)


def frame(ax, title: str | None = None) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0.5, 30.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if title:
        ax.set_title(title, fontsize=9, color=INK)


def legend(fig, with_iia: bool) -> None:
    handles = [Line2D([0], [0], color=PALETTE[c], marker=MARKERS[c], linewidth=1.6,
                      markersize=4, label=c) for c in ("red", "blue", "green")]
    if with_iia:
        handles.append(Line2D([0], [0], color=INK, linestyle=":", linewidth=1.6,
                              label="two options (IIA)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def training_figure(data: dict, out: Path) -> None:
    rows = [("online_final", "Online IPO\n(tau 0.05, final setting)"),
            ("offline_tau0.1", "Offline IPO\n(tau 0.1)")]
    cols = [("majority", "majority"), ("iia", "IIA: 2 options (dotted), 3 (solid)"),
            ("cyclic", "cyclic")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey=True)
    for r, (key, label) in enumerate(rows):
        cells = data[key]["cells"]
        for c, (col, title) in enumerate(cols):
            ax = axes[r][c]
            if col == "iia":
                curves(ax, cells["iia_3alt"], "-")
                curves(ax, cells["iia_2alt"], ":")
            else:
                curves(ax, cells[col])
            frame(ax, title if r == 0 else None)
            if c == 0:
                ax.set_ylabel(f"{label}\nP(colour)", fontsize=8.5, color=INK)
            if r == 1:
                ax.set_xlabel("epoch", fontsize=8.5, color=INK)
    legend(fig, with_iia=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)


def cyclic_figure(data: dict, out: Path) -> None:
    panels = [("cyclic_tau0.02_beta1_0.9", "tau 0.02, Adam beta1 0.9"),
              ("cyclic_tau0.02_lr2e-5", "tau 0.02, beta1 0.9, lr 2e-5"),
              ("cyclic_tau0.02_beta1_0", "tau 0.02, beta1 0"),
              ("online_tau0.1", "tau 0.1, beta1 0.9"),
              ("online_tau0.05", "tau 0.05, beta1 0.9"),
              ("cyclic_tau0.05_ipomd", "tau 0.05, beta1 0.9, IPO-MD 0.125"),
              ("cyclic_tau0.05_beta1_0", "tau 0.05, beta1 0"),
              ("online_final", "tau 0.05, beta1 0, lr decay (final)")]
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.flat, panels):
        curves(ax, data[key]["cells"]["cyclic"])
        ax.axhline(1 / 3, color=INK, linewidth=0.8, linestyle="--", alpha=0.5)
        frame(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=8.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(colour)", fontsize=8.5, color=INK)
    legend(fig, with_iia=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)


def main(out_dir: Path = HERE) -> None:
    data = json.loads((HERE / "data" / "epoch_shares.json").read_text())
    training_figure(data, out_dir / "fig_training.png")
    cyclic_figure(data, out_dir / "fig_cyclic.png")


if __name__ == "__main__":
    main()
