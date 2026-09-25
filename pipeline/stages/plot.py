"""Stage: assemble the Figure 2 reproduction (2x3 grid).

Columns: majority | IIA (2-alt dotted, 3-alt solid) | cyclic
Rows:    Maximal Lottery over training | RLHF over training (the paper's order)

As in the paper, there is no final-distribution row: the verdicts come from
scripts/verdicts.py, and a bar of the last checkpoint misreads the Maximal
Lottery arm, whose output is the mixture of iterates.

Inputs: metrics.jsonl written during training (per-epoch policy distribution).
The stage renders whatever cells exist, so the
figure can be regenerated from fake or partial data.

Colour encoding is semantic (the series ARE the colours red/blue/green).
Palette is CVD-validated; per-colour markers are the secondary encoding
required for the red/green pair.
"""

from pathlib import Path

from pipeline import manifest
from pipeline.costs import stage_timer
from pipeline.storage import Storage

STAGE = "plot"

PALETTE = {"red": "#b2182b", "blue": "#1f77b4", "green": "#2e9e44"}
MARKERS = {"red": "o", "blue": "s", "green": "^"}
INK = "#333333"
GRID = "#dddddd"

COLUMNS = [
    ("majority", "Majority & Condorcet\n2x(R>G>B), 3x(B>R>G)"),
    ("iia", "IIA\n2x(R>B), 3x(B>R)  [+G]"),
    ("cyclic", "Cyclic\n1x(R>G>B), 1x(G>B>R), 1x(B>R>G)"),
]
METHOD_LABELS = {"rlhf": "RLHF (PPO)", "ml": "Maximal Lottery (SPO)"}


def _read_metrics(storage: Storage, config: str, method: str) -> list[dict]:
    path = storage.stage_dir(f"train_{method}", config) / "metrics.jsonl"
    rows = Storage.read_jsonl(path) if path.exists() else []
    # keep per-epoch distribution rows; skip event rows (e.g. checkpoint_step)
    return [r for r in rows if "dist" in r]


def _style_axis(ax):
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)


def _plot_training_curves(ax, metrics: list[dict], linestyle: str = "-"):
    if not metrics:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#999999", fontsize=9)
        return
    steps = [m["step"] for m in metrics]
    colours = sorted(metrics[0]["dist"].keys())
    for colour in colours:
        ax.plot(
            steps, [m["dist"][colour] for m in metrics],
            color=PALETTE[colour], marker=MARKERS[colour], markersize=3.5,
            linewidth=2, linestyle=linestyle, label=colour,
            markevery=max(1, len(steps) // 8),
        )


def render(storage: Storage, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    fig.patch.set_facecolor("white")

    for col, (config_key, title) in enumerate(COLUMNS):
        axes[0][col].set_title(title, fontsize=9.5, color=INK)
        for row, method in enumerate(("ml", "rlhf")):
            ax = axes[row][col]
            _style_axis(ax)
            if config_key == "iia":
                _plot_training_curves(ax, _read_metrics(storage, "iia_3alt", method), "-")
                _plot_training_curves(ax, _read_metrics(storage, "iia_2alt", method), ":")
            else:
                _plot_training_curves(ax, _read_metrics(storage, config_key, method))
            if col == 0:
                ax.set_ylabel(f"{METHOD_LABELS[method]}\nP(colour)", fontsize=9, color=INK)

    for col in range(3):
        axes[1][col].set_xlabel("training epoch", fontsize=8, color=INK)

    legend_elems = [
        Line2D([0], [0], color=PALETTE[c], marker=MARKERS[c], linewidth=2,
               markersize=5, label=c) for c in ("red", "blue", "green")
    ] + [
        Line2D([0], [0], color=INK, linestyle=":", linewidth=2, label="2 alternatives (IIA)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Reproduction of Figure 2 — Jackpot! Alignment as a Maximal Lottery "
                 "(arXiv 2501.19266)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(storage: Storage, profile: dict, force: bool = False) -> dict:
    # hash over the training manifests, so new training results invalidate
    # the figure
    upstream = []
    for config in profile["configs"]:
        for method in profile["methods"]:
            m = manifest.load(storage.stage_dir(f"train_{method}", config))
            upstream.append(m["params_hash"] if m else "missing")
    phash = manifest.params_hash({"upstream": upstream})
    out = storage.stage_dir(STAGE)
    if not force and manifest.is_complete(out, phash):
        return {"skipped": True, "dir": str(out)}

    with stage_timer(out, STAGE):
        render(storage, out / "figure2.png")
    manifest.write_manifest(out, phash, ["figure2.png"])
    return {"skipped": False, "figure": str(out / "figure2.png")}
