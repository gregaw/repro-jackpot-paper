# Reproduction of "Jackpot! Alignment as a Maximal Lottery"

This repository contains an independent reproduction of the synthetic colour-preference
experiments of Maura-Rivero et al., [*Jackpot! Alignment as a Maximal Lottery*](https://arxiv.org/abs/2501.19266)
(Section 6.3 and Figure 2). It trains a small language model to name its favourite colour
under two objectives, RLHF (a reward model, then PPO) and the paper's Maximal Lottery
objective (SPO), over four synthetic voter populations, and checks the outcome against the
paper's predictions. The paper's claim reproduces on its own model, Gemma-2-2B, in all eight
cells: RLHF behaves like the Borda count, the Maximal Lottery picks the Condorcet winner and
splits a cycle evenly.

Paper: [arXiv:2501.19266](https://arxiv.org/abs/2501.19266) by Roberto-Rafael Maura-Rivero,
Marc Lanctot, Francesco Visin and Kate Larson. This is not a fork of the authors' code. The
write-ups that go with the code:

- [`results/reproduction_report.md`](results/reproduction_report.md): the reproduction, every
  setting we changed from the paper's Appendix A.8 and why, costs, and limits. This is the
  main result.
- [`results/experiments_ipo_coval.md`](results/experiments_ipo_coval.md): an addition beyond
  the paper. Online IPO, trained on the policy's own samples, reaches the maximal lottery from
  its final checkpoint, while the same loss trained offline reproduces Borda. It also finds
  that the CoVal dataset of real rankings has too few voters per prompt for its
  Borda/Condorcet disagreements to survive resampling.
- [`results/deep-dive-overview.md`](results/deep-dive-overview.md): a walkthrough of how SPO,
  RLHF/PPO and Online IPO work here, with links to the lines of code that implement each step.

## How it works

| Stage | What it produces |
| --- | --- |
| `data_gen` | 2048 preference triplets per population, sampled from the voter profile |
| `train` (`rlhf`) | a reward model fine-tuned on the triplets, then a PPO policy trained against it |
| `train` (`ml`) | a policy trained with SPO against the empirical preference function, plus the mixture of iterates |
| `train` (`ipo`, `ipo_offline`) | Online IPO, and the same loss trained offline on the triplets as a control |
| `eval` | 1000 samples from each trained policy, parsed into a colour distribution |
| `plot` | Figure 2: training curves for the three experiments |

Every stage writes `manifest.json` last, on success, keyed by a hash of its parameters. A
stage that already has a matching manifest is skipped, so any subset can be re-run, and
`--force` re-runs a stage regardless. `pipeline/stages/` is plain Python with no Modal
import, so the CPU test runs the same training and evaluation code as the GPU runs.
`pipeline/modal_app.py` wraps the stages as Modal GPU functions.

**Where Claude comes in.** No stage calls Claude or any other hosted LLM. The voters are
synthetic, the preferences are computed from their rankings, and the models trained are
open weights. A run needs no API key beyond Modal's and a Hugging Face read token. The code,
the experiment runs and the write-ups were made with [Claude Code](https://claude.com/claude-code),
Anthropic's coding agent, under the author's direction: the author set the scope, the
architecture, the order of work and the budget for each step, decided which fixes to pursue,
and checked the results. Comments that cite a numbered experiment ("experiment 0086") refer
to that development history and are kept as provenance.

## Setup

1. Python 3.11 and [uv](https://docs.astral.sh/uv/). The pinned `torch` has no wheel for
   Python 3.13 or newer, and the Modal image runs 3.11.

   ```bash
   uv venv --python 3.11 .venv
   uv pip install --python .venv/bin/python -r configs/requirements-gpu.txt modal pytest
   ```

2. A [Modal](https://modal.com) account. The L40S that Gemma-2-2B trains on needs a payment
   method on the account; the free tier's L4 is enough for the Qwen profile and for evaluation.

   ```bash
   .venv/bin/modal token new     # or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET
   ```

3. A Hugging Face read token, with the Gemma licence accepted at
   [huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b).
   Qwen2.5-0.5B is not gated, but the wrapper attaches the secret to every GPU function.

   ```bash
   .venv/bin/modal secret create huggingface-secret HF_TOKEN=hf_...   # once per Modal workspace
   ```

No credential is read from a file in this repository. Modal keeps its token in
`~/.modal.toml`; the Hugging Face token lives only in the Modal secret. Two optional
environment variables change what a dispatch runs: `JACKPOT_PROFILE` (a profile other than
Gemma, below) and `JACKPOT_TRAIN_OVERRIDES` (JSON merged over the profile's training settings).

## Quick check

```bash
.venv/bin/python -m pytest
```

This runs the unit tests and `tests/test_e2e_local.py`, which drives every stage on CPU with
a tiny random model in about 30 seconds. It needs no GPU, no network and no credentials. Run
it before spending anything on GPUs.

## Reproducing the results

All commands run from the repository root.

### Figure 2 and the verdict table (report sections 2 and 3)

```bash
.venv/bin/modal run --detach pipeline/modal_app.py --run-id my-run
```

`--detach` matters: the orchestrator runs on Modal, and without it the run dies when the
local client exits. The command returns within seconds and prints the app id. The run then
downloads the model into the Modal Volume once, generates data, trains the eight cells (at
most four at a time), evaluates them and renders the figure. It takes two to three hours and
costs **about \$11 on Gemma-2-2B** (training on an L40S, evaluation on an L4; report
Appendix F). Artifacts land on the Volume `jackpot-artifacts` under `runs/my-run/`.

To watch it:

```bash
.venv/bin/modal run pipeline/modal_app.py --stage status --run-id my-run    # 21-cell completion table
.venv/bin/modal app logs <app-id>                                           # per-step training lines
.venv/bin/python scripts/watch.py <app-id> --run-id my-run --interval 300   # polls both, flags degeneration
```

A failed cell does not stop the run; the orchestrator prints `CELL FAILED` and the status
table shows the gap. Dispatching the same command again retries only the missing cells.

Then fetch the results and judge them:

```bash
.venv/bin/modal run scripts/fetch_artifacts.py --run-id my-run --dest results/runs/my-run
.venv/bin/python scripts/verdicts.py results/runs/my-run/eval
.venv/bin/python scripts/report_costs.py results/runs/my-run
```

`verdicts.py` prints one row per cell with the sampled distribution and a PASS/FAIL against
the paper's prediction, and exits 0 only if all eight pass. A colour "wins" at a share of at
least 0.80, and "uniform" means every share is within 0.10 of 1/3. Maximal Lottery cells are
judged on the mixture of iterates (Algorithm 1's output policy), RLHF cells on the final
checkpoint. `report_costs.py` prices the per-stage timings at the rates in
`configs/rates.yaml`; the Modal usage page is the authoritative figure. Figure 2 is at
`results/runs/my-run/plot/figure2.png`; the figures of the reported runs are checked in as
[`results/figure2_gemma-2-2b.png`](results/figure2_gemma-2-2b.png) and
[`results/figure2_qwen2.5-0.5b.png`](results/figure2_qwen2.5-0.5b.png).

### The cheaper Qwen2.5-0.5B profile (report Appendix D)

```bash
JACKPOT_PROFILE=qwen .venv/bin/modal run --detach pipeline/modal_app.py --run-id my-qwen-run
```

The same grid on a model a quarter of the size, on L4s, for **about \$6**. Repeat the
`JACKPOT_PROFILE=qwen` prefix on every train or eval dispatch for that run (status and fetch
do not need it), or the next dispatch retrains the cells as Gemma.

### Online IPO and CoVal (the IPO write-up)

Appendix B of [`results/experiments_ipo_coval.md`](results/experiments_ipo_coval.md) gives the
commands for each of its tables and figures. The GPU runs use the same pipeline with
`--method ipo` or `--method ipo_offline` on the Qwen profile (about \$11 for everything in
that write-up); the tabular solutions, the CoVal analysis and the figures run on CPU from
[`scripts/ipo_coval/`](scripts/ipo_coval/).

### Useful variants

```bash
# one cell, to check timings and memory before the rest
.venv/bin/modal run --detach pipeline/modal_app.py --stage train --config cyclic --method ml --run-id my-run

# re-evaluate without retraining, or retrain one cell under edited hyperparameters
.venv/bin/modal run --detach pipeline/modal_app.py --stage eval --force --run-id my-run
.venv/bin/modal run --detach pipeline/modal_app.py --stage train --config majority --method ml --force --run-id my-run

# another data and training seed (artifacts land under runs/my-run-s8/)
.venv/bin/modal run --detach pipeline/modal_app.py --seed 8 --run-id my-run

# sample a trained policy interactively (it answers one fixed prompt with a colour; not a chat model)
.venv/bin/modal run pipeline/modal_app.py --stage chat --config cyclic --method ml --run-id my-run
```

Hyperparameters live in `configs/experiments.yaml`: Appendix A.8 values from
`paper_hparams.yaml`, overridden per method under `train:` and per cell under
`train.cells:`. Changing one changes the cell's parameter hash, so the next dispatch
retrains that cell and leaves the others alone.

The plot stage only reads `metrics.jsonl` and `distributions.json`, so Figure 2 can be
re-rendered locally from a fetched run, for example after combining cells from several runs:

```bash
.venv/bin/python -c "
from pipeline.config import load_profile; from pipeline.stages import plot; from pipeline.storage import Storage
plot.run(Storage('results', 'my-run'), load_profile('paper'), force=True)"
```

## Main result

Gemma-2-2B, 1000 samples per cell, shares blue / green / red
(`scripts/verdicts.py`; report section 2):

| Population | RLHF | Maximal Lottery (mixture of iterates) |
| --- | --- | --- |
| majority: 2x(R>G>B), 3x(B>R>G) | red 1.00 (paper: red, the Borda winner) | blue 0.96 (paper: blue, the Condorcet winner) |
| IIA, two options: 2x(R>B), 3x(B>R) | blue 0.94 (paper: blue) | blue 0.94 (paper: blue) |
| IIA, green added | red 0.83 (paper: flips to red) | blue 0.94 (paper: stays blue) |
| cyclic: 1x(R>G>B), 1x(G>B>R), 1x(B>R>G) | blue 0.99 (paper: one arbitrary colour) | 0.38 / 0.27 / 0.34 (paper: about 1/3 each) |

All eight cells pass. On Qwen2.5-0.5B seven of eight pass: on "IIA, green added" RLHF goes
the paper's way (red 0.52 against blue 0.48) but stays under the 0.80 bar.

## Reference

| Config | What it is |
| --- | --- |
| `configs/experiments.yaml` | the Gemma-2-2B profile, with the hyperparameters behind the report |
| `configs/qwen.yaml` | the Qwen2.5-0.5B profile (`JACKPOT_PROFILE=qwen`), also used by the IPO write-up |
| `configs/smoke.yaml` | the tiny-model profile of the CPU test |
| `configs/paper_hparams.yaml` | Appendix A.8 as extracted from the paper; `a8_raw.txt` is the raw text |
| `configs/rates.yaml` | Modal list prices used by `scripts/report_costs.py` |
| `configs/requirements-gpu.txt` | exact pins of the GPU image (trl 0.10.1, the version the paper names) |

| Script | What it does |
| --- | --- |
| `scripts/verdicts.py` | judges evaluated cells against the paper's predictions |
| `scripts/fetch_artifacts.py` | downloads a run's small files from the Modal Volume |
| `scripts/report_costs.py` | prices a fetched run's per-stage timings |
| `scripts/watch.py` | watches a detached run and flags degenerate generation |
| `scripts/export_model.py` | downloads a trained adapter, optionally merged into the base weights (uses `modal volume get`) |
| `scripts/spo_simplex_sim.py` | Algorithm 1 in the probability simplex with no language model, in under a second; shows why the mixture of iterates, not the last iterate, converges on the cycle |
| `scripts/extract_hparams.py` | regenerates `configs/paper_hparams.yaml` from the paper's PDF (`pip install pypdf`) |
| `scripts/ipo_coval/` | the CPU parts of the IPO write-up: tabular IPO, the CoVal analysis, its figures |

## Layout

```
pipeline/            stage wiring, config, storage and manifests, and the Modal wrapper
  stages/            data_gen, train_spo, train_rlhf, train_ipo, evaluate, plot (no Modal imports)
configs/             model profiles, the paper's hyperparameters, pins and prices
scripts/             verdicts, fetching, watching, costs, the simplex simulation
  ipo_coval/         CPU scripts and data for the IPO write-up
tests/               unit tests and the CPU end-to-end run of the whole pipeline
results/             the report, the two additional write-ups, and their figures
```

## Known limits

- Every cell is a single seed. `majority` and `iia_3alt` are the same population on two
  draws of the data, and differ by about 0.14 on the Maximal Lottery arm and 0.17 on RLHF.
- Five per-cell KL anchors (three on Gemma, two on Qwen) were chosen after seeing results;
  the paper uses one setting per arm (report Appendix A).
- We generate 8 tokens per answer. The paper does not say how many it generated; if it was
  one, several of our deviations from A.8 may be artefacts of this choice (report section 4).
- The colour parser matches substrings, so `" hundred"` parses as red. Re-parsing the
  reported samples with word boundaries changes none of them.
- The reward model trains at learning rate 1e-4 where A.8's "trl defaults" would mean 5e-5.
- The Gemma cyclic RLHF cell's colour came out of a breakdown and recovery, not where
  training converged (report Appendix C).

## Citation

Please cite the paper:

```bibtex
@article{maurarivero2025jackpot,
  author  = {Maura-Rivero, Roberto-Rafael and Lanctot, Marc and Visin, Francesco and Larson, Kate},
  title   = {Jackpot! Alignment as a Maximal Lottery},
  journal = {arXiv preprint arXiv:2501.19266},
  year    = {2025}
}
```

and, if you use this code, the reproduction:

```bibtex
@misc{gawron2026jackpotrepro,
  author = {Gawron, Grzegorz},
  title  = {Reproduction of "Jackpot! Alignment as a Maximal Lottery"},
  year   = {2026},
  url    = {https://github.com/gregaw/repro-jackpot-paper}
}
```

## Licence and contact

MIT, see [`LICENSE`](LICENSE). Questions and bug reports are welcome as GitHub issues.
