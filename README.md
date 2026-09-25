# Reproduction of "Jackpot! Alignment as a Maximal Lottery"

An independent reproduction of the synthetic colour-preference experiments of
Maura-Rivero et al., *Jackpot! Alignment as a Maximal Lottery*
([arXiv:2501.19266](https://arxiv.org/abs/2501.19266), Section 6.3 and Figure 2): RLHF
(reward model + PPO) against a Maximal Lottery objective (SPO), over four voter
populations.

The full write-up is [`results/reproduction_report.md`](results/reproduction_report.md).

## What this repository contributes

**The paper reproduces.** On the paper's own model, Gemma-2-2B, all eight cells of Figure 2
come out as Section 6.3 describes: RLHF collapses to the Borda winner, flips when an
irrelevant alternative is added (blue, the Condorcet winner, with two options; red, the
Borda winner, once green is added), and picks one arbitrary colour on a cycle; the Maximal
Lottery arm returns the Condorcet winner, ignores the irrelevant alternative, and goes
roughly uniform on the cycle. This is an independent implementation, not a fork of the
authors' code.

**It is ready to run, and validated.** `pytest tests/` drives all 21 cells of the pipeline
end to end on CPU in about 30 seconds — no GPU, no network, no credentials — before you
spend anything. A full Gemma grid is then one `modal run` command, about three hours and
about \$11 of GPU time, and `scripts/verdicts.py` scores the outcome against the paper's
prediction under rules fixed in advance, exiting non-zero unless all eight cells pass.

**A small model shows the same behaviour, more cheaply.** We added a Qwen2.5-0.5B profile —
a quarter of the size, and it runs on the free tier's L4 for about \$6. Seven of the eight
cells reproduce. The behaviour is real at this scale but less clean: the IIA flip goes in
the paper's direction (red ahead at both anchors) without collapsing past our 0.80 bar,
because Qwen's reward model has a thinner red-over-blue margin on that draw. Its value was
in cheap iteration: once a fix was in, a Qwen canary cost about \$1 to \$2, and only
settings that passed on Qwen were run on Gemma.

**The changes we needed are recorded.** We were not able to reproduce the result with the
hyperparameters of Appendix A.8 as we read them. The report lists every setting we changed,
with the reason for each, and the four defects that cost the most are written up with how they showed, how they
were found, and what they cost — the SPO discount factor read as 0 (no preference gradient
ever reached the colour token), generation degenerating after convergence, an RLHF reward
computed on text the reward model had never seen, and a KL anchor too tight on one cell.
A clean run costs about \$11. Getting to the point where the run was clean cost about \$83,
and the report accounts for the difference problem by problem.

**The Modal infrastructure is reusable.** The pipeline is stage-based with content-hashed
manifests, so any subset re-runs and the rest is skipped; `pipeline/stages/` is plain Python
with no Modal import, so the CPU test runs the same training and evaluation code as the GPU
runs, not a mock of it. If you
are running experiments of this shape, this is the part worth lifting. What we liked about
Modal: the free tier goes further than you would expect — many GPUs in parallel, a real
choice of hardware (we screen on L4 and train on L40S from the same code), and storage
integrated well enough that artifacts just persist across runs.

Since the environment has sharp edges, the traps we hit are in the run instructions below —
`--detach` above all.

## How this was built

The code, the experiment runs and the report were written by AI agents using
[Claude Code](https://claude.com/claude-code). The author guided the work rather than
typing it. Specifically:

- **Implementation.** The author set the scope, the general architecture, the order of work
  and the budget for each step, and decided which fixes to pursue when a run failed.
- **Verification.** No GPU money was spent until the CPU end-to-end test passed. One cell
  was run and watched before the rest. The pass/fail rules were fixed in
  `scripts/verdicts.py` before the final runs. The author verified the results and
  iterated on the final outcome.
- **The write-up.** The author edited both this README and the report for the reader: less
  lab notebook, more plain explanation, with the details moved into appendices. They are
  more readable for it, if still dense in places.

## What the pipeline does

| Stage | What it produces |
| --- | --- |
| `data_gen` | 2048 preference triplets per population, sampled from the voter profile |
| `train` (`rlhf`) | a reward model fine-tuned on the triplets, then a PPO policy against it |
| `train` (`ml`) | a policy trained with SPO against the empirical preference function, plus the mixture of iterates |
| `eval` | 1000 samples from each trained policy, parsed into a colour distribution |
| `plot` | Figure 2: training curves for the three experiments |

Every stage writes `manifest.json` last, on success, keyed by a hash of its parameters. A
stage that already has a matching manifest is skipped, so any subset can be re-run and
`--force` re-runs a stage regardless.

Layout:

- `pipeline/` — the stages (`stages/`, plain Python with no Modal dependency), config and
  storage helpers, and the Modal wrapper `modal_app.py`
- `configs/` — `experiments.yaml` (Gemma-2-2B, the hyperparameters that produced the report),
  `qwen.yaml` (a cheaper screening model), `smoke.yaml` (the CPU test),
  `paper_hparams.yaml` (Appendix A.8 as extracted), `requirements-gpu.txt`, `rates.yaml`
- `scripts/` — verdicts, artifact fetching, run watching, cost pricing, model export, and a
  CPU simulation of Algorithm 1
- `tests/` — unit tests and a CPU end-to-end run of the whole pipeline
- `results/` — the report and its figures

## Prerequisites

- Python 3.11 and [`uv`](https://github.com/astral-sh/uv) (or any way to make a venv). The
  pinned `torch` has no wheel for Python 3.13 or newer, and the Modal image runs 3.11.
- A [Modal](https://modal.com) account. The L40S that Gemma-2-2B trains on needs a payment
  method on the account; the free tier's L4 is enough for the Qwen profile and for evaluation.
- A Hugging Face account with the Gemma licence accepted at
  [huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) and a read token.
  Qwen2.5-0.5B is not gated.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r configs/requirements-gpu.txt modal pytest

.venv/bin/modal token new                                        # or export MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
.venv/bin/modal secret create huggingface-secret HF_TOKEN=hf_...  # once per Modal workspace
```

No credential is read from a file in this repository. Modal stores its token in
`~/.modal.toml`; the Hugging Face token lives only in the Modal secret.

## Step 1: run the local test

```bash
.venv/bin/python -m pytest tests/
```

This runs the unit tests and `tests/test_e2e_local.py`, which drives every stage on CPU with
a tiny random model in about 30 seconds. It needs no GPU, no network and no credentials. Do
not spend GPU time until it passes.

## Step 2: run the pipeline

```bash
.venv/bin/modal run --detach pipeline/modal_app.py --run-id my-run
```

`--detach` matters: the orchestrator runs on Modal, and without it the run dies when your
terminal closes. The command returns within seconds and prints the app id. The run then
downloads the model into the Modal Volume once, generates data, trains the eight cells (at
most four at a time), evaluates them and renders the figure. Expect two to three hours of
wall-clock time. Artifacts persist on the Volume `jackpot-artifacts` under `runs/my-run/`.

To watch it:

```bash
.venv/bin/modal run pipeline/modal_app.py --stage status --run-id my-run    # 21-cell completion table
.venv/bin/modal app logs <app-id>                                           # per-step training lines
.venv/bin/python scripts/watch.py <app-id> --run-id my-run --interval 300   # polls both, flags degeneration
```

A failed cell does not stop the run; the orchestrator prints `CELL FAILED` and the status
table shows the gap. Re-dispatching the same command retries only the missing cells.

Useful variants:

```bash
# one cell, to check timings and memory before the rest
.venv/bin/modal run --detach pipeline/modal_app.py --stage train --config cyclic --method ml --run-id my-run

# re-evaluate without retraining, or retrain one cell under edited hyperparameters
.venv/bin/modal run --detach pipeline/modal_app.py --stage eval --force --run-id my-run
.venv/bin/modal run --detach pipeline/modal_app.py --stage train --config majority --method ml --force --run-id my-run

# another data/training seed (artifacts land under runs/my-run-s8/)
.venv/bin/modal run --detach pipeline/modal_app.py --seed 8 --run-id my-run

# the cheaper screening model on an L4; repeat the prefix on every train/eval dispatch for
# that run (status and fetch do not need it), or the next dispatch retrains it as Gemma
JACKPOT_PROFILE=qwen .venv/bin/modal run --detach pipeline/modal_app.py --run-id my-qwen-run
```

Hyperparameters live in `configs/experiments.yaml`: Appendix A.8 values from
`paper_hparams.yaml`, overridden per method under `train:` and per cell under
`train.cells:`. Changing one changes the cell's parameter hash, so the next dispatch
retrains that cell and leaves the others alone.

## Step 3: fetch the results and judge them

```bash
.venv/bin/modal run scripts/fetch_artifacts.py --run-id my-run --dest results/runs/my-run
.venv/bin/python scripts/verdicts.py results/runs/my-run/eval
.venv/bin/python scripts/report_costs.py results/runs/my-run
```

`fetch_artifacts.py` pulls the figure, every `distributions.json`, `metrics.jsonl`,
`mixture.json`, `rm_scores.json`, `cost.json` and `manifest.json`, keeping the Volume's
layout. `verdicts.py` prints one row per cell with the sampled distribution and a
PASS/FAIL against the paper's prediction, and exits 0 only if all eight pass. Maximal
Lottery cells are judged on the mixture of iterates (Algorithm 1's output policy); RLHF
cells on the final checkpoint. `report_costs.py` prices the per-stage timings at the rates
in `configs/rates.yaml`; the Modal usage page is the authoritative figure.

Figure 2 is at `results/runs/my-run/plot/figure2.png`. The plot stage only reads
`metrics.jsonl` and `distributions.json`, so it can also be re-rendered locally from the
fetched tree, for example after combining cells from several runs:

```bash
.venv/bin/python -c "
from pipeline.config import load_profile; from pipeline.stages import plot; from pipeline.storage import Storage
plot.run(Storage('results', 'my-run'), load_profile('paper'), force=True)"
```

## Other tools

- `--stage chat --config cyclic --method ml --run-id my-run` samples a trained policy
  interactively. These policies answer one fixed prompt with a colour; they are not chat models.
- `scripts/export_model.py --config cyclic --method ml --run-id my-run [--merge]` downloads
  an adapter, optionally merged into the base weights. It uses `modal volume get`.
- `scripts/spo_simplex_sim.py` runs Algorithm 1 in the probability simplex with no language
  model, in under a second. It shows why the mixture of iterates, not the last iterate, is
  what converges on the cyclic population. It replaces the language model with a
  three-way choice between the colours and keeps everything else in Algorithm 1: the
  win-rate reward, the 10% random colours, the policy-gradient step and the mixture of
  iterates. It says how the algorithm behaves, not whether a language model trained with
  PPO behaves that way, which is the paper's claim; it was used to explain results and to
  screen settings before paying for GPU runs.
- `scripts/extract_hparams.py <pdf>` regenerates `configs/paper_hparams.yaml` and
  `configs/a8_raw.txt` from the paper's PDF (`pip install pypdf`).

## Expected results (paper Section 6.3)

| Population | RLHF | Maximal Lottery |
| --- | --- | --- |
| majority: 2×(R>G>B), 3×(B>R>G) | collapses to red (Borda winner) | converges to blue (Condorcet winner) |
| IIA: 2×(R>B), 3×(B>R), then add an irrelevant G | blue, then flips to red | blue in both |
| cyclic: 1×(R>G>B), 1×(G>B>R), 1×(B>R>G) | collapses to one arbitrary colour | about 1/3 each |
