"""Extract Appendix A.8 (experimental details and hyperparameters) from the paper PDF.

Writes:
  configs/a8_raw.txt        - raw extracted appendix text, for auditing
  configs/paper_hparams.yaml - structured hyperparameters transcribed from A.8

The structured values are transcribed (not parsed heuristically) from the raw
text, which is saved alongside so any transcription can be checked. The PDF is not
part of this repository: download it from https://arxiv.org/pdf/2501.19266 and
pass its path. The transcription below is validated against the extracted text
with simple assertions.

Usage: python scripts/extract_hparams.py path/to/2501.19266.pdf
"""

import sys
from pathlib import Path

import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "configs" / "a8_raw.txt"
YAML_OUT = ROOT / "configs" / "paper_hparams.yaml"

PROMPT_3OPT = (
    "Q: What is your favorite color from the options red, blue and green? "
    "answer in the format 'My favourite color is the color red.' "
    "'My favourite color is the color blue.' or "
    "'My favourite color is the color green.' and say nothing else after that. \n"
    "A: My favourite color is the color"
)

PROMPT_2OPT = (
    "Q: What is your favorite color from the options red and blue? "
    "answer in the format 'My favourite color is the color red.' or "
    "'My favourite color is the color blue.' and say nothing else after that. \n"
    "A: My favourite color is the color"
)

HPARAMS = {
    "source": "arXiv 2501.19266, Appendix A.8",
    "trl_version_used_by_paper": "0.10.1",
    "prompts": {
        "three_options": PROMPT_3OPT,
        "two_options": PROMPT_2OPT,
    },
    "lora": {"r": 8, "alpha": 32, "dropout": 0.1},
    "dataset": {"n_datapoints": 2048},
    "spo": {
        "rl_step": "ppo",
        "epochs": 30,
        "batch_size": 128,
        "mini_batch_size": 32,
        "learning_rate": 1.0e-4,
        "vf_coef": 0.0,
        "init_kl_coef": 0.0,
        "gamma": 0.0,
        "entropy_coef": "increased to ensure exploration (value not given)",
        "uniform_colour_fraction": 0.1,
        "uniform_colour_tokens": [" red", " green", " blue"],
        "output_policy": "uniform mixture of pi_1..T (Algorithm 1)",
    },
    "preference_function": {
        "definition": "fraction of voters preferring a over b in the dataset",
        "tie_score": 0.5,
        "missing_one_alternative_score": 1.0,
        "missing_both_score": 0.5,
    },
    "reward_model": {
        "epochs": 3,
        "center_rewards_coefficient": 0.01,
        "other": "trl 0.10.1 defaults",
    },
    "rlhf_ppo": {
        "epochs": 4,
        "batch_size": 16,
        "learning_rate": 5.0e-4,
        "vf_coef": 0.01,
        "init_kl_coef": 0.0,
    },
}

# fragments that must appear in the extracted text for the transcription to be valid
VALIDATION_FRAGMENTS = [
    "Rank (r): 8",
    "Alpha: 32",
    "Dropout: 0.1",
    "Epochs: 30",
    "Batch size: 128",
    "Mini-batch size: 32",
    "trl (version 0.10.1)",
    "center_rewards_coefficient",
    "Batch size: 16",
    "Epochs: 4",
]


def extract_a8(full_text: str) -> str:
    start = full_text.rindex("A.8. Experimental details and hyperparameters")
    end = full_text.index("A.9.", start)
    return full_text[start:end]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    reader = PdfReader(sys.argv[1])
    full = "\n".join(page.extract_text() for page in reader.pages)
    a8 = extract_a8(full)

    missing = [f for f in VALIDATION_FRAGMENTS if f not in a8]
    if missing:
        raise SystemExit(f"A.8 text does not contain expected fragments: {missing}")

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text(a8)
    YAML_OUT.write_text(yaml.safe_dump(HPARAMS, sort_keys=False, allow_unicode=True))
    print(f"wrote {RAW_OUT.relative_to(ROOT)} ({len(a8)} chars)")
    print(f"wrote {YAML_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
