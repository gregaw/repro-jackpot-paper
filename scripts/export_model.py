"""Pull a trained LoRA adapter from the Modal Volume to local disk, optionally
merging it into the base weights for a standalone model.

  .venv/bin/python scripts/export_model.py --config cyclic --method ml
  .venv/bin/python scripts/export_model.py --config cyclic --method ml --merge

Uses `modal volume get`, which needs access to Modal's storage domain.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import load_profile  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["rlhf", "ml"])
    ap.add_argument("--run-id", default="default")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="merge the adapter into the base weights (downloads the base model)")
    args = ap.parse_args()
    base_name = load_profile(os.environ.get("JACKPOT_PROFILE", "paper"))["model"]["name"]

    out = args.out or ROOT / "exports" / f"{args.config}-{args.method}"
    out.mkdir(parents=True, exist_ok=True)
    remote = f"runs/{args.run_id}/train_{args.method}/{args.config}/checkpoint"
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "modal"), "volume", "get", "--force",
         "jackpot-artifacts", remote, str(out)],
        check=True,
    )
    print(f"adapter pulled to {out}")

    if args.merge:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, str(out / "checkpoint")).merge_and_unload()
        merged_dir = out / "merged"
        merged.save_pretrained(str(merged_dir))
        AutoTokenizer.from_pretrained(base_name).save_pretrained(str(merged_dir))
        print(f"merged standalone model at {merged_dir}")


if __name__ == "__main__":
    main()
