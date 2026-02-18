"""Evaluate a candidate PCFG on samples drawn from a checkpoint LM.

This is a *proxy* evaluator for local iteration (not the official hidden TVD).
It reports average log-probability assigned by your PCFG to sampled strings.
"""

import argparse
import subprocess
import os
import sys
import tempfile
from pathlib import Path

from pcfg_utils import PCFG


def sample_strings(script_dir: Path, checkpoint: str, num_samples: int, device: str) -> list[list[str]]:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tf:
        out = Path(tf.name)

    cmd = [
        sys.executable,
        str(script_dir / "sampling.py"),
        checkpoint,
        "--num-samples",
        str(num_samples),
        "--output-file",
        str(out),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True)

    lines = out.read_text(encoding="utf-8").splitlines()
    out.unlink(missing_ok=True)
    return [ln.strip().split() for ln in lines if ln.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcfg", required=True, help="Path to candidate PCFG csv (e.g., ../pcfg1.csv)")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint relative to code/ (e.g., checkpoints/pcfg1.pt)")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    # Make checkpoint path absolute from the script's perspective
    checkpoint_abs_path = os.path.abspath(os.path.join(script_dir.parent, args.checkpoint))

    model_samples = sample_strings(script_dir, checkpoint_abs_path, args.num_samples, args.device)

    g = PCFG()
    g.load_csv(args.pcfg)

    total = 0.0
    covered = 0
    for s in model_samples:
        lp = g.sentence_logprob(s)
        if lp != float("-inf"):
            covered += 1
            total += lp

    avg_lp = total / covered if covered else float("-inf")
    coverage = covered / max(1, len(model_samples))

    print(f"Samples: {len(model_samples)}")
    print(f"Coverage (non-zero prob under PCFG): {coverage:.3f}")
    print(f"Average log-prob on covered samples: {avg_lp:.4f}")


if __name__ == "__main__":
    main()
