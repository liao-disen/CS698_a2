from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
import sys
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from task2.core import avg_logprob_teacher, load_teacher, sample
from task2.io_utils import ensure_dir, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1: build fixed dev set from teacher")
    ap.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    ap.add_argument("--out-dir", default="code/task2/artifacts")
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-ratio", type=float, default=0.2)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--progress", dest="progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    teacher = load_teacher(args.checkpoint, device=args.device)

    samples = sample(
        teacher,
        n=args.num_samples,
        max_len=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        progress=args.progress,
    )

    counts = Counter(tuple(s) for s in samples if s)
    unique = list(counts.items())
    rng = random.Random(args.seed)
    rng.shuffle(unique)

    n_eval = int(len(unique) * max(0.0, min(0.9, args.eval_ratio)))
    dev_eval = unique[:n_eval]
    dev_train = unique[n_eval:]

    train_rows = [{"tokens": list(t), "count": int(c)} for t, c in dev_train]
    eval_rows = [{"tokens": list(t), "count": int(c)} for t, c in dev_eval]
    write_jsonl(out_dir / "dev_train.jsonl", train_rows)
    write_jsonl(out_dir / "dev_eval.jsonl", eval_rows)

    # Teacher sanity hooks on eval set.
    eval_expanded = [list(t) for t, c in dev_eval for _ in range(min(3, c))]
    avg_lp = avg_logprob_teacher(teacher, eval_expanded[:1000], progress=args.progress) if eval_expanded else float("nan")

    len_hist = Counter(len(t) for t, _ in dev_eval)
    uni = Counter()
    bi = Counter()
    for toks, c in tqdm(dev_eval, desc="Eval ngram stats", disable=not args.progress):
        for w in toks:
            uni[w] += c
        for i in range(len(toks) - 1):
            bi[(toks[i], toks[i + 1])] += c

    stats = {
        "num_raw_samples": len(samples),
        "num_unique": len(unique),
        "num_dev_train_unique": len(dev_train),
        "num_dev_eval_unique": len(dev_eval),
        "avg_logprob_teacher_eval_subset": avg_lp,
        "dev_eval_length_hist": {str(k): int(v) for k, v in sorted(len_hist.items())},
        "top_unigrams": [[w, int(c)] for w, c in uni.most_common(50)],
        "top_bigrams": [[[a, b], int(c)] for (a, b), c in bi.most_common(50)],
    }
    write_json(out_dir / "dev_stats.json", stats)

    print(f"Wrote {out_dir / 'dev_train.jsonl'} ({len(train_rows)} unique)")
    print(f"Wrote {out_dir / 'dev_eval.jsonl'} ({len(eval_rows)} unique)")
    print(f"avg_logprob_teacher_eval_subset={avg_lp:.4f}")


if __name__ == "__main__":
    main()
