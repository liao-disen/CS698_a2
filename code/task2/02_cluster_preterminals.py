from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
from tqdm.auto import tqdm

from task2.clustering import kmeans, pca, zscore
from task2.core import load_teacher, prefix_next_dist
from task2.io_utils import read_jsonl, write_json


SPECIALS = {"<bos>", "<eos>", "<pad>", "<unk>"}


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def _choose_feature_vocab(rows: Sequence[dict], m: int) -> List[str]:
    next_counts = Counter()
    for row in rows:
        toks = row["tokens"]
        c = int(row.get("count", 1))
        for i in range(len(toks) - 1):
            next_counts[toks[i + 1]] += c
    return [w for w, _ in next_counts.most_common(m)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: induce PT clusters")
    ap.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    ap.add_argument("--dev-train", default="code/task2/artifacts/dev_train.jsonl")
    ap.add_argument("--out", default="code/task2/artifacts/preterminals.json")
    ap.add_argument("--k-values", default="15,20,25,30")
    ap.add_argument("--top-m", type=int, default=100)
    ap.add_argument("--max-contexts-per-word", type=int, default=50)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--emission-alpha", type=float, default=0.1)
    ap.add_argument("--progress", dest="progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args()

    teacher = load_teacher(args.checkpoint, device=args.device)
    rows = read_jsonl(args.dev_train)

    vocab = [w for w in teacher.vocab if w not in SPECIALS]
    feat_vocab = _choose_feature_vocab(rows, args.top_m)
    feat_idx = {w: i for i, w in enumerate(feat_vocab)}

    contexts: Dict[str, List[List[str]]] = defaultdict(list)
    word_freq = Counter()
    for row in tqdm(rows, desc="Build contexts", disable=not args.progress):
        toks = row["tokens"]
        c = int(row.get("count", 1))
        for i, w in enumerate(toks):
            if w in SPECIALS:
                continue
            word_freq[w] += c
            repeats = min(c, 2)
            for _ in range(repeats):
                if len(contexts[w]) < args.max_contexts_per_word:
                    contexts[w].append(toks[: i + 1])

    words = [w for w in vocab if contexts.get(w)]
    if not words:
        raise RuntimeError("No words with contexts found in dev train")

    # Teacher-derived feature: average P(next|prefix ending at w).
    features = np.zeros((len(words), len(feat_vocab) + 1), dtype=np.float64)
    full_dists: Dict[str, np.ndarray] = {}
    v_idx = {tok: i for i, tok in enumerate(vocab)}
    for wi, w in enumerate(tqdm(words, desc="Word features", disable=not args.progress)):
        acc = np.zeros(len(feat_vocab), dtype=np.float64)
        full = np.zeros(len(vocab), dtype=np.float64)
        ctxs = contexts[w]
        for pref in ctxs:
            d = prefix_next_dist(teacher, pref, top_k=None)
            for tok, p in d.items():
                if tok in feat_idx:
                    acc[feat_idx[tok]] += p
                if tok in v_idx:
                    full[v_idx[tok]] += p
        acc /= max(1, len(ctxs))
        full /= max(1, len(ctxs))
        z = full.sum()
        if z > 0:
            full /= z
        features[wi, : len(feat_vocab)] = acc
        features[wi, -1] = _entropy(acc / max(acc.sum(), 1e-12))
        full_dists[w] = full

    x = zscore(features)
    x = pca(x, n_components=min(args.pca_dim, x.shape[1]))

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    best = None
    for k in tqdm(k_values, desc="Try K", disable=not args.progress):
        assign, centers = kmeans(x, k=k, seed=args.seed, iters=80)

        # Cluster predictability score: avg KL(word||cluster-mean).
        cluster_means = []
        for ci in range(k):
            idx = np.where(assign == ci)[0]
            if len(idx) == 0:
                cluster_means.append(np.ones(len(vocab), dtype=np.float64) / len(vocab))
                continue
            m = np.mean([full_dists[words[j]] for j in idx], axis=0)
            m = m / max(m.sum(), 1e-12)
            cluster_means.append(m)

        weighted_kl = 0.0
        weight_total = 0.0
        for i, w in enumerate(words):
            wt = float(word_freq[w])
            weighted_kl += wt * _kl(full_dists[w], cluster_means[int(assign[i])])
            weight_total += wt
        score = weighted_kl / max(weight_total, 1e-12)

        if best is None or score < best["score"]:
            best = {
                "k": k,
                "score": float(score),
                "assign": assign,
                "centers": centers,
            }

    assert best is not None
    assign = best["assign"]
    k = int(best["k"])

    word_to_pt: Dict[str, str] = {}
    pt_to_words: Dict[str, List[str]] = defaultdict(list)
    for i, w in enumerate(tqdm(words, desc="Assign PT", disable=not args.progress)):
        pt = f"PT_{int(assign[i]) + 1}"
        word_to_pt[w] = pt
        pt_to_words[pt].append(w)

    # Guarantee total coverage (rare unseen words go to smallest cluster).
    fallback = min(pt_to_words.keys(), key=lambda pt: len(pt_to_words[pt]))
    for w in tqdm(vocab, desc="Fill uncovered", disable=not args.progress):
        if w not in word_to_pt:
            word_to_pt[w] = fallback
            pt_to_words[fallback].append(w)

    # Emission probabilities p(PT -> w).
    emission_counts: Dict[str, Counter] = defaultdict(Counter)
    for w in tqdm(vocab, desc="Emission counts", disable=not args.progress):
        emission_counts[word_to_pt[w]][w] += word_freq[w]

    emissions: Dict[str, Dict[str, float]] = {}
    for pt, c in emission_counts.items():
        denom = sum(c.values()) + args.emission_alpha * len(c)
        emissions[pt] = {
            w: float((cnt + args.emission_alpha) / denom)
            for w, cnt in sorted(c.items())
        }

    out = {
        "selected_k": k,
        "cluster_score": best["score"],
        "feature_vocab": feat_vocab,
        "word_to_pt": word_to_pt,
        "pt_to_words": {k: sorted(v) for k, v in sorted(pt_to_words.items())},
        "emissions": emissions,
        "word_freq": dict(word_freq),
    }
    write_json(args.out, out)

    print(f"selected_k={k}, cluster_score={best['score']:.6f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
