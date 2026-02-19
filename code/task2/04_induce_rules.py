from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from task2.clustering import kmeans, pca, zscore
from task2.core import load_teacher, logprob, pcfg_inside_prob, prefix_next_dist, Grammar, Rule
from task2.io_utils import read_json, read_jsonl, write_json


SPECIALS = {"<bos>", "<eos>", "<pad>", "<unk>"}


def _collect_internal_nodes(tree, out):
    if tree.get("leaf"):
        return
    out.append((tree["i"], tree["j"], tree["k"], tree))
    _collect_internal_nodes(tree["left"], out)
    _collect_internal_nodes(tree["right"], out)


def _label_tree(tree, node_to_label, pts):
    if tree.get("leaf"):
        i = tree["i"]
        return pts[i]
    key = (tree["i"], tree["j"], tree["k"])
    lhs = node_to_label[key]
    left = _label_tree(tree["left"], node_to_label, pts)
    right = _label_tree(tree["right"], node_to_label, pts)
    return (lhs, left, right)


def _extract_rules(labeled, counts):
    if isinstance(labeled, str):
        return
    lhs, left, right = labeled
    l = left[0] if isinstance(left, tuple) else left
    r = right[0] if isinstance(right, tuple) else right
    counts[(lhs, l, r)] += 1
    _extract_rules(left, counts)
    _extract_rules(right, counts)


def _normalize_rule_counts(rule_counts: Counter) -> Dict[Tuple[str, str, str], float]:
    lhs_tot = Counter()
    for (a, _, _), c in rule_counts.items():
        lhs_tot[a] += c
    return {(a, b, c): v / lhs_tot[a] for (a, b, c), v in rule_counts.items()}


def _renorm_log(log_scores: List[float]) -> np.ndarray:
    if not log_scores:
        return np.zeros(0, dtype=np.float64)
    m = max(log_scores)
    arr = np.exp(np.array(log_scores, dtype=np.float64) - m)
    z = arr.sum()
    return arr / z if z > 0 else np.ones_like(arr) / len(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: induce binary nonterminal rules")
    ap.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    ap.add_argument("--parses", default="code/task2/artifacts/parses.jsonl")
    ap.add_argument("--preterminals", default="code/task2/artifacts/preterminals.json")
    ap.add_argument("--dev-eval", default="code/task2/artifacts/dev_eval.jsonl")
    ap.add_argument("--out", default="code/task2/artifacts/grammar_raw.json")
    ap.add_argument("--metrics-out", default="code/task2/artifacts/rule_metrics.json")
    ap.add_argument("--num-nts", type=int, default=24)
    ap.add_argument("--feat-top-m", type=int, default=120)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--progress", dest="progress", action="store_true", default=True)
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args()

    teacher = load_teacher(args.checkpoint, device=args.device)
    parses = read_jsonl(args.parses)
    pt_data = read_json(args.preterminals)
    dev_eval = read_jsonl(args.dev_eval)

    vocab = [w for w in teacher.vocab if w not in SPECIALS]

    # Build node features from teacher next-token dist at span boundary j.
    # Feature vocab from dev_eval next-token empirical counts.
    nxt_counts = Counter()
    for row in tqdm(dev_eval, desc="Eval token stats", disable=not args.progress):
        toks = row["tokens"]
        c = int(row.get("count", 1))
        for i in range(len(toks) - 1):
            nxt_counts[toks[i + 1]] += c
    feat_vocab = [w for w, _ in nxt_counts.most_common(args.feat_top_m)]

    nodes = []
    feats = []
    for row in tqdm(parses, desc="Span node features", disable=not args.progress):
        toks = row["tokens"]
        internal = []
        _collect_internal_nodes(row["tree"], internal)
        for (i, j, k, _node) in internal:
            d = prefix_next_dist(teacher, toks[:j], top_k=None)
            vec = np.array([d.get(w, 0.0) for w in feat_vocab], dtype=np.float64)
            vec = vec / max(vec.sum(), 1e-12)
            nodes.append((id(row), tuple(toks), i, j, k, row["pts"], row["tree"]))
            feats.append(vec)

    if not feats:
        raise RuntimeError("No internal nodes found in parses")

    x = zscore(np.vstack(feats))
    x = pca(x, n_components=min(args.pca_dim, x.shape[1]))
    assign, _ = kmeans(x, k=args.num_nts, seed=args.seed, iters=100)

    # Group labels per sentence tree key.
    # Key on (tokens tuple, i,j,k) because ids are not stable after reload.
    node_label = {}
    for idx, nd in enumerate(tqdm(nodes, desc="Assign NT labels", disable=not args.progress)):
        _rid, toks, i, j, k, _pts, _tree = nd
        node_label[(toks, i, j, k)] = f"NT_{int(assign[idx]) + 1}"

    rule_counts = Counter()
    for row in tqdm(parses, desc="Extract binary rules", disable=not args.progress):
        toks = tuple(row["tokens"])
        pts = row["pts"]
        t = row["tree"]

        # Force root as S for compatibility.
        root_key = (toks, t["i"], t["j"], t["k"]) if not t.get("leaf") else None
        if root_key is not None:
            node_label[root_key] = "S"

        # Build temporary lookup for this sentence.
        lookup = {}
        internal = []
        _collect_internal_nodes(t, internal)
        for i, j, k, _ in internal:
            lookup[(i, j, k)] = node_label[(toks, i, j, k)]

        def build_local(n):
            if n.get("leaf"):
                return pts[n["i"]]
            lhs = lookup[(n["i"], n["j"], n["k"])]
            return (lhs, build_local(n["left"]), build_local(n["right"]))

        labeled = build_local(t)
        _extract_rules(labeled, rule_counts)

    binary_probs = _normalize_rule_counts(rule_counts)
    binary_rules = [
        {"lhs": a, "rhs": [b, c], "prob": float(p)}
        for (a, b, c), p in sorted(binary_probs.items())
    ]

    lex_rules = []
    for pt, d in sorted(pt_data["emissions"].items()):
        for w, p in sorted(d.items()):
            lex_rules.append({"lhs": pt, "rhs": [w], "prob": float(p)})

    grammar_json = {
        "start": "S",
        "binary_rules": binary_rules,
        "lex_rules": lex_rules,
        "meta": {
            "num_binary_rules": len(binary_rules),
            "num_lex_rules": len(lex_rules),
            "num_nt_clusters": args.num_nts,
        },
    }
    write_json(args.out, grammar_json)

    # Teacher vs PCFG proxy metrics on dev_eval.
    g = Grammar(rules=[Rule(r["lhs"], tuple(r["rhs"]), float(r["prob"])) for r in binary_rules + lex_rules])
    eval_sents = [row["tokens"] for row in dev_eval[:1000]]
    t_lp = [logprob(teacher, s) for s in tqdm(eval_sents, desc="Teacher logprob", disable=not args.progress)]
    q_lp = [
        math.log(max(pcfg_inside_prob(g, s), 1e-300))
        for s in tqdm(eval_sents, desc="PCFG inside logprob", disable=not args.progress)
    ]
    corr = float(np.corrcoef(np.array(t_lp), np.array(q_lp))[0, 1]) if len(t_lp) >= 2 else float("nan")

    p = _renorm_log(t_lp)
    q = _renorm_log(q_lp)
    tvd = float(0.5 * np.abs(p - q).sum()) if len(p) else float("nan")

    write_json(
        args.metrics_out,
        {
            "teacher_pcfg_logprob_corr": corr,
            "proxy_tvd": tvd,
            "num_eval": len(eval_sents),
            "num_binary_rules": len(binary_rules),
        },
    )

    print(f"saved {args.out}")
    print(f"binary_rules={len(binary_rules)}, corr={corr:.4f}, proxy_tvd={tvd:.4f}")


if __name__ == "__main__":
    main()
