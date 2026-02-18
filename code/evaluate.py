"""Enhanced Task-2/3 evaluator.

Stages:
1) Distribution fidelity (TVD proxy on Transformer-sampled strings)
2) Structural hallucination check (PCFG-sampled strings scored by Transformer)
3) Whitebox local next-token consistency (prefix probing KL)
"""

import argparse
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from pcfg_utils import PCFG
from sampling import _load_checkpoint

EPS = 1e-12


def _resolve_checkpoint(path_arg: str, script_dir: Path) -> Path:
    p = Path(path_arg)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        (script_dir.parent / path_arg),
        (script_dir / path_arg),
        (script_dir / "checkpoints" / Path(path_arg).name),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Checkpoint not found from: {path_arg}")


def _device_from_arg(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(device_arg)


def _sample_from_transformer(model, artifacts, num_samples: int, max_new_tokens: int, device, seed: int):
    import torch

    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    bos_id = artifacts.bos_id
    eos_id = artifacts.eos_id
    pad_id = artifacts.pad_id

    out = []
    for _ in range(num_samples):
        ids = [bos_id]
        for _step in range(max_new_tokens):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=inp).logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if next_id == eos_id:
                break

        toks = []
        for tid in ids[1:]:
            if tid == eos_id:
                break
            if tid == pad_id:
                continue
            toks.append(artifacts.id_to_token[tid])
        if toks:
            out.append(toks)
    return out


def transformer_sentence_logprob(model, artifacts, tokens, device) -> float:
    """Exact autoregressive log P(tokens + <eos> | <bos>)."""
    import torch

    ids = [artifacts.bos_id]
    ids.extend([artifacts.token_to_id.get(t, artifacts.unk_id) for t in tokens])
    ids.append(artifacts.eos_id)

    logp = 0.0
    for t in range(1, len(ids)):
        inp = torch.tensor([ids[:t]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
        p = float(probs[ids[t]].item())
        logp += math.log(max(p, EPS))
    return logp


def _normalize_log_scores(log_scores):
    if not log_scores:
        return []
    m = max(log_scores)
    exps = [math.exp(s - m) for s in log_scores]
    z = sum(exps) + EPS
    return [e / z for e in exps]


def _l1_distance(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))


def _js_divergence(p, q):
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(x, y):
        s = 0.0
        for a, b in zip(x, y):
            if a > 0:
                s += a * math.log(a / max(b, EPS))
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _rule_integrity_report(g: PCFG):
    lhs_sums = g.lhs_probability_sums()
    lhs_bad = {k: v for k, v in lhs_sums.items() if abs(v - 1.0) > 1e-3}

    term_counts = g.terminal_coverage_counts()
    uncovered = [w for w, c in term_counts.items() if c == 0]
    multi = [w for w, c in term_counts.items() if c > 1]

    return {
        "total_rules": g.rule_count(),
        "lhs_count": len(lhs_sums),
        "lhs_bad_count": len(lhs_bad),
        "lhs_bad_examples": list(lhs_bad.items())[:8],
        "terminal_types": len(term_counts),
        "terminals_multi_assigned": len(multi),
        "terminals_uncovered": len(uncovered),
    }


def _pcfg_next_token_distribution_mc(g: PCFG, prefix_tokens, vocab, *, samples=3000, seed=0):
    rng = random.Random(seed)
    counts = Counter()

    tries = 0
    max_tries = samples * 10
    plen = len(prefix_tokens)

    while sum(counts.values()) < samples and tries < max_tries:
        tries += 1
        sent = g.sample_sentence(start_symbol="S", max_steps=200)
        if len(sent) <= plen:
            continue
        if sent[:plen] == prefix_tokens:
            counts[sent[plen]] += 1

    dist = np.zeros(len(vocab), dtype=np.float64)
    if counts:
        total = sum(counts.values())
        idx = {w: i for i, w in enumerate(vocab)}
        for w, c in counts.items():
            if w in idx:
                dist[idx[w]] = c / total
    return dist


def _transformer_next_token_distribution(model, artifacts, prefix_tokens, vocab, device):
    import torch

    prefix_ids = [artifacts.bos_id]
    prefix_ids.extend([artifacts.token_to_id.get(t, artifacts.unk_id) for t in prefix_tokens])

    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    vec = np.zeros(len(vocab), dtype=np.float64)
    for i, w in enumerate(vocab):
        tid = artifacts.token_to_id.get(w, None)
        if tid is not None:
            vec[i] = probs[tid]
    z = vec.sum()
    if z > 0:
        vec /= z
    return vec


def _kl(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--num-samples", type=int, default=500, help="Transformer sample count for TVD proxy")
    ap.add_argument("--pcfg-hallucination-samples", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=32)

    ap.add_argument("--prefix-probe-count", type=int, default=8)
    ap.add_argument("--prefix-len", type=int, default=2)
    ap.add_argument("--pcfg-prefix-mc-samples", type=int, default=3000)

    ap.add_argument("--near-zero-logp", type=float, default=-30.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    ckpt = _resolve_checkpoint(args.checkpoint, script_dir)
    device = _device_from_arg(args.device)

    model, artifacts = _load_checkpoint(ckpt, device=device)

    g = PCFG()
    g.load_csv(args.pcfg)

    # -------- Stage 1: Distribution fidelity (TVD proxy) --------
    model_samples = _sample_from_transformer(
        model,
        artifacts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        device=device,
        seed=args.seed,
    )

    p_logs = []
    q_logs = []
    covered = 0
    for s in model_samples:
        lp_t = transformer_sentence_logprob(model, artifacts, s, device=device)
        lp_q = g.sentence_logprob(s)
        p_logs.append(lp_t)
        q_logs.append(lp_q if lp_q != float("-inf") else -1e9)
        if lp_q != float("-inf"):
            covered += 1

    p = _normalize_log_scores(p_logs)
    q = _normalize_log_scores(q_logs)
    l1 = _l1_distance(p, q)
    tvd_proxy = 0.5 * l1
    js = _js_divergence(p, q)

    # -------- Stage 2: Hallucination check --------
    pcfg_samples = [g.sample_sentence(start_symbol="S", max_steps=200) for _ in range(args.pcfg_hallucination_samples)]
    pcfg_samples = [s for s in pcfg_samples if s]

    hall_lp = [transformer_sentence_logprob(model, artifacts, s, device=device) for s in pcfg_samples]
    near_zero = sum(1 for x in hall_lp if x < args.near_zero_logp)
    hall_rate = near_zero / max(1, len(hall_lp))

    # -------- Stage 3: Prefix probing KL --------
    prefixes = []
    for s in model_samples:
        if len(s) >= args.prefix_len:
            prefixes.append(tuple(s[: args.prefix_len]))
    random.Random(args.seed).shuffle(prefixes)
    prefixes = prefixes[: args.prefix_probe_count]

    vocab = g.vocabulary()
    fwd_kl_vals = []
    rev_kl_vals = []
    for i, pref in enumerate(prefixes):
        p_next = _transformer_next_token_distribution(model, artifacts, list(pref), vocab, device)
        q_next = _pcfg_next_token_distribution_mc(
            g,
            list(pref),
            vocab,
            samples=args.pcfg_prefix_mc_samples,
            seed=args.seed + i,
        )
        if q_next.sum() <= 0:
            continue
        fwd_kl_vals.append(_kl(p_next, q_next))
        rev_kl_vals.append(_kl(q_next, p_next))

    # -------- Integrity checks --------
    integ = _rule_integrity_report(g)

    print("=== Stage 1: Distribution Fidelity (TVD Proxy) ===")
    print(f"Transformer samples: {len(model_samples)}")
    print(f"PCFG coverage on transformer samples: {covered}/{len(model_samples)} = {covered/max(1, len(model_samples)):.3f}")
    print(f"Proxy TVD over sampled support: {tvd_proxy:.6f}")
    print(f"L1 distance over sampled support: {l1:.6f}")
    print(f"JS divergence over sampled support: {js:.6f}")

    print("\n=== Stage 2: Structural Integrity (Hallucination Check) ===")
    print(f"PCFG samples tested: {len(pcfg_samples)}")
    print(f"Transformer near-zero logp threshold: {args.near_zero_logp}")
    print(f"Near-zero fraction: {near_zero}/{len(pcfg_samples)} = {hall_rate:.3f}")
    if hall_lp:
        print(f"Mean transformer logp on PCFG samples: {sum(hall_lp)/len(hall_lp):.4f}")

    print("\n=== Stage 3: Whitebox Prefix Probing ===")
    print(f"Prefixes tested: {len(prefixes)}")
    if fwd_kl_vals:
        print(f"Avg KL(P_transformer || Q_pcfg_next): {sum(fwd_kl_vals)/len(fwd_kl_vals):.6f}")
        print(f"Avg KL(Q_pcfg_next || P_transformer): {sum(rev_kl_vals)/len(rev_kl_vals):.6f}")
    else:
        print("Prefix KL unavailable (insufficient prefix-conditioned PCFG samples)")

    print("\n=== Rule Sparsity / Constraint Checks ===")
    print(f"Total rules: {integ['total_rules']}")
    print(f"LHS groups: {integ['lhs_count']}")
    print(f"LHS sum!=1 count: {integ['lhs_bad_count']}")
    if integ["lhs_bad_examples"]:
        print(f"Bad LHS examples (lhs,sum): {integ['lhs_bad_examples']}")
    print(f"Terminal types covered: {integ['terminal_types']}")
    print(f"Terminals multi-assigned: {integ['terminals_multi_assigned']}")


if __name__ == "__main__":
    main()
