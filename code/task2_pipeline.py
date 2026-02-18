from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

import numpy as np


EPS = 1e-12


def _maybe_tqdm(iterable, *, enabled: bool = False, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


@dataclass
class LoadedModelBundle:
    model: object
    artifacts: object
    vocab_words: List[str]
    word_to_id: Dict[str, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_model_bundle(checkpoint_path: str, device: str = "cpu") -> LoadedModelBundle:
    import torch

    from sampling import _load_checkpoint

    if device == "auto":
        chosen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        chosen = torch.device(device)

    model, artifacts = _load_checkpoint(checkpoint_path, device=chosen)
    specials = {artifacts.bos_token, artifacts.eos_token, artifacts.pad_token, artifacts.unk_token}
    vocab_words = [w for w in artifacts.id_to_token if w and w not in specials]
    word_to_id = artifacts.token_to_id
    return LoadedModelBundle(model=model, artifacts=artifacts, vocab_words=vocab_words, word_to_id=word_to_id)


def sample_sentences(
    bundle: LoadedModelBundle,
    num_samples: int,
    max_new_tokens: int,
    top_k: int = 0,
    batch_size: int = 64,
    show_progress: bool = False,
) -> List[List[str]]:
    import torch

    from sampling import _decode_sentence

    model = bundle.model
    artifacts = bundle.artifacts
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    out: List[List[str]] = []
    bos_id = artifacts.bos_id
    eos_id = artifacts.eos_id

    produced = 0
    pbar = tqdm(total=num_samples, desc="Sampling", unit="sent", leave=False) if (show_progress and tqdm is not None) else None
    while produced < num_samples:
        cur_batch = min(max(1, batch_size), num_samples - produced)
        produced += cur_batch
        if pbar is not None:
            pbar.update(cur_batch)
        input_ids = torch.full((cur_batch, 1), bos_id, dtype=torch.long, device=device)
        continuations: List[List[int]] = [[] for _ in range(cur_batch)]
        finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)

            if top_k > 0 and top_k < probs.size(-1):
                top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                picked_local = torch.multinomial(top_probs, num_samples=1).squeeze(1)
                next_ids = top_idx.gather(1, picked_local.unsqueeze(1)).squeeze(1)
            else:
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)

            for i in range(cur_batch):
                if not finished[i].item():
                    tok = int(next_ids[i].item())
                    continuations[i].append(tok)
                    if tok == eos_id:
                        finished[i] = True

            input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break

        for cont in continuations:
            text = _decode_sentence(cont, artifacts=artifacts).strip()
            toks = text.split() if text else []
            if toks:
                out.append(toks)

    if pbar is not None:
        pbar.close()
    return out


def get_wte_embeddings(bundle: LoadedModelBundle) -> np.ndarray:
    import torch

    with torch.no_grad():
        emb = bundle.model.transformer.wte.weight.detach().cpu().numpy()
    indices = [bundle.word_to_id[w] for w in bundle.vocab_words]
    return emb[np.asarray(indices)]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + EPS
    return x / denom


def _kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    n = x.shape[0]
    centers = [x[rng.randint(0, n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / (d2.sum() + EPS)
        idx = rng.choice(n, p=probs)
        centers.append(x[idx])
    return np.asarray(centers)


def cluster_kmeans(embeddings: np.ndarray, k: int, seed: int = 0, iters: int = 35) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = _l2_normalize(embeddings.astype(np.float64))
    centers = _kmeans_plus_plus_init(x, k, rng)

    assign = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_assign = dist.argmin(axis=1)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            members = x[assign == c]
            if len(members) == 0:
                centers[c] = x[rng.randint(0, x.shape[0])]
            else:
                centers[c] = members.mean(axis=0)
    return assign


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    z = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS) + m
    if axis is None:
        return z.reshape(())
    return np.squeeze(z, axis=axis)


def cluster_gmm_diag(embeddings: np.ndarray, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = _l2_normalize(embeddings.astype(np.float64))
    n, d = x.shape

    init_assign = cluster_kmeans(x, k, seed=seed, iters=10)
    means = np.vstack(
        [x[init_assign == c].mean(axis=0) if np.any(init_assign == c) else x[rng.randint(0, n)] for c in range(k)]
    )
    variances = np.vstack(
        [
            x[init_assign == c].var(axis=0) + 1e-2
            if np.any(init_assign == c)
            else np.full(d, 1.0, dtype=np.float64)
            for c in range(k)
        ]
    )
    weights = np.full(k, 1.0 / k, dtype=np.float64)

    for _ in range(iters):
        log_resp = np.empty((n, k), dtype=np.float64)
        for c in range(k):
            var = np.maximum(variances[c], 1e-5)
            diff = x - means[c]
            log_gauss = -0.5 * (
                np.sum(np.log(2.0 * np.pi * var)) + np.sum((diff * diff) / var[None, :], axis=1)
            )
            log_resp[:, c] = np.log(weights[c] + EPS) + log_gauss

        norm = _logsumexp(log_resp, axis=1)
        resp = np.exp(log_resp - norm[:, None])

        nk = resp.sum(axis=0) + EPS
        weights = nk / nk.sum()
        means = (resp.T @ x) / nk[:, None]

        for c in range(k):
            diff = x - means[c]
            variances[c] = (resp[:, c][:, None] * diff * diff).sum(axis=0) / nk[c]
            variances[c] = np.maximum(variances[c], 1e-5)

    return resp.argmax(axis=1)


def build_tag_map(words: Sequence[str], cluster_assignments: np.ndarray) -> Dict[str, str]:
    return {w: f"PT_{int(c):02d}" for w, c in zip(words, cluster_assignments.tolist())}


def sentence_to_tags(sent: Sequence[str], word_to_tag: Dict[str, str]) -> List[str]:
    return [word_to_tag[w] for w in sent if w in word_to_tag]


def collect_word_frequencies(sents: Sequence[Sequence[str]], vocab_words: Sequence[str]) -> Counter:
    freq = Counter()
    allowed = set(vocab_words)
    for s in sents:
        for w in s:
            if w in allowed:
                freq[w] += 1
    return freq


def _next_probs_cached(model: object, prefix_ids: Tuple[int, ...], cache: Dict[Tuple[int, ...], np.ndarray], device: object) -> np.ndarray:
    import torch

    key = tuple(prefix_ids)
    if key in cache:
        return cache[key]

    inp = torch.tensor([list(key)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    cache[key] = probs
    return probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.clip(p, EPS, 1.0)
    q_safe = np.clip(q, EPS, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def collect_substitution_contexts(
    sents: Sequence[Sequence[str]],
    word_to_id: Dict[str, int],
    bos_id: int,
    max_contexts_per_word: int,
    max_prefix_len: int,
    seed: int,
) -> Dict[str, List[Tuple[int, ...]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)

    for sent in sents:
        ids = [word_to_id[w] for w in sent if w in word_to_id]
        words = [w for w in sent if w in word_to_id]
        for i, w in enumerate(words):
            start = max(0, i - max_prefix_len)
            prefix = tuple([bos_id] + ids[start:i])
            buckets[w].append(prefix)

    for w in list(buckets.keys()):
        ctxs = buckets[w]
        if len(ctxs) > max_contexts_per_word:
            buckets[w] = rng.sample(ctxs, max_contexts_per_word)
    return buckets


def _cluster_representatives(words: Sequence[str], assignments: np.ndarray, embeddings: np.ndarray, reps_per_cluster: int = 3) -> Dict[int, List[int]]:
    reps: Dict[int, List[int]] = defaultdict(list)
    for c in sorted(set(assignments.tolist())):
        idxs = np.where(assignments == c)[0]
        if len(idxs) == 0:
            continue
        centroid = embeddings[idxs].mean(axis=0)
        dists = np.sum((embeddings[idxs] - centroid[None, :]) ** 2, axis=1)
        top = idxs[np.argsort(dists)[: reps_per_cluster]].tolist()
        reps[c] = top
    return reps


def refine_clusters_with_substitution(
    bundle: LoadedModelBundle,
    words: Sequence[str],
    embeddings: np.ndarray,
    assignments: np.ndarray,
    contexts: Dict[str, List[Tuple[int, ...]]],
    max_probe_words: int = 80,
    reps_per_cluster: int = 3,
    probe_weight: float = 0.35,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = random.Random(seed)

    model = bundle.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        import torch

        device = torch.device("cpu")

    reps = _cluster_representatives(words, assignments, embeddings, reps_per_cluster=reps_per_cluster)
    cluster_ids = sorted(reps.keys())
    if not cluster_ids:
        return assignments, {"avg_probe_kl": 0.0, "moved_words": 0.0}

    word_indices = [i for i, w in enumerate(words) if w in contexts and contexts[w]]
    if len(word_indices) > max_probe_words:
        word_indices = rng.sample(word_indices, max_probe_words)

    emb_norm = _l2_normalize(embeddings)
    centroids = {
        c: emb_norm[np.where(assignments == c)[0]].mean(axis=0) if np.any(assignments == c) else np.zeros(emb_norm.shape[1])
        for c in cluster_ids
    }

    cache: Dict[Tuple[int, ...], np.ndarray] = {}
    moved = 0
    total_probe_kl = 0.0
    total_probe_terms = 0

    for wi in word_indices:
        w = words[wi]
        wid = bundle.word_to_id[w]
        ctxs = contexts[w]

        candidate_scores: List[Tuple[float, int, float]] = []
        for c in cluster_ids:
            rep_idxs = reps[c]
            if not rep_idxs:
                continue

            kl_values: List[float] = []
            for prefix in ctxs:
                base = _next_probs_cached(model, tuple(list(prefix) + [wid]), cache, device)
                for ri in rep_idxs:
                    rid = bundle.word_to_id[words[ri]]
                    sub = _next_probs_cached(model, tuple(list(prefix) + [rid]), cache, device)
                    kl_values.append(kl_divergence(base, sub))

            probe_kl = float(np.mean(kl_values)) if kl_values else 0.0
            emb_score = float(1.0 - np.dot(emb_norm[wi], centroids[c]))
            score = (1.0 - probe_weight) * emb_score + probe_weight * probe_kl
            candidate_scores.append((score, c, probe_kl))

        if not candidate_scores:
            continue

        candidate_scores.sort(key=lambda x: x[0])
        best_score, best_c, best_probe = candidate_scores[0]
        _ = best_score
        total_probe_kl += best_probe
        total_probe_terms += 1

        if int(assignments[wi]) != int(best_c):
            assignments[wi] = int(best_c)
            moved += 1

    avg_probe = total_probe_kl / max(1, total_probe_terms)
    return assignments, {"avg_probe_kl": avg_probe, "moved_words": float(moved)}


def _replace_pair_once(seq: List[str], pair: Tuple[str, str], lhs: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(lhs)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _cyk_sentence_covered_tags(
    tag_seq: Sequence[str],
    binary_rules: Sequence[Tuple[str, str, str]],
    start_symbol: str = "S",
) -> bool:
    n = len(tag_seq)
    if n == 0:
        return False

    chart = defaultdict(set)
    for i, tag in enumerate(tag_seq):
        chart[(i, i + 1)].add(tag)

    by_rhs = defaultdict(list)
    for lhs, b, c in binary_rules:
        by_rhs[(b, c)].append(lhs)

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left = chart[(i, k)]
                right = chart[(k, j)]
                if not left or not right:
                    continue
                for b in left:
                    for c in right:
                        for lhs in by_rhs.get((b, c), []):
                            chart[(i, j)].add(lhs)

    return start_symbol in chart[(0, n)]


def _cyk_coverage_tags(
    tag_sequences: Sequence[Sequence[str]],
    binary_rules: Sequence[Tuple[str, str, str]],
    max_sentences: int | None = None,
    start_symbol: str = "S",
) -> Tuple[int, int, float]:
    subset = list(tag_sequences[:max_sentences]) if max_sentences else list(tag_sequences)
    covered = 0
    for seq in subset:
        if _cyk_sentence_covered_tags(seq, binary_rules, start_symbol=start_symbol):
            covered += 1
    total = len(subset)
    return covered, total, covered / max(1, total)


def induce_binary_rules(
    tagged_sentences: Sequence[Sequence[str]],
    num_rules: int = 20,
    seed: int = 0,
    num_start_pair_rules: int = 3,
    heldout_tagged_sentences: Sequence[Sequence[str]] | None = None,
    pair_candidate_limit: int = 160,
    recursion_candidate_limit: int = 60,
    eval_candidate_limit: int = 40,
    max_coverage_eval_sentences: int = 1500,
    coverage_min_gain: float = 0.0,
    recursion_support_min: int = 5,
    include_split_recursion: bool = True,
) -> Tuple[List[Tuple[str, str, str]], Dict[Tuple[str, str, str], float], Dict[str, float]]:
    _ = seed  # deterministic frequency-driven selection
    train = [list(s) for s in tagged_sentences if len(s) >= 2]
    heldout = [list(s) for s in (heldout_tagged_sentences or []) if len(s) >= 2]
    if not train:
        raise ValueError("No tag sequences with length >= 2; cannot induce binary rules")
    if not heldout:
        heldout = train

    pair_counts = Counter()
    left_chain_counts = Counter()
    right_chain_counts = Counter()
    flat_counts = Counter()
    split_recursion_support = 0

    for seq in train:
        n = len(seq)
        flat_counts.update(seq)
        for i in range(n - 1):
            pair_counts[(seq[i], seq[i + 1])] += 1
        for i in range(0, max(0, n - 2)):
            left_chain_counts[seq[i]] += 1
        for i in range(2, n):
            right_chain_counts[seq[i]] += 1
        if n >= 4:
            split_recursion_support += (n - 3)

    candidate_support: Dict[Tuple[str, str, str], float] = {}

    for (b, c), cnt in pair_counts.most_common(max(1, pair_candidate_limit)):
        candidate_support[("S", b, c)] = float(cnt)

    for b, cnt in left_chain_counts.most_common(max(1, recursion_candidate_limit)):
        if cnt >= recursion_support_min:
            candidate_support[("S", b, "S")] = float(cnt)

    for c, cnt in right_chain_counts.most_common(max(1, recursion_candidate_limit)):
        if cnt >= recursion_support_min:
            candidate_support[("S", "S", c)] = float(cnt)

    if include_split_recursion and split_recursion_support > 0:
        candidate_support[("S", "S", "S")] = float(split_recursion_support)

    if not candidate_support:
        frequent_tags = [t for t, _ in flat_counts.most_common(2)] or ["PT_00", "PT_01"]
        if len(frequent_tags) == 1:
            frequent_tags.append(frequent_tags[0])
        candidate_support[("S", frequent_tags[0], frequent_tags[1])] = 1.0

    selected: List[Tuple[str, str, str]] = []
    selected_set = set()

    for (b, c), _cnt in pair_counts.most_common(max(1, num_start_pair_rules)):
        r = ("S", b, c)
        if r in candidate_support and r not in selected_set:
            selected.append(r)
            selected_set.add(r)
        if len(selected) >= num_rules:
            break

    _, _, current_cov = _cyk_coverage_tags(
        heldout,
        selected,
        max_sentences=max_coverage_eval_sentences if max_coverage_eval_sentences > 0 else None,
        start_symbol="S",
    )

    remaining = [r for r in candidate_support.keys() if r not in selected_set]

    while len(selected) < num_rules and remaining:
        remaining_sorted = sorted(remaining, key=lambda r: candidate_support[r], reverse=True)
        shortlist = remaining_sorted[: max(1, eval_candidate_limit)]

        best_rule = None
        best_cov = current_cov
        best_gain = float("-inf")
        best_support = float("-inf")

        for cand in shortlist:
            trial_rules = selected + [cand]
            _, _, trial_cov = _cyk_coverage_tags(
                heldout,
                trial_rules,
                max_sentences=max_coverage_eval_sentences if max_coverage_eval_sentences > 0 else None,
                start_symbol="S",
            )
            gain = trial_cov - current_cov
            supp = candidate_support[cand]
            if gain > best_gain or (gain == best_gain and supp > best_support):
                best_rule = cand
                best_cov = trial_cov
                best_gain = gain
                best_support = supp

        if best_rule is None:
            break

        if best_gain >= coverage_min_gain or best_gain >= 0.0:
            selected.append(best_rule)
            selected_set.add(best_rule)
            current_cov = best_cov
            remaining = [r for r in remaining if r != best_rule]
            continue

        break

    for cand in sorted(remaining, key=lambda r: candidate_support[r], reverse=True):
        if len(selected) >= num_rules:
            break
        if cand in selected_set:
            continue
        selected.append(cand)
        selected_set.add(cand)

    if len(selected) < num_rules:
        all_tags = [t for t, _ in flat_counts.most_common()] or ["PT_00", "PT_01"]
        if len(all_tags) == 1:
            all_tags.append(all_tags[0])
        i = 0
        while len(selected) < num_rules:
            r = ("S", all_tags[i % len(all_tags)], all_tags[(i + 1) % len(all_tags)])
            i += 1
            if r in selected_set:
                continue
            selected.append(r)
            selected_set.add(r)
            candidate_support.setdefault(r, 1.0)

    selected = selected[:num_rules]
    selected_support = {r: float(candidate_support.get(r, 1.0)) for r in selected}
    _, heldout_total, heldout_cov = _cyk_coverage_tags(
        heldout,
        selected,
        max_sentences=max_coverage_eval_sentences if max_coverage_eval_sentences > 0 else None,
        start_symbol="S",
    )
    stats = {
        "heldout_coverage": float(heldout_cov),
        "heldout_total": float(heldout_total),
    }
    return selected, selected_support, stats


def build_lexical_probs(
    vocab_words: Sequence[str],
    word_to_tag: Dict[str, str],
    word_freq: Counter,
    alpha: float = 0.1,
) -> Dict[Tuple[str, str], float]:
    by_tag = defaultdict(list)
    for w in vocab_words:
        by_tag[word_to_tag[w]].append(w)

    probs: Dict[Tuple[str, str], float] = {}
    for tag, words in by_tag.items():
        z = sum(word_freq[w] for w in words) + alpha * len(words)
        if z <= 0.0:
            z = float(len(words))
        for w in words:
            probs[(tag, w)] = (word_freq[w] + alpha) / z
    return probs


def initialize_binary_probs(
    binary_rules: Sequence[Tuple[str, str, str]],
    rule_scores: Dict[Tuple[str, str, str], float] | None = None,
    smoothing: float = 1.0,
) -> Dict[Tuple[str, str, str], float]:
    by_lhs = defaultdict(list)
    for r in binary_rules:
        by_lhs[r[0]].append(r)

    probs: Dict[Tuple[str, str, str], float] = {}
    for lhs, rs in by_lhs.items():
        raw = []
        for r in rs:
            score = 1.0
            if rule_scores is not None:
                score = float(rule_scores.get(r, 0.0))
            raw.append(max(0.0, score) + smoothing)
        z = sum(raw)
        for r, rv in zip(rs, raw):
            probs[r] = rv / z if z > 0.0 else (1.0 / max(1, len(rs)))
    return probs


def _normalize_rule_probs_binary(binary_probs: Dict[Tuple[str, str, str], float]) -> Dict[Tuple[str, str, str], float]:
    by_lhs = defaultdict(float)
    for (lhs, _, _), p in binary_probs.items():
        by_lhs[lhs] += p

    out = {}
    for r, p in binary_probs.items():
        z = by_lhs[r[0]]
        out[r] = p / z if z > 0 else 0.0
    return out


def _normalize_rule_probs_lex(lex_probs: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    by_lhs = defaultdict(float)
    for (lhs, _), p in lex_probs.items():
        by_lhs[lhs] += p

    out = {}
    for r, p in lex_probs.items():
        z = by_lhs[r[0]]
        out[r] = p / z if z > 0 else 0.0
    return out


def _sentence_inside_outside(
    sent: Sequence[str],
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs: Dict[Tuple[str, str, str], float],
    lex_probs: Dict[Tuple[str, str], float],
    start_symbol: str = "S",
) -> Tuple[float, Dict[Tuple[int, int, str], float], Dict[Tuple[int, int, str], float]]:
    n = len(sent)
    if n == 0:
        return float("-inf"), {}, {}

    symbols = set([start_symbol])
    for lhs, b, c in binary_rules:
        symbols.add(lhs)
        symbols.add(b)
        symbols.add(c)
    for lhs, _ in lex_probs.keys():
        symbols.add(lhs)

    inside = defaultdict(lambda: float("-inf"))

    # lexical spans
    for i, w in enumerate(sent):
        for A in symbols:
            p = lex_probs.get((A, w), 0.0)
            if p > 0:
                inside[(i, i + 1, A)] = math.log(p)

    by_rhs = defaultdict(list)
    for r in binary_rules:
        by_rhs[(r[1], r[2])].append(r)

    def logsumexp_pair(a: float, b: float) -> float:
        if a == float("-inf"):
            return b
        if b == float("-inf"):
            return a
        m = a if a > b else b
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    # inside
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left_syms = [A for A in symbols if inside[(i, k, A)] != float("-inf")]
                right_syms = [B for B in symbols if inside[(k, j, B)] != float("-inf")]
                for b in left_syms:
                    ib = inside[(i, k, b)]
                    for c in right_syms:
                        ic = inside[(k, j, c)]
                        for (lhs, _, _) in by_rhs.get((b, c), []):
                            p = binary_probs[(lhs, b, c)]
                            if p <= 0:
                                continue
                            v = math.log(p) + ib + ic
                            inside[(i, j, lhs)] = logsumexp_pair(inside[(i, j, lhs)], v)

    log_z = inside[(0, n, start_symbol)]
    if log_z == float("-inf"):
        return log_z, dict(inside), {}

    # outside
    outside = defaultdict(lambda: float("-inf"))
    outside[(0, n, start_symbol)] = 0.0

    for span in range(n, 0, -1):
        for i in range(0, n - span + 1):
            j = i + span
            for (lhs, b, c) in binary_rules:
                o = outside[(i, j, lhs)]
                if o == float("-inf"):
                    continue
                p = binary_probs.get((lhs, b, c), 0.0)
                if p <= 0:
                    continue
                lp = math.log(p)
                for k in range(i + 1, j):
                    ib = inside[(i, k, b)]
                    ic = inside[(k, j, c)]
                    if ib != float("-inf") and ic != float("-inf"):
                        # contribute to left child
                        left_val = o + lp + ic
                        prev = outside[(i, k, b)]
                        outside[(i, k, b)] = logsumexp_pair(prev, left_val)

                        # contribute to right child
                        right_val = o + lp + ib
                        prev = outside[(k, j, c)]
                        outside[(k, j, c)] = logsumexp_pair(prev, right_val)

    return log_z, dict(inside), dict(outside)


def run_inside_outside_em(
    sentences: Sequence[Sequence[str]],
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs_init: Dict[Tuple[str, str, str], float],
    lex_probs_init: Dict[Tuple[str, str], float],
    em_iters: int = 8,
    tol: float = 1e-4,
    start_symbol: str = "S",
    show_progress: bool = False,
) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str], float], List[float]]:
    binary_probs = dict(binary_probs_init)
    lex_probs = dict(lex_probs_init)

    history: List[float] = []

    em_iterable = _maybe_tqdm(range(em_iters), enabled=show_progress, desc="EM", unit="iter", leave=False)
    for em_idx in em_iterable:
        expected_bin = defaultdict(float)
        expected_lex = defaultdict(float)
        total_log_likelihood = 0.0
        used = 0

        sent_iterable = _maybe_tqdm(
            sentences,
            enabled=show_progress,
            desc=f"EM sent (iter {em_idx + 1}/{em_iters})",
            unit="sent",
            leave=False,
        )
        for sent in sent_iterable:
            log_z, inside, outside = _sentence_inside_outside(
                sent,
                binary_rules=binary_rules,
                binary_probs=binary_probs,
                lex_probs=lex_probs,
                start_symbol=start_symbol,
            )
            if log_z == float("-inf"):
                continue

            used += 1
            total_log_likelihood += log_z

            n = len(sent)

            for (lhs, b, c) in binary_rules:
                p = binary_probs.get((lhs, b, c), 0.0)
                if p <= 0:
                    continue
                lp = math.log(p)
                acc = float("-inf")
                for span in range(2, n + 1):
                    for i in range(0, n - span + 1):
                        j = i + span
                        o = outside.get((i, j, lhs), float("-inf"))
                        if o == float("-inf"):
                            continue
                        for k in range(i + 1, j):
                            ib = inside.get((i, k, b), float("-inf"))
                            ic = inside.get((k, j, c), float("-inf"))
                            if ib == float("-inf") or ic == float("-inf"):
                                continue
                            v = o + lp + ib + ic - log_z
                            if acc == float("-inf"):
                                acc = v
                            else:
                                m = v if v > acc else acc
                                acc = m + math.log(math.exp(acc - m) + math.exp(v - m))
                if acc != float("-inf"):
                    expected_bin[(lhs, b, c)] += math.exp(acc)

            for i, w in enumerate(sent):
                for (lhs, term), p in lex_probs.items():
                    if term != w or p <= 0:
                        continue
                    o = outside.get((i, i + 1, lhs), float("-inf"))
                    if o == float("-inf"):
                        continue
                    gamma = math.exp(o + math.log(p) - log_z)
                    expected_lex[(lhs, term)] += gamma

        if used == 0:
            history.append(float("-inf"))
            break

        # M-step
        by_lhs_bin = defaultdict(float)
        by_lhs_lex = defaultdict(float)

        for r, c in expected_bin.items():
            by_lhs_bin[r[0]] += c
        for r, c in expected_lex.items():
            by_lhs_lex[r[0]] += c

        # Keep support fixed to initial rule sets; add tiny smoothing for stability.
        for r in binary_probs.keys():
            c = expected_bin.get(r, 0.0) + 1e-8
            z = by_lhs_bin[r[0]] + 1e-8 * sum(1 for rr in binary_probs.keys() if rr[0] == r[0])
            binary_probs[r] = c / z if z > 0 else binary_probs[r]

        for r in lex_probs.keys():
            c = expected_lex.get(r, 0.0) + 1e-8
            z = by_lhs_lex[r[0]] + 1e-8 * sum(1 for rr in lex_probs.keys() if rr[0] == r[0])
            lex_probs[r] = c / z if z > 0 else lex_probs[r]

        binary_probs = _normalize_rule_probs_binary(binary_probs)
        lex_probs = _normalize_rule_probs_lex(lex_probs)

        avg_ll = total_log_likelihood / used
        history.append(avg_ll)
        if show_progress and tqdm is not None and hasattr(em_iterable, "set_postfix"):
            em_iterable.set_postfix(avg_ll=f"{avg_ll:.4f}")
        if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
            break

    return binary_probs, lex_probs, history


def cyk_sentence_covered(
    sent: Sequence[str],
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs: Dict[Tuple[str, str, str], float],
    lex_probs: Dict[Tuple[str, str], float],
    start_symbol: str = "S",
) -> bool:
    n = len(sent)
    if n == 0:
        return False

    chart = defaultdict(set)

    # lexical
    for i, w in enumerate(sent):
        for (lhs, term), p in lex_probs.items():
            if term == w and p > 0.0:
                chart[(i, i + 1)].add(lhs)

    # binary
    by_rhs = defaultdict(list)
    for (lhs, b, c) in binary_rules:
        if binary_probs.get((lhs, b, c), 0.0) > 0:
            by_rhs[(b, c)].append(lhs)

    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                left = chart[(i, k)]
                right = chart[(k, j)]
                if not left or not right:
                    continue
                for b in left:
                    for c in right:
                        for lhs in by_rhs.get((b, c), []):
                            chart[(i, j)].add(lhs)

    return start_symbol in chart[(0, n)]


def cyk_coverage(
    sentences: Sequence[Sequence[str]],
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs: Dict[Tuple[str, str, str], float],
    lex_probs: Dict[Tuple[str, str], float],
    max_sentences: int | None = None,
    start_symbol: str = "S",
) -> Tuple[int, int, float]:
    subset = list(sentences[:max_sentences]) if max_sentences else list(sentences)
    covered = 0
    for s in subset:
        if cyk_sentence_covered(s, binary_rules, binary_probs, lex_probs, start_symbol=start_symbol):
            covered += 1
    total = len(subset)
    return covered, total, covered / max(1, total)


def _apply_lhs_probability_floor(probs: Dict[Tuple, float], floor: float) -> Dict[Tuple, float]:
    if floor <= 0.0:
        return dict(probs)

    by_lhs = defaultdict(list)
    for r, p in probs.items():
        by_lhs[r[0]].append((r, max(0.0, float(p))))

    out: Dict[Tuple, float] = {}
    for lhs, items in by_lhs.items():
        n = len(items)
        if n == 0:
            continue

        # If floor is infeasible, back off to uniform.
        eff_floor = min(floor, 1.0 / n)

        lows = [(r, p) for r, p in items if p < eff_floor]
        highs = [(r, p) for r, p in items if p >= eff_floor]

        fixed_mass = eff_floor * len(lows)
        remaining = max(0.0, 1.0 - fixed_mass)

        for r, _ in lows:
            out[r] = eff_floor

        if not highs:
            # Everything got floored: distribute uniformly.
            u = 1.0 / n
            for r, _ in items:
                out[r] = u
            continue

        sum_high = sum(p for _, p in highs)
        if sum_high <= 0:
            u = remaining / len(highs)
            for r, _ in highs:
                out[r] = u
        else:
            for r, p in highs:
                out[r] = remaining * (p / sum_high)

    return out


def prune_and_postprocess(
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs: Dict[Tuple[str, str, str], float],
    lex_probs: Dict[Tuple[str, str], float],
    min_prob: float = 1e-5,
    target_binary_rules: int = 20,
    rule_prob_floor: float = 0.0,
) -> Tuple[List[Tuple[str, str, str]], Dict[Tuple[str, str, str], float], Dict[Tuple[str, str], float]]:
    binary_with_p = [(r, binary_probs.get(r, 0.0)) for r in binary_rules]
    kept = [rp for rp in binary_with_p if rp[1] >= min_prob]

    if len(kept) < target_binary_rules:
        removed = sorted([rp for rp in binary_with_p if rp[1] < min_prob], key=lambda x: x[1], reverse=True)
        needed = target_binary_rules - len(kept)
        kept.extend(removed[:needed])

    if len(kept) > target_binary_rules:
        kept = sorted(kept, key=lambda x: x[1], reverse=True)[:target_binary_rules]

    final_rules = [r for r, _ in kept]
    final_binary_probs = {r: max(binary_probs.get(r, 0.0), min_prob) for r in final_rules}
    final_binary_probs = _normalize_rule_probs_binary(final_binary_probs)
    final_binary_probs = _apply_lhs_probability_floor(final_binary_probs, rule_prob_floor)
    final_binary_probs = _normalize_rule_probs_binary(final_binary_probs)

    final_lex_probs = {r: max(p, min_prob) for r, p in lex_probs.items()}
    final_lex_probs = _normalize_rule_probs_lex(final_lex_probs)
    final_lex_probs = _apply_lhs_probability_floor(final_lex_probs, rule_prob_floor)
    final_lex_probs = _normalize_rule_probs_lex(final_lex_probs)

    return final_rules, final_binary_probs, final_lex_probs


def export_rules_for_csv(
    binary_rules: Sequence[Tuple[str, str, str]],
    binary_probs: Dict[Tuple[str, str, str], float],
    lex_probs: Dict[Tuple[str, str], float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for lhs, b, c in binary_rules:
        rows.append(
            {
                "LHS": lhs,
                "LHS Type": "nonterminal",
                "RHS": f"{b} {c}",
                "Probability": float(binary_probs[(lhs, b, c)]),
            }
        )

    for (lhs, term), p in sorted(lex_probs.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append(
            {
                "LHS": lhs,
                "LHS Type": "preterminal",
                "RHS": term,
                "Probability": float(p),
            }
        )

    # Normalize one last time by LHS for hard constraint.
    by_lhs = defaultdict(float)
    for row in rows:
        by_lhs[row["LHS"]] += float(row["Probability"])

    for row in rows:
        z = by_lhs[row["LHS"]]
        row["Probability"] = float(row["Probability"]) / z if z > 0 else 0.0

    return rows
