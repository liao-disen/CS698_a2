from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from sampling import LoadedArtifacts, _load_checkpoint, _sample_sentence_ids

EPS = 1e-12


@dataclass(frozen=True)
class LoadedTeacher:
    model: "torch.nn.Module"
    artifacts: LoadedArtifacts
    device: "torch.device"

    @property
    def vocab(self) -> List[str]:
        return self.artifacts.id_to_token


@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    prob: float


@dataclass
class Grammar:
    rules: List[Rule]

    def __post_init__(self) -> None:
        self.by_lhs: Dict[str, List[Rule]] = defaultdict(list)
        self.binary_by_rhs: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
        self.lex_by_word: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.nonterminals: set[str] = set()
        for r in self.rules:
            self.by_lhs[r.lhs].append(r)
            self.nonterminals.add(r.lhs)
            if len(r.rhs) == 2:
                self.binary_by_rhs[(r.rhs[0], r.rhs[1])].append((r.lhs, r.prob))
            elif len(r.rhs) == 1:
                self.lex_by_word[r.rhs[0]].append((r.lhs, r.prob))


def _resolve_checkpoint(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute() and p.exists():
        return p

    script_dir = Path(__file__).resolve().parent.parent
    candidates = [
        script_dir / path_arg,
        script_dir / "checkpoints" / p.name,
        script_dir.parent / path_arg,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {path_arg}")


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(device_arg)


def load_teacher(checkpoint: str, device: str = "cpu") -> LoadedTeacher:
    dev = _device_from_arg(device)
    model, artifacts = _load_checkpoint(_resolve_checkpoint(checkpoint), device=dev)
    return LoadedTeacher(model=model, artifacts=artifacts, device=dev)


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0 or top_p >= 1:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    csum = torch.cumsum(sorted_probs, dim=0)
    mask = csum <= top_p
    if mask.numel() > 0:
        mask[0] = True
    kept = torch.zeros_like(probs)
    kept[sorted_idx[mask]] = probs[sorted_idx[mask]]
    z = kept.sum()
    return kept / z if z > 0 else probs


def _next_probs(teacher: LoadedTeacher, prefix_ids: Sequence[int], temperature: float) -> torch.Tensor:
    input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=teacher.device)
    with torch.no_grad():
        logits = teacher.model(input_ids=input_ids).logits[0, -1, :]
        if temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
    return probs


def sample(
    teacher: LoadedTeacher,
    n: int,
    max_len: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 0,
    progress: bool = False,
) -> List[List[str]]:
    if n < 1:
        return []
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if abs(temperature - 1.0) > 1e-8:
        raise ValueError("sample() reuses sampling.py and currently supports temperature=1.0 only")
    if abs(top_p - 1.0) > 1e-8:
        raise ValueError("sample() reuses sampling.py and currently supports top_p=1.0 only")

    bos = teacher.artifacts.bos_id
    eos = teacher.artifacts.eos_id
    pad = teacher.artifacts.pad_id

    out: List[List[str]] = []
    for _ in tqdm(range(n), disable=not progress, desc="Teacher sampling"):
        ids = _sample_sentence_ids(
            teacher.model,
            prefix_ids=[bos],
            artifacts=teacher.artifacts,
            top_k=0,
            max_new_tokens=max_len,
            device=teacher.device,
        )

        toks: List[str] = []
        for tid in ids[1:]:
            if tid == eos:
                break
            if tid == pad:
                continue
            toks.append(teacher.artifacts.id_to_token[tid])
        if toks:
            out.append(toks)
    return out


def logprob(teacher: LoadedTeacher, seq: Sequence[str], include_eos: bool = True) -> float:
    ids = [teacher.artifacts.bos_id]
    ids.extend(teacher.artifacts.token_to_id.get(t, teacher.artifacts.unk_id) for t in seq)
    if include_eos:
        ids.append(teacher.artifacts.eos_id)

    total = 0.0
    for t in range(1, len(ids)):
        probs = _next_probs(teacher, ids[:t], temperature=1.0)
        p = float(probs[ids[t]].item())
        total += math.log(max(p, EPS))
    return total


def prefix_next_dist(
    teacher: LoadedTeacher,
    prefix: Sequence[str],
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Dict[str, float]:
    ids = [teacher.artifacts.bos_id]
    ids.extend(teacher.artifacts.token_to_id.get(t, teacher.artifacts.unk_id) for t in prefix)
    probs = _next_probs(teacher, ids, temperature=max(temperature, 1e-6))

    if top_k is None or top_k <= 0 or top_k >= probs.shape[0]:
        pairs = ((teacher.artifacts.id_to_token[i], float(probs[i].item())) for i in range(probs.shape[0]))
        return dict(pairs)

    vals, idx = torch.topk(probs, k=top_k)
    return {teacher.artifacts.id_to_token[int(i.item())]: float(v.item()) for v, i in zip(vals, idx)}


def avg_logprob_teacher(teacher: LoadedTeacher, samples: Iterable[Sequence[str]], progress: bool = False) -> float:
    sample_list = list(samples)
    vals = [
        logprob(teacher, s)
        for s in tqdm(sample_list, desc="Teacher avg logprob", disable=(not progress) or len(sample_list) == 0)
    ]
    return float(sum(vals) / max(1, len(vals)))


def load_pcfg_csv(path: str) -> Grammar:
    rules: List[Rule] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lhs = row["LHS"].strip()
            rhs = tuple(row["RHS"].strip().split())
            prob = float(row["Probability"])
            rules.append(Rule(lhs=lhs, rhs=rhs, prob=prob))
    return Grammar(rules=rules)


def _logsumexp(vals: List[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def pcfg_inside_prob(grammar: Grammar, seq: Sequence[str], start_symbol: str = "S") -> float:
    n = len(seq)
    if n == 0:
        return 0.0

    chart: Dict[Tuple[int, int, str], float] = defaultdict(lambda: float("-inf"))

    for i, w in enumerate(seq):
        for lhs, p in grammar.lex_by_word.get(w, []):
            chart[(i, i + 1, lhs)] = _logsumexp([chart[(i, i + 1, lhs)], math.log(max(p, EPS))])

    nts = list(grammar.nonterminals)
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            for k in range(i + 1, j):
                for b in nts:
                    lb = chart[(i, k, b)]
                    if lb == float("-inf"):
                        continue
                    for c in nts:
                        lc = chart[(k, j, c)]
                        if lc == float("-inf"):
                            continue
                        for lhs, p in grammar.binary_by_rhs.get((b, c), []):
                            v = math.log(max(p, EPS)) + lb + lc
                            chart[(i, j, lhs)] = _logsumexp([chart[(i, j, lhs)], v])

    root_lp = chart[(0, n, start_symbol)]
    return 0.0 if root_lp == float("-inf") else math.exp(root_lp)
