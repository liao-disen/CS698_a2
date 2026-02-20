#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LoadedArtifacts:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    bos_token: str
    eos_token: str
    pad_token: str

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]


def _build_id_to_token(token_to_id: Dict[str, int]) -> List[str]:
    vocab_size = max(token_to_id.values()) + 1
    out = [""] * vocab_size
    for t, i in token_to_id.items():
        out[i] = t
    return out


def load_checkpoint(path: Path, device: "torch.device"):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    payload = torch.load(path, map_location=device)
    token_to_id = payload["vocab"]
    specials = payload.get("special_tokens", {})
    bos = specials.get("bos_token", "<bos>")
    eos = specials.get("eos_token", "<eos>")
    pad = specials.get("pad_token", "<pad>")
    if bos not in token_to_id and eos in token_to_id:
        bos = eos

    model = GPT2LMHeadModel(GPT2Config(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, LoadedArtifacts(token_to_id, _build_id_to_token(token_to_id), bos, eos, pad)


def sample_sentence(model, art: LoadedArtifacts, device: "torch.device", max_new_tokens: int) -> str:
    import torch
    ids = [art.bos_id]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(input_ids=x).logits[0, -1, :], dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
        if nxt == art.eos_id:
            break
    toks = []
    for i in ids[1:]:
        if i == art.eos_id:
            break
        if i == art.pad_id:
            continue
        toks.append(art.id_to_token[i])
    return " ".join(toks)


def parse_and_count(words: List[str], word_to_tag: Dict[str, str], rules: List[Tuple[str, str, str]], counts: Counter) -> bool:
    n = len(words)
    if n == 0:
        return False

    rhs_to_lhs = defaultdict(list)
    for a, b, c in rules:
        rhs_to_lhs[(b, c)].append(a)

    chart = [[set() for _ in range(n)] for _ in range(n)]
    back = [[defaultdict(tuple) for _ in range(n)] for _ in range(n)]  # lhs -> (k,b,c)

    for i, w in enumerate(words):
        t = word_to_tag.get(w)
        if t is None:
            return False
        chart[i][i].add(t)

    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for b in chart[i][k]:
                    for c in chart[k + 1][j]:
                        for a in rhs_to_lhs.get((b, c), []):
                            if a not in chart[i][j]:
                                back[i][j][a] = (k, b, c)
                            chart[i][j].add(a)

    if "S" not in chart[0][n - 1]:
        return False

    def walk(i: int, j: int, sym: str):
        if i == j:
            return
        k, b, c = back[i][j][sym]
        counts[(sym, b, c)] += 1
        walk(i, k, b)
        walk(k + 1, j, c)

    walk(0, n - 1, "S")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    ap.add_argument("--word-categories", default="code/task2_pre/word_categories_k14.json")
    ap.add_argument("--binary-rules", default="code/task2_pre/artifacts/binary_rules_25.json")
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-csv", default="pcfg2_from_rules.csv")
    ap.add_argument("--out-json", default="code/task2_pre/artifacts/rule_prob_estimate.json")
    args = ap.parse_args()

    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[2]
    ckpt = (root / args.checkpoint).resolve()
    cats = json.loads((root / args.word_categories).read_text())
    rules_obj = json.loads((root / args.binary_rules).read_text())
    word_to_tag = cats["word_to_tag"]
    rules = [tuple(r) for r in rules_obj["rules"]]

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    model, art = load_checkpoint(ckpt, device)

    binary_counts = Counter()
    lexical_counts = Counter()
    covered = 0
    total = 0

    for _ in range(args.num_samples):
        s = sample_sentence(model, art, device, args.max_new_tokens).strip()
        if not s:
            continue
        words = s.split()
        total += 1
        for w in words:
            t = word_to_tag.get(w)
            if t is not None:
                lexical_counts[(t, w)] += 1
        if parse_and_count(words, word_to_tag, rules, binary_counts):
            covered += 1

    # normalize binary by lhs with smoothing
    lhs_to_rules = defaultdict(list)
    for a, b, c in rules:
        lhs_to_rules[a].append((a, b, c))

    binary_probs = {}
    for a, rs in lhs_to_rules.items():
        denom = sum(binary_counts[r] for r in rs) + args.alpha * len(rs)
        for r in rs:
            binary_probs[r] = (binary_counts[r] + args.alpha) / denom if denom > 0 else 1.0 / len(rs)

    # lexical normalize by tag
    tag_to_words = defaultdict(list)
    for (t, w), c in lexical_counts.items():
        tag_to_words[t].append((w, c))

    lexical_probs = {}
    for t, wc in tag_to_words.items():
        denom = sum(c for _, c in wc) + args.alpha * len(wc)
        for w, c in wc:
            lexical_probs[(t, w)] = (c + args.alpha) / denom

    # write csv
    out_csv = (root / args.out_csv).resolve()
    rows = []
    rid = 1
    for a, b, c in rules:
        rows.append({"ID": rid, "LHS": a, "LHS Type": "nonterminal", "RHS": f"{b} {c}", "Probability": binary_probs[(a, b, c)]})
        rid += 1
    for (t, w), p in sorted(lexical_probs.items()):
        rows.append({"ID": rid, "LHS": t, "LHS Type": "preterminal", "RHS": w, "Probability": p})
        rid += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["ID", "LHS", "LHS Type", "RHS", "Probability"])
        wr.writeheader()
        wr.writerows(rows)

    out_json = (root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "samples_checked": total,
        "covered": covered,
        "coverage": covered / total if total else 0.0,
        "binary_rule_count": len(rules),
        "lexical_rule_count": len(lexical_probs),
        "out_csv": str(out_csv),
    }, indent=2))

    print(f"samples_checked={total}")
    print(f"covered={covered}")
    print(f"coverage={(covered / total if total else 0.0):.4f}")
    print(f"saved_csv={out_csv}")
    print(f"saved_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
