#!/usr/bin/env python3
"""Sample from checkpoint and measure sentence coverage under predefined tag/binary rules.

Inputs:
- word category map JSON (e.g., code/task2_pre/word_categories_k14.json)
- binary rule JSON (e.g., code/task2_pre/artifacts/binary_rules_25.json)

Output:
- coverage summary
- uncovered sentences + reason
- optional JSON report
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class LoadedArtifacts:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str

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
    id_to_token = [""] * vocab_size
    for tok, idx in token_to_id.items():
        id_to_token[idx] = tok
    return id_to_token


def load_checkpoint(checkpoint_path: Path, device: "torch.device") -> Tuple["GPT2LMHeadModel", LoadedArtifacts]:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    payload = torch.load(checkpoint_path, map_location=device)
    token_to_id = payload["vocab"]
    specials = payload.get("special_tokens", {})
    bos_token = specials.get("bos_token", "<bos>")
    eos_token = specials.get("eos_token", "<eos>")
    pad_token = specials.get("pad_token", "<pad>")
    unk_token = specials.get("unk_token", "<unk>")

    if bos_token not in token_to_id and eos_token in token_to_id:
        bos_token = eos_token

    config = GPT2Config(**payload["config"])
    model = GPT2LMHeadModel(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return model, LoadedArtifacts(
        token_to_id=token_to_id,
        id_to_token=_build_id_to_token(token_to_id),
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        unk_token=unk_token,
    )


def sample_sentence(model, artifacts: LoadedArtifacts, device: "torch.device", max_new_tokens: int, top_k: int) -> str:
    import torch

    ids = [artifacts.bos_id]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=x).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        if top_k > 0 and top_k < probs.shape[0]:
            p, idx = torch.topk(probs, k=top_k)
            p = p / p.sum()
            pick = int(torch.multinomial(p, 1).item())
            next_id = int(idx[pick].item())
        else:
            next_id = int(torch.multinomial(probs, 1).item())

        ids.append(next_id)
        if next_id == artifacts.eos_id:
            break

    toks: List[str] = []
    for t in ids[1:]:
        if t == artifacts.eos_id:
            break
        if t == artifacts.pad_id:
            continue
        toks.append(artifacts.id_to_token[t])
    return " ".join(toks)


def cky_covers(words: List[str], word_to_tag: Dict[str, str], binary_rules: List[Tuple[str, str, str]]) -> Tuple[bool, str]:
    n = len(words)
    if n == 0:
        return False, "empty sentence"

    # lexical tagging
    lex_tags: List[Set[str]] = []
    for w in words:
        tag = word_to_tag.get(w)
        if tag is None:
            return False, f"unknown word: {w}"
        lex_tags.append({tag})

    # index grammar B C -> A
    rhs_to_lhs: Dict[Tuple[str, str], Set[str]] = {}
    for a, b, c in binary_rules:
        rhs_to_lhs.setdefault((b, c), set()).add(a)

    chart: List[List[Set[str]]] = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        chart[i][i] = set(lex_tags[i])

    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            cell = chart[i][j]
            for k in range(i, j):
                left = chart[i][k]
                right = chart[k + 1][j]
                if not left or not right:
                    continue
                for b in left:
                    for c in right:
                        for a in rhs_to_lhs.get((b, c), ()):  # maybe empty
                            cell.add(a)

    if "S" in chart[0][n - 1]:
        return True, ""
    return False, f"no S derivation from tags: {' '.join(next(iter(t)) for t in lex_tags)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample sentences and check coverage by predefined grammar rules")
    parser.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    parser.add_argument("--word-categories", default="code/task2_pre/word_categories_k14.json")
    parser.add_argument("--binary-rules", default="code/task2_pre/artifacts/binary_rules_25.json")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--show-uncovered", type=int, default=50)
    args = parser.parse_args()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device if args.device != "auto" else "cpu")

    root = Path(__file__).resolve().parents[2]
    checkpoint = (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    cat_path = (root / args.word_categories).resolve() if not Path(args.word_categories).is_absolute() else Path(args.word_categories)
    rule_path = (root / args.binary_rules).resolve() if not Path(args.binary_rules).is_absolute() else Path(args.binary_rules)

    cats = json.loads(cat_path.read_text())
    rules_obj = json.loads(rule_path.read_text())
    word_to_tag: Dict[str, str] = cats["word_to_tag"]
    binary_rules = [tuple(x) for x in rules_obj["rules"]]

    model, artifacts = load_checkpoint(checkpoint, device)

    covered = 0
    uncovered_rows = []
    samples = []

    for _ in range(args.num_samples):
        sent = sample_sentence(model, artifacts, device, args.max_new_tokens, args.top_k).strip()
        if not sent:
            continue
        words = sent.split()
        ok, reason = cky_covers(words, word_to_tag, binary_rules)
        samples.append(sent)
        if ok:
            covered += 1
        else:
            uncovered_rows.append({"sentence": sent, "reason": reason})

    total = len(samples)
    cov = covered / total if total else 0.0

    print(f"samples_checked={total}")
    print(f"covered={covered}")
    print(f"coverage={cov:.4f}")
    print("\nUncovered sentences:")
    for row in uncovered_rows[: args.show_uncovered]:
        print(f"- {row['sentence']}\n  reason: {row['reason']}")

    if args.report_json:
        rp = Path(args.report_json)
        if not rp.is_absolute():
            rp = (root / rp).resolve()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({
            "num_samples_requested": args.num_samples,
            "samples_checked": total,
            "covered": covered,
            "coverage": cov,
            "uncovered": uncovered_rows,
        }, indent=2, ensure_ascii=False))
        print(f"\nSaved report: {rp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
