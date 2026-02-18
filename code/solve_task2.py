import argparse
import csv
import os
import subprocess
import tempfile
from collections import Counter, defaultdict


# -----------------------------------------------------------------------------
# Syntax-guided Task 2 solver
#
# Goal: recover a compact CNF PCFG for English-like data.
# - Build one preterminal assignment per terminal word (required by task tips).
# - Use lightweight English syntax heuristics for POS-like tags.
# - Use sampled context to split verbs into transitive/intransitive.
# - Emit exactly 20 nonterminal binary rules (Task 2 prior) + lexical rules.
# -----------------------------------------------------------------------------


def sample_sentences(checkpoint_path: str, num_samples: int, device: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        "python",
        os.path.join(script_dir, "sampling.py"),
        checkpoint_path,
        "--num-samples",
        str(num_samples),
        "--output-file",
        out_path,
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True)

    sents = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if toks:
                sents.append(toks)

    os.remove(out_path)
    return sents


def collect_vocab(checkpoint_path: str, device: str = "cpu"):
    from sampling import _load_checkpoint

    model, artifacts = _load_checkpoint(os.path.abspath(checkpoint_path), device=device)
    _ = model  # silence lints
    vocab = [w for w in artifacts.id_to_token if w and w not in {"<bos>", "<eos>", "<pad>", "<unk>"}]
    return vocab


def guess_base_tag(word: str):
    det = {"the", "a", "an", "this", "that", "these", "those"}
    prep = {
        "in", "on", "at", "by", "with", "from", "to", "near", "inside", "outside", "upstairs",
        "downstairs", "abroad", "away", "back", "home", "everywhere", "elsewhere", "around", "over",
        "under", "across", "through", "within", "without", "beyond", "before", "after", "during",
    }
    aux = {"can", "may", "will", "should", "could", "would", "might", "must"}
    adv = {
        "today", "tomorrow", "yesterday", "soon", "really", "very", "quite", "too", "also",
        "always", "often", "never", "here", "there",
    }
    adjs = {
        "big", "small", "noisy", "quiet", "happy", "sad", "serious", "kind", "brave", "smart",
        "ancient", "modern", "quick", "slow", "gentle", "careful", "eager", "honest", "calm",
    }

    if word in det:
        return "DET"
    if word in prep:
        return "PREP"
    if word in aux:
        return "AUX"
    if word in adv:
        return "ADV"
    if word in adjs:
        return "ADJ"
    if word.isdigit() or word in {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
    }:
        return "NUM"

    # morphology-driven defaults
    if word.endswith("ly"):
        return "ADV"
    if word.endswith("s") and len(word) > 3:
        # many nouns are plural, but verbs can also end with -s.
        return "NOUN"

    return "UNK"


def infer_tags(vocab, samples):
    # Initial pass from static heuristics
    tag = {w: guess_base_tag(w) for w in vocab}

    # Collect contexts
    left = defaultdict(Counter)
    right = defaultdict(Counter)
    freq = Counter()

    for s in samples:
        for i, w in enumerate(s):
            freq[w] += 1
            if i > 0:
                left[w][s[i - 1]] += 1
            if i + 1 < len(s):
                right[w][s[i + 1]] += 1

    det_words = {"the", "a", "an", "this", "that", "these", "those"}
    num_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
    prep_words = {
        "in", "on", "at", "by", "with", "from", "to", "near", "inside", "outside", "upstairs",
        "downstairs", "abroad", "away", "back", "home", "everywhere", "elsewhere", "around", "over",
        "under", "across", "through", "within", "without", "beyond", "before", "after", "during",
    }

    # Fill UNK using context + morphology, balancing noun-vs-verb evidence
    for w in vocab:
        if tag[w] != "UNK":
            continue

        l = left[w]
        r = right[w]

        noun_score = (
            sum(c for t, c in l.items() if t in det_words or t in num_words or t in prep_words)
            + sum(c for t, c in r.items() if t in prep_words)
        )
        verb_score = (
            sum(c for t, c in l.items() if t in det_words or t in num_words)
            + sum(c for t, c in r.items() if t in det_words or t in num_words)
        )

        if w.endswith("ed") or w.endswith("ing"):
            verb_score += 3
        if w.endswith("s") and len(w) > 3:
            noun_score += 1

        tag[w] = "VERB" if verb_score >= noun_score else "NOUN"

    # Split VERB into VTR / VINTR.
    # Use (a) clear English intransitive verb families + (b) right-context object evidence.
    intr_roots = {
        "arriv", "depart", "sleep", "laugh", "smil", "cry", "cough", "sneez",
        "wander", "travel", "pause", "collapse", "shout", "wait", "dance"
    }

    noun_starts = det_words | num_words | {
        w for w, t in tag.items() if t in {"NOUN", "ADJ", "NUM"}
    }

    def stemish(x: str):
        for suf in ("ing", "ed", "es", "s"):
            if x.endswith(suf) and len(x) > len(suf) + 1:
                return x[: -len(suf)]
        return x

    for w in list(vocab):
        if tag[w] != "VERB":
            continue

        st = stemish(w)
        if any(st.startswith(r) for r in intr_roots):
            tag[w] = "VINTR"
            continue

        total_r = sum(right[w].values())
        obj_like = sum(c for nxt, c in right[w].items() if nxt in noun_starts)
        ratio = (obj_like / total_r) if total_r > 0 else 0.0
        tag[w] = "VTR" if ratio >= 0.40 else "VINTR"

    return tag, freq


def normalize_probs_by_lhs(rules):
    by_lhs = defaultdict(float)
    for r in rules:
        by_lhs[r["LHS"]] += r["Probability"]

    out = []
    for r in rules:
        z = by_lhs[r["LHS"]]
        p = r["Probability"] / z if z > 0 else 0.0
        out.append({**r, "Probability": p})
    return out


def build_nonterminal_rules():
    # Exactly 20 binary rules (Task 2 prior), all CNF.
    # Tuned for sampled distribution: NP/PP-heavy with optional leading ADJ/ADV and modal auxiliaries.
    rules = [
        # S expansions
        {"LHS": "S", "RHS": "NP VP", "Probability": 0.44},
        {"LHS": "S", "RHS": "NP VINTR", "Probability": 0.18},
        {"LHS": "S", "RHS": "NOUN VP", "Probability": 0.08},
        {"LHS": "S", "RHS": "NOUN VINTR", "Probability": 0.04},
        {"LHS": "S", "RHS": "ADJ S", "Probability": 0.10},
        {"LHS": "S", "RHS": "ADV S", "Probability": 0.10},
        {"LHS": "S", "RHS": "S ADV", "Probability": 0.04},
        {"LHS": "S", "RHS": "NP AUXVP", "Probability": 0.02},

        # auxiliary bridge
        {"LHS": "AUXVP", "RHS": "AUX VP", "Probability": 1.00},

        # NP structure
        {"LHS": "NP", "RHS": "DET NOUN", "Probability": 0.44},
        {"LHS": "NP", "RHS": "NUM NOUN", "Probability": 0.17},
        {"LHS": "NP", "RHS": "DET NBAR", "Probability": 0.12},
        {"LHS": "NP", "RHS": "NP PP", "Probability": 0.27},
        {"LHS": "NBAR", "RHS": "ADJ NOUN", "Probability": 1.00},

        # PP
        {"LHS": "PP", "RHS": "PREP NP", "Probability": 1.00},

        # VP structure
        {"LHS": "VP", "RHS": "VTR NP", "Probability": 0.47},
        {"LHS": "VP", "RHS": "VTR PP", "Probability": 0.10},
        {"LHS": "VP", "RHS": "VINTR PP", "Probability": 0.17},
        {"LHS": "VP", "RHS": "VP PP", "Probability": 0.20},
        {"LHS": "VP", "RHS": "VP ADV", "Probability": 0.06},
    ]

    assert len(rules) == 20
    wrapped = [{"LHS": r["LHS"], "LHS Type": "nonterminal", "RHS": r["RHS"], "Probability": r["Probability"]} for r in rules]
    return normalize_probs_by_lhs(wrapped)


def build_lexical_rules(vocab, tags, freq):
    # One terminal generated by exactly one preterminal tag.
    by_tag = defaultdict(list)
    for w in vocab:
        by_tag[tags[w]].append(w)

    rules = []
    alpha = 0.5
    for t, ws in by_tag.items():
        total = sum(freq[w] for w in ws) + alpha * len(ws)
        for w in ws:
            p = (freq[w] + alpha) / total if total > 0 else 1.0 / len(ws)
            rules.append({"LHS": t, "LHS Type": "preterminal", "RHS": w, "Probability": p})

    return rules


def save_pcfg(path, nonterm_rules, lex_rules):
    rules = nonterm_rules + lex_rules
    # normalize by LHS once more for safety
    rules = normalize_probs_by_lhs(rules)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "LHS", "LHS Type", "RHS", "Probability"])
        for i, r in enumerate(rules):
            writer.writerow([i, r["LHS"], r["LHS Type"], r["RHS"], f"{r['Probability']:.6f}"])


def solve_task2(checkpoint_path, output_path, num_samples=4000, device="cpu"):
    checkpoint_path = os.path.abspath(checkpoint_path)

    print(f"[1/4] Sampling {num_samples} sentences from checkpoint...")
    samples = sample_sentences(checkpoint_path, num_samples=num_samples, device=device)

    print("[2/4] Loading vocabulary...")
    vocab = collect_vocab(checkpoint_path, device=device)

    print("[3/4] Inferring lexical categories (syntax-guided heuristics)...")
    tags, freq = infer_tags(vocab, samples)
    tag_counts = Counter(tags.values())
    print("Tag inventory:", dict(sorted(tag_counts.items(), key=lambda x: x[0])))

    print("[4/4] Building PCFG and writing CSV...")
    nonterm_rules = build_nonterminal_rules()
    lex_rules = build_lexical_rules(vocab, tags, freq)
    save_pcfg(output_path, nonterm_rules, lex_rules)

    print(f"Saved grammar to: {output_path}")
    print(f"Rules: {len(nonterm_rules)} nonterminal + {len(lex_rules)} preterminal = {len(nonterm_rules)+len(lex_rules)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    parser.add_argument("--output", default="pcfg2.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    solve_task2(args.checkpoint, args.output, num_samples=args.samples, device=args.device)
