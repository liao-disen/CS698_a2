import argparse
import csv
import os
import random
from collections import Counter

from task2_pipeline import (
    build_lexical_probs,
    build_tag_map,
    cluster_gmm_diag,
    cluster_kmeans,
    collect_substitution_contexts,
    collect_word_frequencies,
    cyk_coverage,
    export_rules_for_csv,
    get_wte_embeddings,
    induce_binary_rules,
    initialize_binary_probs,
    load_model_bundle,
    prune_and_postprocess,
    refine_clusters_with_substitution,
    run_inside_outside_em,
    sample_sentences,
    sentence_to_tags,
    set_seed,
)


def _resolve_checkpoint_path(path_str: str) -> str:
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, path_str),
        os.path.join(script_dir, "checkpoints", os.path.basename(path_str)),
        os.path.join(script_dir, "..", path_str),
        os.path.join(script_dir, "..", "code", path_str),
    ]
    for cand in candidates:
        cand_abs = os.path.abspath(cand)
        if os.path.exists(cand_abs):
            return cand_abs
    raise FileNotFoundError(f"Checkpoint not found: {path_str}")


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "LHS", "LHS Type", "RHS", "Probability"])
        for i, r in enumerate(rows):
            writer.writerow([i, r["LHS"], r["LHS Type"], r["RHS"], f"{float(r['Probability']):.6f}"])


def solve_task2(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    checkpoint = _resolve_checkpoint_path(args.checkpoint)

    print(f"[1/8] Loading checkpoint from {checkpoint}")
    bundle = load_model_bundle(checkpoint, device=args.device)

    print(f"[2/8] Sampling {args.samples} sentences for induction")
    sentences = sample_sentences(
        bundle,
        num_samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        batch_size=args.sample_batch_size,
        show_progress=args.progress,
    )
    print(f"Sampled sentences: {len(sentences)}")

    print("[3/8] Learning preterminal clusters from wte.weight")
    embeddings = get_wte_embeddings(bundle)
    if args.cluster_method == "gmm":
        assignments = cluster_gmm_diag(embeddings, k=args.num_tags, seed=args.seed, iters=args.cluster_iters)
    else:
        assignments = cluster_kmeans(embeddings, k=args.num_tags, seed=args.seed, iters=args.cluster_iters)

    contexts = collect_substitution_contexts(
        sentences,
        word_to_id=bundle.word_to_id,
        bos_id=bundle.artifacts.bos_id,
        max_contexts_per_word=args.probe_contexts_per_word,
        max_prefix_len=args.probe_prefix_len,
        seed=args.seed,
    )

    if args.enable_substitution_probe:
        print("[4/8] Refining clusters with substitution KL probe")
        assignments, probe_stats = refine_clusters_with_substitution(
            bundle,
            bundle.vocab_words,
            embeddings,
            assignments,
            contexts,
            max_probe_words=args.max_probe_words,
            reps_per_cluster=args.probe_representatives,
            probe_weight=args.probe_weight,
            seed=args.seed,
        )
        print(
            f"Probe avg KL: {probe_stats['avg_probe_kl']:.4f}, moved words: {int(probe_stats['moved_words'])}"
        )
    else:
        print("[4/8] Skipping substitution KL refinement (--enable-substitution-probe not set)")

    word_to_tag = build_tag_map(bundle.vocab_words, assignments)
    tag_counts = Counter(word_to_tag.values())
    print(f"Induced preterminal tags: {len(tag_counts)}")

    print("[5/8] Mining exactly 20 binary nonterminal rules (held-out CYK coverage tuning)")
    tagged_sents = [sentence_to_tags(s, word_to_tag) for s in sentences]
    tagged_sents = [s for s in tagged_sents if len(s) >= 2]
    if not tagged_sents:
        raise RuntimeError("No tagged sentences with length >= 2 after sampling")

    rng = random.Random(args.seed)
    shuffled_tagged = list(tagged_sents)
    rng.shuffle(shuffled_tagged)
    holdout_n = int(len(shuffled_tagged) * max(0.0, min(0.9, args.coverage_holdout_ratio)))
    if args.max_coverage_tuning_sentences > 0:
        holdout_n = min(holdout_n, args.max_coverage_tuning_sentences)
    if holdout_n <= 0 and len(shuffled_tagged) >= 20:
        holdout_n = min(args.default_coverage_holdout_min, len(shuffled_tagged) // 5)

    heldout_tagged = shuffled_tagged[:holdout_n]
    induction_tagged = shuffled_tagged[holdout_n:] if holdout_n < len(shuffled_tagged) else shuffled_tagged
    if not induction_tagged:
        induction_tagged = shuffled_tagged

    binary_rules, binary_rule_support, induce_stats = induce_binary_rules(
        induction_tagged,
        num_rules=args.num_binary_rules,
        seed=args.seed,
        num_start_pair_rules=args.num_start_pair_rules,
        heldout_tagged_sentences=heldout_tagged,
        pair_candidate_limit=args.coverage_pair_candidates,
        recursion_candidate_limit=args.coverage_recursion_candidates,
        eval_candidate_limit=args.coverage_eval_candidates,
        max_coverage_eval_sentences=args.max_coverage_eval_sentences,
        coverage_min_gain=args.coverage_min_gain,
        recursion_support_min=args.coverage_recursion_min_support,
        include_split_recursion=args.enable_split_recursion,
    )
    print(
        f"Coverage tuning held-out: {int(induce_stats['heldout_total'])} "
        f"(coverage={induce_stats['heldout_coverage']:.3f})"
    )
    assert len(binary_rules) == args.num_binary_rules

    word_freq = collect_word_frequencies(sentences, bundle.vocab_words)
    lex_probs = build_lexical_probs(bundle.vocab_words, word_to_tag, word_freq, alpha=args.lex_alpha)

    print("[6/8] Estimating probabilities with Inside-Outside EM")
    binary_probs_init = initialize_binary_probs(
        binary_rules,
        rule_scores=binary_rule_support,
        smoothing=args.init_rule_smoothing,
    )
    em_data = [s for s in sentences if 1 <= len(s) <= args.max_sent_len_for_em]
    if args.max_em_sentences > 0:
        em_data = em_data[: args.max_em_sentences]

    binary_probs, lex_probs, ll_hist = run_inside_outside_em(
        em_data,
        binary_rules=binary_rules,
        binary_probs_init=binary_probs_init,
        lex_probs_init=lex_probs,
        em_iters=args.em_iters,
        tol=args.em_tol,
        start_symbol="S",
        show_progress=args.progress,
    )
    if ll_hist:
        print(f"EM iterations: {len(ll_hist)}, last avg log-likelihood: {ll_hist[-1]:.4f}")
    else:
        print("EM did not run (no covered training sentences).")

    print("[7/8] Post-processing constraints + CYK coverage")
    final_binary_rules, final_binary_probs, final_lex_probs = prune_and_postprocess(
        binary_rules,
        binary_probs,
        lex_probs,
        min_prob=args.min_rule_prob,
        target_binary_rules=args.num_binary_rules,
    )

    covered, total, cov_ratio = cyk_coverage(
        [s for s in sentences if 1 <= len(s) <= args.max_sent_len_for_coverage],
        final_binary_rules,
        final_binary_probs,
        final_lex_probs,
        max_sentences=args.max_coverage_sentences if args.max_coverage_sentences > 0 else None,
        start_symbol="S",
    )
    print(f"CYK coverage: {covered}/{total} = {cov_ratio:.3f}")

    print(f"[8/8] Writing grammar to {args.output}")
    rows = export_rules_for_csv(final_binary_rules, final_binary_probs, final_lex_probs)

    # Hard constraint: each terminal appears in exactly one preterminal rule.
    terminal_lhs = {}
    for r in rows:
        if r["LHS Type"] != "preterminal":
            continue
        w = r["RHS"]
        lhs = r["LHS"]
        if w in terminal_lhs and terminal_lhs[w] != lhs:
            raise RuntimeError(f"Terminal {w} assigned to multiple preterminals")
        terminal_lhs[w] = lhs

    _write_csv(args.output, rows)
    print(
        f"Saved {len(rows)} total rules ({len(final_binary_rules)} nonterminal + "
        f"{len(rows) - len(final_binary_rules)} preterminal)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Task 2 model-driven PCFG induction")
    p.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    p.add_argument("--output", default="pcfg2.csv")
    p.add_argument("--samples", type=int, default=30000)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--sample-batch-size", type=int, default=512)

    p.add_argument("--num-tags", type=int, default=12)
    p.add_argument("--cluster-method", choices=["kmeans", "gmm"], default="kmeans")
    p.add_argument("--cluster-iters", type=int, default=30)

    p.add_argument("--enable-substitution-probe", dest="enable_substitution_probe", action="store_true", default=True)
    p.add_argument("--disable-substitution-probe", dest="enable_substitution_probe", action="store_false")
    p.add_argument("--max-probe-words", type=int, default=80)
    p.add_argument("--probe-contexts-per-word", type=int, default=5)
    p.add_argument("--probe-prefix-len", type=int, default=4)
    p.add_argument("--probe-representatives", type=int, default=3)
    p.add_argument("--probe-weight", type=float, default=0.35)

    p.add_argument("--num-binary-rules", type=int, default=20)
    p.add_argument("--num-start-pair-rules", type=int, default=3)
    p.add_argument("--coverage-holdout-ratio", type=float, default=0.2)
    p.add_argument("--default-coverage-holdout-min", type=int, default=500)
    p.add_argument("--max-coverage-tuning-sentences", type=int, default=3000)
    p.add_argument("--coverage-pair-candidates", type=int, default=180)
    p.add_argument("--coverage-recursion-candidates", type=int, default=80)
    p.add_argument("--coverage-eval-candidates", type=int, default=50)
    p.add_argument("--max-coverage-eval-sentences", type=int, default=1500)
    p.add_argument("--coverage-min-gain", type=float, default=0.0)
    p.add_argument("--coverage-recursion-min-support", type=int, default=8)
    p.add_argument("--enable-split-recursion", action="store_true", default=True)
    p.add_argument("--disable-split-recursion", dest="enable_split_recursion", action="store_false")

    p.add_argument("--lex-alpha", type=float, default=0.1)
    p.add_argument("--em-iters", type=int, default=10)
    p.add_argument("--em-tol", type=float, default=1e-4)
    p.add_argument("--max-em-sentences", type=int, default=4000)
    p.add_argument("--max-sent-len-for-em", type=int, default=12)
    p.add_argument("--init-rule-smoothing", type=float, default=1.0)

    p.add_argument("--max-coverage-sentences", type=int, default=1000)
    p.add_argument("--max-sent-len-for-coverage", type=int, default=16)

    p.add_argument("--min-rule-prob", type=float, default=1e-5)
    p.add_argument("--progress", dest="progress", action="store_true", default=True)
    p.add_argument("--no-progress", dest="progress", action="store_false")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    solve_task2(parser.parse_args())
