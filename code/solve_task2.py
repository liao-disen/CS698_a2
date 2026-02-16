import argparse
import subprocess
import tempfile
import os
import csv
import math
from collections import defaultdict
import torch
import torch.nn.functional as F

# Task 2 Strategy:
# 1. Clustering/Tagging:
#    - Terminals are English words.
#    - We need to map words -> Preterminals (POS Tags).
#    - Method: Use the Transformer's embeddings or next-token distributions to cluster words.
#      Words with similar distributions are likely the same POS.
#
# 2. Grammar Induction:
#    - Once we have POS tags, we learn the structure (Rules).
#    - Initialize a CNF grammar with likely rules (S -> NP VP, etc.).
#    - Use EM (Inside-Outside) to refine probabilities.
#    - Or heuristic: Count bigrams of tags in sampled sentences.

def get_word_embeddings(checkpoint_path, device='cpu'):
    """Load model and return word embeddings matrix."""
    # We need to import the internal loading function from sampling.py or duplicate it
    # Easier to just use the code we already have
    from sampling import _load_checkpoint
    
    # Resolving path relative to script if needed
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        return None, None

    model, artifacts = _load_checkpoint(checkpoint_path, device=device)
    
    # Extract embeddings (wte)
    # GPT2 structure: transformer.wte
    embeddings = model.transformer.wte.weight.detach().cpu()
    
    # We only care about actual words, not special tokens for clustering (mostly)
    vocab = artifacts.id_to_token
    return embeddings, vocab

def cluster_words(embeddings, vocab, num_clusters=15):
    """Cluster words into POS tags using K-Means on embeddings."""
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Filter out special tokens (<bos>, <eos>, <pad>, <unk>)
    special_tokens = {'<bos>', '<eos>', '<pad>', '<unk>'}
    valid_indices = [i for i, w in enumerate(vocab) if w not in special_tokens and w != '']
    valid_embeddings = embeddings[valid_indices].numpy()
    valid_words = [vocab[i] for i in valid_indices]
    
    # Normalize embeddings
    norms = np.linalg.norm(valid_embeddings, axis=1, keepdims=True)
    valid_embeddings = valid_embeddings / (norms + 1e-9)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valid_embeddings)
    
    word_to_cluster = {w: labels[i] for i, w in enumerate(valid_words)}
    
    # Print clusters to manual verify
    clusters = defaultdict(list)
    for w, l in word_to_cluster.items():
        clusters[l].append(w)
        
    print("\n--- Word Clusters (Potential POS Tags) ---")
    tag_map = {} # cluster_id -> Tag Name (e.g. 'NNS', 'V')
    
    for cid in sorted(clusters.keys()):
        words = clusters[cid]
        print(f"Cluster {cid}: {words[:10]} ... ({len(words)} words)")
        
        # Heuristic naming (manual for now, or automated based on known words)
        # We can implement a simple lookup if we know some words
        known_nouns = {'dogs', 'cats', 'mice', 'people', 'cars'}
        known_verbs = {'chase', 'like', 'eat', 'see', 'run'}
        known_dets = {'the', 'a', 'an', 'this'}
        known_adj = {'big', 'small', 'red'}
        
        counts = {'NNS': 0, 'V': 0, 'DT': 0, 'JJ': 0}
        for w in words:
            if w in known_nouns: counts['NNS'] += 1
            if w in known_verbs: counts['V'] += 1
            if w in known_dets: counts['DT'] += 1
            if w in known_adj: counts['JJ'] += 1
            
        # Assign best tag or generic 'T_cid'
        best_tag = max(counts, key=counts.get)
        if counts[best_tag] > 0:
            tag_name = best_tag
            # Handle collision? Maybe T_NNS_1, T_NNS_2
            if any(v == tag_name for v in tag_map.values()):
                tag_name = f"{best_tag}_{cid}"
        else:
            tag_name = f"CAT_{cid}"
            
        tag_map[cid] = tag_name
        print(f"  -> Assigned Tag: {tag_name}")

    return word_to_cluster, tag_map

def solve_task2(checkpoint_path, output_path, num_samples=2000, device='cpu'):
    checkpoint_path = os.path.abspath(checkpoint_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Cluster Words to find Terminals -> Preterminals
    print("1. analyzing embeddings to find word categories...")
    embeddings, vocab = get_word_embeddings(checkpoint_path, device=device)
    
    # Task 2 has 200 preterminal rules (A -> w) and 20 nonterminal rules.
    # This implies about 20 POS tags if evenly distributed, or fewer.
    # Let's try clustering into ~15-20 categories.
    word_to_cluster, tag_map = cluster_words(embeddings, vocab, num_clusters=20)
    
    # 2. Sample Sentences
    print(f"\n2. Sampling {num_samples} sentences...")
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    cmd = [
        "python", os.path.join(script_dir, "sampling.py"),
        checkpoint_path,
        "--num-samples", str(num_samples),
        "--output-file", tmp_path,
        "--device", device,
    ]
    subprocess.run(cmd, check=True)
    
    with open(tmp_path, 'r') as f:
        samples = [line.strip().split() for line in f if line.strip()]
    os.remove(tmp_path)
    
    # 3. Convert sentences to Tag Sequences
    # S -> NP VP (which becomes tag sequences like DT NNS V ...)
    tag_sequences = []
    for s in samples:
        seq = []
        for w in s:
            if w in word_to_cluster:
                cid = word_to_cluster[w]
                seq.append(tag_map[cid])
            else:
                seq.append("UNK")
        tag_sequences.append(seq)
        
    # 4. Grammar Induction on Tag Sequences
    # Simple counting of bigrams/trigrams to infer structure?
    # Or strict PCFG induction.
    # Given the constraint: "20 nonterminal expansion rules", the grammar is small.
    # It's likely standard things like S->NP VP, NP->DT NNS, etc.
    
    # Let's count Tag Bigrams to see what follows what
    bigram_counts = defaultdict(int)
    for seq in tag_sequences:
        for i in range(len(seq)-1):
            bigram_counts[(seq[i], seq[i+1])] += 1
            
    print("\nMost Frequent Tag Bigrams:")
    sorted_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    for bg, count in sorted_bigrams[:15]:
        print(f"  {bg}: {count}")

    # Heuristic Construction of Rules based on common English patterns + observed data
    # We need to construct the CSV.
    # Since this is a "solve" script, we can start by just outputting the Preterminal rules (which we found via clustering)
    # And a basic placeholder set of Nonterminal rules (S -> A B) that we might refine manually or via code.
    
    rules = []
    
    # Preterminal Rules: Tag -> word
    # Probability = 1 / size_of_cluster (Uniform approximation for now)
    # Better: Count actual word frequency in samples
    
    word_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    for s in samples:
        for w in s:
            word_counts[w] += 1
            if w in word_to_cluster:
                tag = tag_map[word_to_cluster[w]]
                tag_counts[tag] += 1
                
    for w, count in word_counts.items():
        if w not in word_to_cluster: continue
        tag = tag_map[word_to_cluster[w]]
        # P(Tag -> w) = Count(w) / Count(Tag)
        prob = count / tag_counts[tag] if tag_counts[tag] > 0 else 0
        rules.append({
            'LHS': tag,
            'LHS Type': 'preterminal',
            'RHS': w,
            'Probability': prob
        })
        
    # Nonterminal Rules (Placeholder / Heuristic)
    # We need to parse the tag sequences to learn S -> A B.
    # For a first pass, let's assume a simple structure if we identify standard tags.
    # If we have DT, NNS, V...
    # S -> NP VP
    # NP -> DT NNS
    # VP -> V NNS (or similar)
    
    # IMPORTANT: The assignment says 20 nonterminal rules.
    # We should print the tag map and bigrams to let the user (you) decide the structure
    # or implement the CYK/Inside-Outside loop here.
    
    print("\n--- Generating Skeleton PCFG ---")
    # For now, I will write the preterminal rules to file, and a dummy S rule.
    # You will likely need to refine the Nonterminal rules based on the 'Most Frequent Bigrams' output.
    
    # Dummy top rule to make it valid
    rules.append({'LHS': 'S', 'LHS Type': 'nonterminal', 'RHS': 'CAT_0 CAT_1', 'Probability': 1.0})

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'LHS', 'LHS Type', 'RHS', 'Probability'])
        for i, r in enumerate(rules):
            writer.writerow([i, r['LHS'], r['LHS Type'], r['RHS'], f"{r['Probability']:.6f}"])
            
    print(f"\nSkeleton PCFG saved to {output_path}")
    print("Next steps: Inspect clusters and bigrams, then define the Nonterminal rules manually or via EM.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="code/checkpoints/pcfg2.pt")
    parser.add_argument("--output", default="pcfg2_initial.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    solve_task2(args.checkpoint, args.output, num_samples=args.samples, device=args.device)
