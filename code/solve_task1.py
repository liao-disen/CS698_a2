import argparse
import csv
import subprocess
import tempfile
import os
from collections import defaultdict

def solve_task1(checkpoint_path, output_path, num_samples=1000, device='cpu'):
    # 1. Sample from model
    print(f"Sampling {num_samples} sentences from {checkpoint_path}...")
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_path = tmp_file.name

    import sys
    cmd = [
        sys.executable, "code/sampling.py",
        checkpoint_path,
        "--num-samples", str(num_samples),
        "--output-file", tmp_path,
        "--device", device
    ]
    subprocess.run(cmd, check=True)

    # 2. Read samples
    with open(tmp_path, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    os.remove(tmp_path)

    # 3. Analyze counts
    # Rules (Task 1):
    # S -> NP VP | NNS VP
    # NP -> DT NNS
    # VP -> V NNS
    # DT -> the
    # NNS -> dogs | cats | mice
    # V -> chase | like
    
    counts = defaultdict(int)
    # Initialize counts to avoid ZeroDivisionError if samples miss something (unlikely with 1000 samples)
    
    for s in samples:
        words = s.split()
        # Filter out clearly garbage/truncated lines if any
        if not words: continue

        if len(words) == 4:
            # S -> NP VP -> DT NNS V NNS
            counts[('S', 'NP VP')] += 1
            counts[('NP', 'DT NNS')] += 1 
            counts[('VP', 'V NNS')] += 1
            counts[('DT', 'the')] += 1
            counts[('NNS', words[1])] += 1
            counts[('V', words[2])] += 1
            counts[('NNS', words[3])] += 1
        elif len(words) == 3:
            # S -> NNS VP -> NNS V NNS
            counts[('S', 'NNS VP')] += 1
            counts[('VP', 'V NNS')] += 1
            counts[('NNS', words[0])] += 1
            counts[('V', words[1])] += 1
            counts[('NNS', words[2])] += 1
        else:
            # If length is weird, maybe ignore or log
            pass

    # 4. Normalize and build rules
    rules = []
    
    def normalize(lhs, options):
        total = sum(counts[(lhs, opt)] for opt in options)
        for opt in options:
            prob = counts[(lhs, opt)] / total if total > 0 else 0.0
            
            # Determine type
            rhs_list = opt.split()
            # Heuristic for type: if RHS is lower case words (terminals), it's preterminal
            # In Task 1, terminals are: the, dogs, cats, mice, chase, like
            terminals = {'the', 'dogs', 'cats', 'mice', 'chase', 'like'}
            if len(rhs_list) == 1 and rhs_list[0] in terminals:
                 lhs_type = "preterminal"
            else:
                 lhs_type = "nonterminal"
            
            rules.append({
                'LHS': lhs,
                'RHS': opt,
                'Probability': prob,
                'LHS Type': lhs_type
            })

    normalize('S', ['NP VP', 'NNS VP'])
    normalize('NP', ['DT NNS'])
    normalize('VP', ['V NNS'])
    normalize('DT', ['the'])
    normalize('NNS', ['dogs', 'cats', 'mice'])
    normalize('V', ['chase', 'like'])

    # 5. Write CSV
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'LHS', 'LHS Type', 'RHS', 'Probability'])
        for i, r in enumerate(rules):
            writer.writerow([i, r['LHS'], r['LHS Type'], r['RHS'], f"{r['Probability']:.6f}"])
    
    print(f"PCFG 1 saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="code/checkpoints/pcfg1.pt")
    parser.add_argument("--output", default="pcfg1.csv")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    solve_task1(args.checkpoint, args.output, args.samples, args.device)
