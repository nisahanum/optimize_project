# experiment_ablation_baseline.py
"""
Ablation & Baseline Experiments for Reviewer #4

This script runs simplified baseline configurations to show
what is LOST when IFPOM is simplified.

Models:
B0: Single-objective (Z1 only), no synergy, no mixed financing
B1: Multi-objective (Z1, Z2), no synergy, no mixed financing
B2: Multi-objective (Z1, Z2), WITH synergy, no mixed financing

Reference:
- IFPOM Full (Z1, Z2, Z3 + synergy + mixed financing)
  is produced separately via experiment_moead_tradeoff_analysis.py

This script is intentionally SIMPLE and DEFENSIBLE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np

# --- Make root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common_ifpom_final as ifpom


# -------------------------------------------------
# Simplified objective evaluators (ABLATIONS)
# -------------------------------------------------
def eval_B0(ind, projects):
    """Single-objective: Z1 only, no synergy"""
    Z1 = 0.0
    for i, p in enumerate(projects):
        if ind['x'][i] != 1:
            continue
        Z1 += p['svs'] * (1.0 - p['risk'])
    return [Z1]


def eval_B1(ind, projects):
    """Multi-objective: Z1 + Z2, no synergy, fixed financing"""
    Z1, Z2 = 0.0, 0.0
    for i, p in enumerate(projects):
        if ind['x'][i] != 1:
            continue
        Z1 += p['svs'] * (1.0 - p['risk'])
        Z2 += p['fuzzy_cost'][1] * p['risk']
    return [Z1, Z2]


def eval_B2(ind, projects, delta):
    """Multi-objective: Z1 + Z2, WITH synergy, fixed financing"""
    Z1, Z2 = 0.0, 0.0
    n = len(projects)

    for i, p in enumerate(projects):
        if ind['x'][i] != 1:
            continue
        synergy_i = p['synergy_same'] + p['synergy_cross']
        Z1 += (p['svs'] + synergy_i) * (1.0 - p['risk'])
        Z2 += p['fuzzy_cost'][1] * p['risk']

    for i in range(n):
        if ind['x'][i] != 1:
            continue
        for j in range(i + 1, n):
            if ind['x'][j] != 1:
                continue
            Z1 += delta[i, j]

    return [Z1, Z2]


# -------------------------------------------------
# Main runner
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step1', required=True, help='Path to step1 results')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--pop', type=int, default=80)
    ap.add_argument('--gen', type=int, default=200)
    args = ap.parse_args()

    step1_dir = Path(args.step1).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    projects = json.loads((step1_dir / 'projects.json').read_text())
    delta = np.load(step1_dir / 'delta_matrix.npy')
    n = len(projects)

    results = {}

    for label, evaluator in [('B0', eval_B0), ('B1', eval_B1), ('B2', eval_B2)]:
        population, _, _ = ifpom.initialize_ifpom(args.pop, n)

        for ind in population:
            if label == 'B0':
                ind['Z'] = evaluator(ind, projects)
            elif label == 'B1':
                ind['Z'] = evaluator(ind, projects)
            else:
                ind['Z'] = evaluator(ind, projects, delta)

        results[label] = [ind['Z'] for ind in population]

    (outdir / 'ablation_results.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8'
    )

    print('=== Ablation Experiments Completed ===')
    for k, v in results.items():
        print(f"{k}: {len(v)} solutions")


if __name__ == '__main__':
    main()
