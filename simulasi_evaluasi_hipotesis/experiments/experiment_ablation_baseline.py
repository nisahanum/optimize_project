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
from datetime import datetime

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
        if ind["x"][i] != 1:
            continue
        Z1 += float(p["svs"]) * (1.0 - float(p["risk"]))
    return [float(Z1)]


def eval_B1(ind, projects):
    """Multi-objective: Z1 + Z2, no synergy, fixed financing"""
    Z1, Z2 = 0.0, 0.0
    for i, p in enumerate(projects):
        if ind["x"][i] != 1:
            continue
        Z1 += float(p["svs"]) * (1.0 - float(p["risk"]))
        Z2 += float(p["fuzzy_cost"][1]) * float(p["risk"])
    return [float(Z1), float(Z2)]


def eval_B2(ind, projects, delta):
    """Multi-objective: Z1 + Z2, WITH synergy, fixed financing"""
    Z1, Z2 = 0.0, 0.0
    n = len(projects)

    for i, p in enumerate(projects):
        if ind["x"][i] != 1:
            continue
        synergy_i = float(p.get("synergy_same", 0.0)) + float(p.get("synergy_cross", 0.0))
        Z1 += (float(p["svs"]) + synergy_i) * (1.0 - float(p["risk"]))
        Z2 += float(p["fuzzy_cost"][1]) * float(p["risk"])

    # pairwise synergy (delta_ij)
    for i in range(n):
        if ind["x"][i] != 1:
            continue
        for j in range(i + 1, n):
            if ind["x"][j] != 1:
                continue
            Z1 += float(delta[i, j])

    return [float(Z1), float(Z2)]


def _summary_stats(pop_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Minimal ringkas untuk tabel/visualisasi."""
    Z = [it["Z"] for it in pop_items if it.get("Z") is not None]
    if not Z:
        return {"n": 0}

    dim = len(Z[0])
    Z_arr = np.array(Z, dtype=float)

    out: Dict[str, Any] = {"n": int(len(Z))}
    # Z1 (maximize)
    out["best_Z1"] = float(np.max(Z_arr[:, 0]))
    out["mean_Z1"] = float(np.mean(Z_arr[:, 0]))

    if dim >= 2:
        # Z2 (minimize)
        out["best_Z2"] = float(np.min(Z_arr[:, 1]))
        out["mean_Z2"] = float(np.mean(Z_arr[:, 1]))

    return out


# -------------------------------------------------
# Main runner
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", required=True, help="Path to step1 results")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=0, help="Not used (kept for CLI compatibility)")
    args = ap.parse_args()

    step1_dir = Path(args.step1).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    projects = json.loads((step1_dir / "projects.json").read_text(encoding="utf-8"))
    delta = np.load(step1_dir / "delta_matrix.npy")
    n = len(projects)

    # -----------------------------
    # Run baselines
    # -----------------------------
    models_payload: Dict[str, Any] = {}

    for label, evaluator in [("B0", eval_B0), ("B1", eval_B1), ("B2", eval_B2)]:
        # NOTE: baseline ini tidak menjalankan optimisasi (hanya sampling populasi awal)
        population, _, _ = ifpom.initialize_ifpom(args.pop, n)

        pop_items: List[Dict[str, Any]] = []
        for ind in population:
            if label == "B0":
                Z = evaluator(ind, projects)
            elif label == "B1":
                Z = evaluator(ind, projects)
            else:
                Z = evaluator(ind, projects, delta)

            ind["Z"] = Z
            pop_items.append({"x": ind["x"], "Z": Z})

        if label == "B0":
            objectives = ["Z1"]
            directions = ["max"]
        else:
            objectives = ["Z1", "Z2"]
            directions = ["max", "min"]

        models_payload[label] = {
            "objectives": objectives,
            "directions": directions,
            "population": pop_items,
            "summary": _summary_stats(pop_items),
        }

    # -----------------------------
    # OUTPUT WRAPPER (FIX UTAMA)
    # -----------------------------
    payload: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": "experiment_ablation_baseline.py",
            "step1_dir": str(step1_dir),
            "outdir": str(outdir),
            "n_projects": int(n),
            "pop_size": int(args.pop),
            "note": "Baselines are sampling-based (no optimization) to isolate missing modeling components.",
        },
        "models": models_payload,
    }

    (outdir / "ablation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # also export per-model json to avoid splitter dependency
    by_model_dir = outdir / "by_model"
    by_model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, obj in payload["models"].items():
        (by_model_dir / f"{model_name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("=== Ablation Experiments Completed ===")
    print(f"Saved: {outdir / 'ablation_results.json'}")
    print(f"Per-model JSON: {by_model_dir}")
    for k, v in payload["models"].items():
        print(f"{k}: n={v['summary'].get('n', 0)} | summary={v['summary']}")


if __name__ == "__main__":
    main()
