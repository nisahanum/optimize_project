# experiment_plot_pareto_front.py
# Step 3: Plot 3D Pareto Front (Z1–Z2–Z3) from Step 2 MOEA/D outputs
#
# Usage:
#   python experiment_plot_pareto_front.py --step2 ../results/step2_moead_cosine --outdir ../results/step2_moead_cosine
#
# Output:
#   pareto_front_3d.png
#   pareto_points.csv
#   pareto_points.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_population(step2_dir: Path) -> List[Dict[str, Any]]:
    pop_path = step2_dir / "final_population.json"
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing: {pop_path}")
    return json.loads(pop_path.read_text(encoding="utf-8"))


def extract_points(population: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    """Return Z matrix (N x 3) and indices kept (valid Z only)."""
    Z_list = []
    keep_idx = []
    for i, ind in enumerate(population):
        Z = ind.get("Z", None)
        if Z is None or len(Z) != 3:
            continue
        # ensure numeric
        try:
            z1, z2, z3 = float(Z[0]), float(Z[1]), float(Z[2])
        except Exception:
            continue
        Z_list.append([z1, z2, z3])
        keep_idx.append(i)
    if not Z_list:
        raise ValueError("No valid Z points found in final_population.json")
    return np.array(Z_list, dtype=float), keep_idx


def pareto_mask_maxmin(Z: np.ndarray) -> np.ndarray:
    """
    Pareto for mixed directions:
      maximize Z1, minimize Z2, maximize Z3
    Non-dominated points are kept.

    A dominates B if:
      Z1_A >= Z1_B, Z2_A <= Z2_B, Z3_A >= Z3_B
      and at least one strict.
    """
    n = Z.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[i]:
                continue
            A = Z[j]
            B = Z[i]
            dominates = (A[0] >= B[0]) and (A[1] <= B[1]) and (A[2] >= B[2]) and (
                (A[0] > B[0]) or (A[1] < B[1]) or (A[2] > B[2])
            )
            if dominates:
                keep[i] = False
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2", type=str, required=True, help="Path to Step2 output dir (contains final_population.json)")
    ap.add_argument("--outdir", type=str, default=None, help="Where to save plots (default = step2 dir)")
    ap.add_argument("--all_points", action="store_true", help="Plot all points (skip Pareto filtering)")
    args = ap.parse_args()

    step2_dir = Path(args.step2).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else step2_dir
    outdir.mkdir(parents=True, exist_ok=True)

    population = load_population(step2_dir)
    Z, keep_idx = extract_points(population)

    if args.all_points:
        Z_plot = Z
        tag = "all"
        mask = np.ones(Z.shape[0], dtype=bool)
    else:
        mask = pareto_mask_maxmin(Z)
        Z_plot = Z[mask]
        tag = "pareto"

    # Save points (csv + json)
    csv_path = outdir / f"{tag}_points.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("Z1,Z2,Z3\n")
        for z1, z2, z3 in Z_plot:
            f.write(f"{z1},{z2},{z3}\n")

    json_path = outdir / f"{tag}_points.json"
    json_path.write_text(json.dumps(Z_plot.tolist(), indent=2), encoding="utf-8")

    # 3D plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Z_plot[:, 0], Z_plot[:, 1], Z_plot[:, 2], marker="o")

    ax.set_xlabel("Z1 (Strategic Value) ↑")
    ax.set_ylabel("Z2 (Risk-Adjusted Cost) ↓")
    ax.set_zlabel("Z3 (Synergy Value) ↑")

    title = "Pareto Front (Z1–Z2–Z3)" if not args.all_points else "All Solutions (Z1–Z2–Z3)"
    ax.set_title(title)

    out_png = outdir / ("pareto_front_3d.png" if not args.all_points else "all_solutions_3d.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("=== Step 3 Completed: Pareto Figure Saved ===")
    print(f"Points: {Z_plot.shape[0]} ({'all' if args.all_points else 'pareto'})")
    print(f"Saved: {out_png}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
