from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pareto_mask(Z: np.ndarray, directions: List[str]) -> np.ndarray:
    """
    Return boolean mask for non-dominated points.
    directions: list of "max" or "min" corresponding to columns of Z.
    """
    assert Z.ndim == 2
    n, _ = Z.shape

    # Convert to minimization for dominance checks
    Zm = Z.astype(float).copy()
    for j, d in enumerate(directions):
        if d == "max":
            Zm[:, j] = -Zm[:, j]

    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_nd[i]:
            continue
        for k in range(n):
            if k == i:
                continue
            # k dominates i in minimization space
            if np.all(Zm[k] <= Zm[i]) and np.any(Zm[k] < Zm[i]):
                is_nd[i] = False
                break
    return is_nd


def extract_Z(population: List[Dict[str, Any]]) -> np.ndarray:
    return np.array([ind["Z"] for ind in population], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", type=str, default="B0.json")
    ap.add_argument("--b1", type=str, default="B1.json")
    ap.add_argument("--b2", type=str, default="B2.json")
    ap.add_argument("--full", type=str, default="Full_IFPOM.json")
    ap.add_argument("--out", type=str, default="fig_pareto_comparison.png")
    ap.add_argument("--dpi", type=int, default=250)
    args = ap.parse_args()

    # --- Debug prints (biar tidak terasa "nothing happened")
    print("[INFO] Running Pareto comparison plot...")
    print("[INFO] Output:", args.out)

    B0 = load_json(Path(args.b0))
    B1 = load_json(Path(args.b1))
    B2 = load_json(Path(args.b2))
    FULL = load_json(Path(args.full))

    # --- Extract objectives
    Z0 = extract_Z(B0["population"])          # (n,1)
    Z1 = extract_Z(B1["population"])          # (n,2)
    Z2 = extract_Z(B2["population"])          # (n,2)
    ZF = extract_Z(FULL["population"])        # (n,3)

    m0 = pareto_mask(Z0, B0["directions"])
    m1 = pareto_mask(Z1, B1["directions"])
    m2 = pareto_mask(Z2, B2["directions"])
    mf = pareto_mask(ZF, FULL["directions"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    # --- Panel A: B0 (Z1 only) -> plot vs index
    idx0 = np.arange(len(Z0))
    ax00.scatter(idx0[~m0], Z0[~m0, 0], marker="o", label="Population")
    ax00.scatter(idx0[m0],  Z0[m0, 0],  marker="s", label="Pareto (best)")
    ax00.set_title("B0: Single-objective (Z1)")
    ax00.set_xlabel("Solution index")
    ax00.set_ylabel("Z1 (higher is better)")
    ax00.legend()

    # --- Panel B: B1 (Z1 vs Z2)
    ax01.scatter(Z1[~m1, 0], Z1[~m1, 1], marker="o", label="Population")
    ax01.scatter(Z1[m1, 0],  Z1[m1, 1],  marker="s", label="Pareto front")
    ax01.set_title("B1: Bi-objective (Z1–Z2), no synergy")
    ax01.set_xlabel("Z1 (↑)")
    ax01.set_ylabel("Z2 (↓)")
    ax01.legend()

    # --- Panel C: B2 (Z1 vs Z2)
    ax10.scatter(Z2[~m2, 0], Z2[~m2, 1], marker="o", label="Population")
    ax10.scatter(Z2[m2, 0],  Z2[m2, 1],  marker="s", label="Pareto front")
    ax10.set_title("B2: Bi-objective (Z1–Z2), partial systemic effect")
    ax10.set_xlabel("Z1 (↑)")
    ax10.set_ylabel("Z2 (↓)")
    ax10.legend()

    # --- Panel D: Full IFPOM (Z1 vs Z2) + Pareto
    ax11.scatter(ZF[~mf, 0], ZF[~mf, 1], marker="o", label="Population")
    ax11.scatter(ZF[mf, 0],  ZF[mf, 1],  marker="s", label="Pareto set")
    ax11.set_title("Full IFPOM: Tri-objective (Z1–Z2–Z3)")
    ax11.set_xlabel("Z1 (↑)")
    ax11.set_ylabel("Z2 (↓)")
    ax11.legend()

    fig.tight_layout()

    # --- Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=args.dpi)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
