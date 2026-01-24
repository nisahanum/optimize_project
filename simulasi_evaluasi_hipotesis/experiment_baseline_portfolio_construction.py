# run_step1_prepare_inputs.py
# Step 1 Runner (Clean): Load + Validate + Prepare IFPOM Inputs
#
# What this does:
# 1) Load project dataset (original_projects.py)
# 2) Load synergy matrix delta_ij (CSV)
# 3) Validate consistency (dimensions, symmetry, diagonal, ranges)
# 4) Ensure required fields exist (risk, funding sums, theta cap check)
# 5) Export prepared artifacts for Step 2+ (JSON summary, projects JSON, delta NPY/CSV copy)
#
# Usage:
#   python run_step1_prepare_inputs.py --synergy synergy_matrix.csv --outdir results/step1
#
# Notes:
# - This runner does NOT import common_ifpom.py to avoid dependency on tchebycheff_utils.
# - It prepares inputs so your Step 2 (MOEA/D run) can be reproducible and reviewer-friendly.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import original_projects


# -----------------------------
# Risk utilities (lightweight)
# -----------------------------
def compute_technical_risk(trl: float, complexity: float) -> float:
    # TRL 1..9 -> higher TRL means lower risk
    # Normalize: (9-trl)/8 in [0,1]
    return ((9.0 - float(trl)) / 8.0) * float(complexity)


def compute_financial_risk(alpha: float, beta: float, theta: float, gamma: float, delta: float) -> float:
    # Simple monotonic scoring (can be aligned to your manuscript Table VIII)
    # Equity lowest; Vendor/PPP higher.
    return (
        alpha * 0.0 +
        beta * 0.3 +
        theta * 1.0 +
        gamma * 0.1 +
        delta * 0.6
    )


def ensure_project_risk(p: Dict[str, Any], w_tech: float = 0.6, w_fin: float = 0.4) -> float:
    # If risk already exists, keep it (do not overwrite).
    if "risk" in p and p["risk"] is not None:
        return float(p["risk"])

    trl = p.get("trl", 5)
    complexity = p.get("complexity", 0.5)

    alpha = float(p.get("alpha", 0.0))
    beta = float(p.get("beta", 0.0))
    theta = float(p.get("theta", 0.0))
    gamma = float(p.get("gamma", 0.0))
    delta = float(p.get("delta", 0.0))

    r_tech = compute_technical_risk(trl, complexity)
    r_fin = compute_financial_risk(alpha, beta, theta, gamma, delta)

    risk = w_tech * r_tech + w_fin * r_fin
    # Guardrail used in your code style
    risk = max(0.05, float(risk))
    p["risk_tech"] = float(r_tech)
    p["risk_fin"] = float(r_fin)
    p["risk"] = float(risk)
    return float(risk)


# -----------------------------
# Funding checks / repair notes
# -----------------------------
def funding_sum(p: Dict[str, Any]) -> float:
    return float(p.get("alpha", 0.0)) + float(p.get("beta", 0.0)) + float(p.get("theta", 0.0)) + float(p.get("gamma", 0.0)) + float(p.get("delta", 0.0))


def check_theta_cap(p: Dict[str, Any], cap: float = 0.4) -> bool:
    return float(p.get("theta", 0.0)) <= cap + 1e-9


# -----------------------------
# Synergy matrix loader (robust)
# -----------------------------
def load_synergy_matrix_csv(csv_path: Path) -> np.ndarray:
    # Expected: square matrix, numeric
    mat = np.loadtxt(csv_path, delimiter=",")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Synergy matrix must be square. Got shape={mat.shape}")
    return mat.astype(float)


@dataclass
class SynergyValidationReport:
    n: int
    symmetric: bool
    max_asymmetry: float
    diagonal_zero: bool
    diagonal_max_abs: float
    min_val: float
    max_val: float


def validate_synergy_matrix(delta: np.ndarray, tol: float = 1e-9) -> SynergyValidationReport:
    n = delta.shape[0]

    asym = np.abs(delta - delta.T)
    max_asym = float(np.max(asym))
    symmetric = bool(max_asym <= 1e-6)

    diag = np.diag(delta)
    diag_max_abs = float(np.max(np.abs(diag)))
    diagonal_zero = bool(diag_max_abs <= 1e-6)

    min_val = float(np.min(delta))
    max_val = float(np.max(delta))

    return SynergyValidationReport(
        n=n,
        symmetric=symmetric,
        max_asymmetry=max_asym,
        diagonal_zero=diagonal_zero,
        diagonal_max_abs=diag_max_abs,
        min_val=min_val,
        max_val=max_val,
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synergy", type=str, required=True, help="Path to synergy_matrix.csv (delta_ij)")
    ap.add_argument("--outdir", type=str, default="results/step1", help="Output directory for prepared artifacts")
    ap.add_argument("--w_tech", type=float, default=0.6, help="Technical risk weight (baseline)")
    ap.add_argument("--w_fin", type=float, default=0.4, help="Financial risk weight (baseline)")
    ap.add_argument("--theta_cap", type=float, default=0.4, help="Vendor financing cap (theta)")
    args = ap.parse_args()

    synergy_path = Path(args.synergy).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load projects
    projects: List[Dict[str, Any]] = original_projects.load_project_data()
    n_projects = len(projects)

    # 2) Load synergy matrix
    delta = load_synergy_matrix_csv(synergy_path)

    # 3) Validate dimension match
    if delta.shape[0] != n_projects:
        raise ValueError(
            f"Mismatch: projects={n_projects} but synergy matrix size={delta.shape[0]}. "
            f"Fix by using the correct synergy_matrix.csv aligned to the project ordering."
        )

    # 4) Validate synergy matrix properties
    rep = validate_synergy_matrix(delta)

    # 5) Ensure risk exists; funding sanity checks
    funding_issues = []
    theta_issues = []
    for p in projects:
        ensure_project_risk(p, w_tech=args.w_tech, w_fin=args.w_fin)

        s = funding_sum(p)
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-6):
            funding_issues.append({"id": p.get("id"), "funding_sum": s})

        if not check_theta_cap(p, cap=args.theta_cap):
            theta_issues.append({"id": p.get("id"), "theta": float(p.get("theta", 0.0)), "cap": args.theta_cap})

    # 6) Export artifacts
    # Save delta in .npy (fast) and also copy CSV for traceability
    np.save(outdir / "delta_matrix.npy", delta)
    (outdir / "synergy_matrix.csv").write_text(synergy_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Save projects as JSON (so Step 2 runner can be deterministic)
    with (outdir / "projects.json").open("w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)

    # Save summary for paper appendix / reproducibility note
    summary = {
        "n_projects": n_projects,
        "project_ids": [p.get("id") for p in projects],
        "risk_baseline": {"w_tech": args.w_tech, "w_fin": args.w_fin},
        "theta_cap": args.theta_cap,
        "synergy_validation": {
            "n": rep.n,
            "symmetric": rep.symmetric,
            "max_asymmetry": rep.max_asymmetry,
            "diagonal_zero": rep.diagonal_zero,
            "diagonal_max_abs": rep.diagonal_max_abs,
            "min_val": rep.min_val,
            "max_val": rep.max_val,
        },
        "funding_sum_issues": funding_issues,
        "theta_cap_issues": theta_issues,
        "artifacts": {
            "projects_json": str((outdir / "projects.json").as_posix()),
            "delta_npy": str((outdir / "delta_matrix.npy").as_posix()),
            "delta_csv_copy": str((outdir / "synergy_matrix.csv").as_posix()),
        },
    }

    with (outdir / "step1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 7) Console print (short, reviewer-friendly)
    print("=== Step 1 Completed: Inputs Prepared ===")
    print(f"Projects: {n_projects}")
    print(f"Synergy matrix: {delta.shape} | symmetric={rep.symmetric} | diag_zero={rep.diagonal_zero} | range=[{rep.min_val:.4f}, {rep.max_val:.4f}]")
    print(f"Risk baseline: w_tech={args.w_tech}, w_fin={args.w_fin}")
    if funding_issues:
        print(f"[WARN] Funding sums != 1.0 for {len(funding_issues)} project(s). See step1_summary.json")
    if theta_issues:
        print(f"[WARN] Theta cap violated for {len(theta_issues)} project(s). See step1_summary.json")
    print(f"Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
