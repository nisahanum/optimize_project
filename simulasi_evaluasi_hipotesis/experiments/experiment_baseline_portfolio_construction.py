# experiment_baseline_portfolio_construction.py
# Step 1 (Baseline Portfolio Construction): Load + Validate + Prepare IFPOM Inputs
#
# What this does:
# 1) Load project dataset from original_projects.py
# 2) Load synergy matrix delta_ij from a CSV (numeric square matrix)
# 3) Validate consistency (dimension match, symmetry, diagonal, numeric range)
# 4) Ensure required computed fields exist (risk_tech, risk_fin, risk) if missing
# 5) Check funding sums and vendor cap (theta_cap) as diagnostics (no forced repair)
# 6) Export prepared artifacts for Step 2 (MOEA/D runner):
#    - projects.json
#    - delta_matrix.npy
#    - synergy_matrix_copy.csv
#    - step1_summary.json
#
# Usage (Mac/Linux):
#   python experiment_baseline_portfolio_construction.py \
#     --synergy /full/path/synergy_matrix_cosine_normalized.csv \
#     --outdir results/step1_cosine \
#     --w_tech 0.6 --w_fin 0.4 --theta_cap 0.4
#
# Usage (Windows PowerShell one-line):
#   python experiment_baseline_portfolio_construction.py --synergy "C:\path\synergy_matrix.csv" --outdir "results\step1_cosine" --w_tech 0.6 --w_fin 0.4 --theta_cap 0.4
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # simulasi_evaluasi_hipotesis
sys.path.insert(0, str(ROOT))

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

import original_projects


# -----------------------------
# Risk utilities (lightweight)
# -----------------------------
def compute_technical_risk(trl: float, complexity: float) -> float:
    """
    Technical risk based on TRL (1..9) and complexity (0..1).
    Higher TRL => lower risk. We use normalized (9-TRL)/8 in [0,1].
    """
    trl_f = float(trl)
    comp_f = float(complexity)
    trl_norm = (9.0 - trl_f) / 8.0  # TRL=9 -> 0, TRL=1 -> 1
    return max(0.0, trl_norm) * max(0.0, comp_f)


def compute_financial_risk(alpha: float, beta: float, theta: float, gamma: float, delta: float) -> float:
    """
    Simple monotonic scoring for financial risk from funding mix.
    This is a *scoring function* (not a market-estimated probability).
    You can align coefficients to your manuscript assumptions.
    """
    a = float(alpha)
    b = float(beta)
    t = float(theta)
    g = float(gamma)
    d = float(delta)
    return (
        a * 0.0 +   # internal equity: lowest
        b * 0.3 +   # soft loan: moderate
        t * 1.0 +   # vendor financing: highest
        g * 0.1 +   # grants: low-to-moderate overhead
        d * 0.6     # PPP/JV: higher coordination/contract risk
    )


def ensure_project_risk(p: Dict[str, Any], w_tech: float, w_fin: float) -> None:
    """
    Ensure p contains:
      - risk_tech
      - risk_fin
      - risk (overall)
    If p already has 'risk', we keep it and only fill missing subfields if needed.
    """
    # Extract base fields with sensible defaults
    trl = float(p.get("trl", 5))
    complexity = float(p.get("complexity", 0.5))

    alpha = float(p.get("alpha", 0.0))
    beta = float(p.get("beta", 0.0))
    theta = float(p.get("theta", 0.0))
    gamma = float(p.get("gamma", 0.0))
    delta = float(p.get("delta", 0.0))

    r_tech = compute_technical_risk(trl, complexity)
    r_fin = compute_financial_risk(alpha, beta, theta, gamma, delta)

    # Store components
    p["risk_tech"] = float(r_tech)
    p["risk_fin"] = float(r_fin)

    # If overall risk exists, do not overwrite (but ensure it's float)
    if p.get("risk", None) is not None:
        p["risk"] = float(p["risk"])
        return

    # Baseline composition (convex combination)
    risk = (float(w_tech) * r_tech) + (float(w_fin) * r_fin)

    # Guardrail consistent with your earlier implementation
    p["risk"] = float(max(0.05, risk))


# Repair Funding
def repair_funding_to_alpha(p: dict) -> None:
    """
    Repair funding so that α absorbs residual and sum == 1.0
    This keeps funding realistic and reviewer-safe.
    """
    alpha = float(p.get("alpha", 0.0))
    beta  = float(p.get("beta", 0.0))
    theta = float(p.get("theta", 0.0))
    gamma = float(p.get("gamma", 0.0))
    delta = float(p.get("delta", 0.0))

    total = alpha + beta + theta + gamma + delta
    if abs(total - 1.0) > 1e-6:
        p["alpha"] = max(0.0, 1.0 - (beta + theta + gamma + delta))

# -----------------------------
# Funding checks (diagnostics)
# -----------------------------
def funding_sum(p: Dict[str, Any]) -> float:
    return (
        float(p.get("alpha", 0.0)) +
        float(p.get("beta", 0.0)) +
        float(p.get("theta", 0.0)) +
        float(p.get("gamma", 0.0)) +
        float(p.get("delta", 0.0))
    )


def check_theta_cap(p: Dict[str, Any], cap: float) -> bool:
    return float(p.get("theta", 0.0)) <= float(cap) + 1e-9


# -----------------------------
# Synergy matrix loader/validator
# -----------------------------
def load_synergy_matrix_csv(csv_path: Path, project_ids: List[Any]) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0)

    if df.isna().any().any():
        raise ValueError("Synergy matrix contains NaN (likely empty cell).")

    # pastikan label kolom & index ada
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    pid = [str(x) for x in project_ids]

    missing_rows = [x for x in pid if x not in df.index]
    missing_cols = [x for x in pid if x not in df.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"Synergy CSV labels do not match project ids. "
            f"Missing rows={missing_rows[:5]} cols={missing_cols[:5]} (showing up to 5)."
        )

    # REINDEX sesuai urutan proyek
    df = df.loc[pid, pid]

    mat = df.values.astype(float)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Synergy matrix must be square, got {mat.shape}")

    return mat



@dataclass
class SynergyValidationReport:
    n: int
    symmetric: bool
    max_asymmetry: float
    diagonal_zero: bool
    diagonal_max_abs: float
    min_val: float
    max_val: float


def validate_synergy_matrix(delta: np.ndarray, tol: float = 1e-6) -> SynergyValidationReport:
    n = int(delta.shape[0])

    asym = np.abs(delta - delta.T)
    max_asym = float(np.max(asym))
    symmetric = bool(max_asym <= tol)

    diag = np.diag(delta)
    diag_max_abs = float(np.max(np.abs(diag)))
    diagonal_zero = bool(diag_max_abs <= tol)

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
    ap = argparse.ArgumentParser(description="Step 1: Baseline portfolio input construction for IFPOM experiments.")
    ap.add_argument("--synergy", type=str, required=True, help="Path to synergy matrix CSV (delta_ij).")
    ap.add_argument("--outdir", type=str, default="results/step1", help="Output directory for Step 1 artifacts.")
    ap.add_argument("--w_tech", type=float, default=0.6, help="Baseline weight for technical risk.")
    ap.add_argument("--w_fin", type=float, default=0.4, help="Baseline weight for financial risk.")
    ap.add_argument("--theta_cap", type=float, default=0.4, help="Vendor financing cap (theta).")
    ap.add_argument("--repair_funding", action="store_true",
               help="If set, repair funding so alpha absorbs residual and sum==1. Default: diagnostics only.")
    args = ap.parse_args()

    synergy_path = Path(args.synergy).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load projects (must exist in original_projects.py)
    # Expected: original_projects.load_project_data() returns List[Dict]
    projects: List[Dict[str, Any]] = original_projects.load_project_data()
    n_projects = len(projects)

    # 2) Load synergy matrix
    project_ids = [p.get("id") for p in projects]
    delta = load_synergy_matrix_csv(synergy_path, project_ids=project_ids)
    np.fill_diagonal(delta, 0.0)

    # 3) Dimension match check
    if delta.shape[0] != n_projects:
        raise ValueError(
            f"Mismatch: projects={n_projects} but synergy matrix size={delta.shape[0]}. "
            "Ensure the synergy matrix row/column order aligns to your project list ordering."
        )

    # 4) Validate synergy properties
    rep = validate_synergy_matrix(delta)

    # 5) Ensure risk exists; record funding / theta diagnostics
    funding_issues: List[Dict[str, Any]] = []
    theta_issues: List[Dict[str, Any]] = []

    for p in projects:
        if args.repair_funding:
            repair_funding_to_alpha(p)   # atau renormalize_funding(p)
        ensure_project_risk(p, w_tech=args.w_tech, w_fin=args.w_fin)

        s = funding_sum(p)
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-6):
            funding_issues.append({"id": p.get("id"), "funding_sum": float(s)})

        if not check_theta_cap(p, cap=args.theta_cap):
            theta_issues.append({"id": p.get("id"), "theta": float(p.get("theta", 0.0)), "cap": float(args.theta_cap)})

    # 6) Export artifacts
    # delta
    np.save(outdir / "delta_matrix.npy", delta)
    # copy CSV for traceability
    (outdir / "synergy_matrix_copy.csv").write_text(synergy_path.read_text(encoding="utf-8"), encoding="utf-8")
    # projects
    (outdir / "projects.json").write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "n_projects": n_projects,
        "project_ids": [p.get("id") for p in projects],
        "risk_baseline": {"w_tech": float(args.w_tech), "w_fin": float(args.w_fin)},
        "theta_cap": float(args.theta_cap),
        "synergy_source": str(synergy_path),
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
        "repair_funding_enabled": bool(args.repair_funding),
        "theta_cap_issues": theta_issues,
        "artifacts": {
            "projects_json": str((outdir / "projects.json").as_posix()),
            "delta_npy": str((outdir / "delta_matrix.npy").as_posix()),
            "synergy_csv_copy": str((outdir / "synergy_matrix_copy.csv").as_posix()),
        },
    }
    (outdir / "step1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7) Console output (short, reviewer-friendly)
    print("=== Step 1 Completed: Inputs Prepared ===")
    print(f"Projects: {n_projects}")
    print(f"Synergy source: {synergy_path}")
    print(
        f"Synergy matrix: {delta.shape} | symmetric={rep.symmetric} | diag_zero={rep.diagonal_zero} "
        f"| range=[{rep.min_val:.4f}, {rep.max_val:.4f}]"
    )
    print(f"Risk baseline: w_tech={args.w_tech}, w_fin={args.w_fin} | theta_cap={args.theta_cap}")
    if funding_issues:
        print(f"[WARN] Funding sums != 1.0 for {len(funding_issues)} project(s). See step1_summary.json")
    if theta_issues:
        print(f"[WARN] Theta cap violated for {len(theta_issues)} project(s). See step1_summary.json")
    print(f"Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
