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
# Usage (simple):
#   python run_step1_prepare_inputs.py
#
# Usage (explicit):
#   python run_step1_prepare_inputs.py --synergy cosine
#   python run_step1_prepare_inputs.py --synergy cleaning_data/synergy_matrix.csv --outdir results/step1
#
# Notes:
# - This runner does NOT import common_ifpom.py.
# - It prepares inputs so your Step 2 (MOEA/D run) can be reproducible and reviewer-friendly.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import original_projects

# Optional: read defaults from config.py if available
try:
    import config  # type: ignore
except Exception:
    config = None  # noqa: N816


# -----------------------------
# Risk utilities (lightweight)
# -----------------------------
def compute_technical_risk(trl: float, complexity: float) -> float:
    # TRL 1..9 -> higher TRL means lower risk
    return ((9.0 - float(trl)) / 8.0) * float(complexity)


def compute_financial_risk(alpha: float, beta: float, theta: float, gamma: float, delta: float) -> float:
    # Simple monotonic scoring (align later to manuscript if needed)
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
    risk = max(0.05, float(risk))  # guardrail
    p["risk_tech"] = float(r_tech)
    p["risk_fin"] = float(r_fin)
    p["risk"] = float(risk)
    return float(risk)


# -----------------------------
# Funding checks
# -----------------------------
def funding_sum(p: Dict[str, Any]) -> float:
    return (
        float(p.get("alpha", 0.0)) +
        float(p.get("beta", 0.0)) +
        float(p.get("theta", 0.0)) +
        float(p.get("gamma", 0.0)) +
        float(p.get("delta", 0.0))
    )


def check_theta_cap(p: Dict[str, Any], cap: float = 0.4) -> bool:
    return float(p.get("theta", 0.0)) <= cap + 1e-9


# -----------------------------
# Synergy matrix loader (robust)
# -----------------------------
def _resolve_path(candidate: str, base_dir: Path) -> Path:
    p = Path(candidate).expanduser()
    if p.is_absolute() and p.exists():
        return p
    # try relative to current working dir
    if p.exists():
        return p.resolve()
    # try relative to script directory
    p2 = (base_dir / p).resolve()
    if p2.exists():
        return p2
    # try project root = parent of script dir (common layout)
    p3 = (base_dir.parent / p).resolve()
    if p3.exists():
        return p3
    return p.resolve()


def resolve_synergy_arg(synergy_arg: Optional[str], base_dir: Path) -> Path:
    """
    Accepts:
    - a CSV path (absolute/relative)
    - a label: 'cosine' or 'psplib' (mapped to known default paths if available)
    - None: use config.DEFAULT_SYNERGY_PATH if available, else raise.
    """
    # 1) If user passes nothing: try config default
    if not synergy_arg:
        if config is not None and hasattr(config, "DEFAULT_SYNERGY_PATH"):
            return _resolve_path(str(getattr(config, "DEFAULT_SYNERGY_PATH")), base_dir)
        raise ValueError("Synergy is not provided and config.DEFAULT_SYNERGY_PATH is missing. Provide --synergy <path|label>.")

    # 2) If user passes label
    label = synergy_arg.strip().lower()
    label_map = {}

    # prefer config-based mapping if present
    if config is not None:
        if hasattr(config, "COSINE_SYNERGY_PATH"):
            label_map["cosine"] = str(getattr(config, "COSINE_SYNERGY_PATH"))
        if hasattr(config, "PSPLIB_SYNERGY_PATH"):
            label_map["psplib"] = str(getattr(config, "PSPLIB_SYNERGY_PATH"))

    # safe fallbacks (edit if your repo uses different locations)
    label_map.setdefault("cosine", "cleaning_data/synergy_matrix.csv")
    label_map.setdefault("psplib", "cleaning_data/synergy_matrix_psplib.csv")

    if label in label_map:
        resolved = _resolve_path(label_map[label], base_dir)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Synergy label '{label}' mapped to '{resolved}', but file not found. "
                f"Fix config path or pass an explicit CSV path via --synergy <csv>."
            )
        return resolved

    # 3) Otherwise treat as path
    resolved = _resolve_path(synergy_arg, base_dir)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Synergy file not found: '{synergy_arg}'. "
            f"Pass a valid CSV path or use label 'cosine'/'psplib'."
        )
    return resolved


def load_synergy_matrix_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)  # kolom pertama = label baris
    mat = df.to_numpy(dtype=float)
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


def validate_synergy_matrix(delta: np.ndarray) -> SynergyValidationReport:
    n = delta.shape[0]
    asym = np.abs(delta - delta.T)
    max_asym = float(np.max(asym))
    symmetric = bool(max_asym <= 1e-6)

    diag = np.diag(delta)
    diag_max_abs = float(np.max(np.abs(diag)))
    diagonal_zero = bool(diag_max_abs <= 1e-6)

    return SynergyValidationReport(
        n=n,
        symmetric=symmetric,
        max_asymmetry=max_asym,
        diagonal_zero=diagonal_zero,
        diagonal_max_abs=diag_max_abs,
        min_val=float(np.min(delta)),
        max_val=float(np.max(delta)),
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synergy",
        type=str,
        default=None,
        help="Synergy source: CSV path OR label {cosine, psplib}. If omitted, uses config.DEFAULT_SYNERGY_PATH when available.",
    )
    ap.add_argument("--outdir", type=str, default=None, help="Output directory for prepared artifacts")
    ap.add_argument("--w_tech", type=float, default=None, help="Technical risk weight (baseline)")
    ap.add_argument("--w_fin", type=float, default=None, help="Financial risk weight (baseline)")
    ap.add_argument("--theta_cap", type=float, default=None, help="Vendor financing cap (theta)")
    args = ap.parse_args()

    # Defaults (prefer config.py if available)
    w_tech = float(args.w_tech) if args.w_tech is not None else float(getattr(config, "W_TECH", 0.6) if config else 0.6)
    w_fin = float(args.w_fin) if args.w_fin is not None else float(getattr(config, "W_FIN", 0.4) if config else 0.4)
    theta_cap = float(args.theta_cap) if args.theta_cap is not None else float(getattr(config, "THETA_CAP", 0.4) if config else 0.4)

    outdir_default = getattr(config, "DEFAULT_OUTDIR", "results/step1") if config else "results/step1"
    outdir = _resolve_path(args.outdir or outdir_default, base_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load projects (uses your current function name)
    projects: List[Dict[str, Any]] = original_projects.load_project_data()
    n_projects = len(projects)

    # 2) Resolve and load synergy
    synergy_path = resolve_synergy_arg(args.synergy, base_dir)
    delta = load_synergy_matrix_csv(synergy_path)

    # 3) Validate dimension match
    if delta.shape[0] != n_projects:
        raise ValueError(
            f"Mismatch: projects={n_projects} but synergy matrix size={delta.shape[0]}. "
            f"Use a synergy_matrix.csv aligned to the project ordering."
        )

    # 4) Validate synergy matrix properties
    rep = validate_synergy_matrix(delta)

    # 5) Ensure risk exists; funding sanity checks
    funding_issues = []
    theta_issues = []
    for p in projects:
        ensure_project_risk(p, w_tech=w_tech, w_fin=w_fin)

        s = funding_sum(p)
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-6):
            funding_issues.append({"id": p.get("id"), "funding_sum": s})

        if not check_theta_cap(p, cap=theta_cap):
            theta_issues.append({"id": p.get("id"), "theta": float(p.get("theta", 0.0)), "cap": theta_cap})

    # 6) Export artifacts
    np.save(outdir / "delta_matrix.npy", delta)
    (outdir / "synergy_matrix.csv").write_text(synergy_path.read_text(encoding="utf-8"), encoding="utf-8")

    with (outdir / "projects.json").open("w", encoding="utf-8") as f:
        json.dump(projects, f, ensure_ascii=False, indent=2)

    summary = {
        "n_projects": n_projects,
        "project_ids": [p.get("id") for p in projects],
        "risk_baseline": {"w_tech": w_tech, "w_fin": w_fin},
        "theta_cap": theta_cap,
        "synergy_source": str(synergy_path.as_posix()),
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

    # 7) Console print (short)
    print("=== Step 1 Completed: Inputs Prepared ===")
    print(f"Projects: {n_projects}")
    print(f"Synergy source: {synergy_path}")
    print(
        f"Synergy matrix: {delta.shape} | symmetric={rep.symmetric} | diag_zero={rep.diagonal_zero} "
        f"| range=[{rep.min_val:.4f}, {rep.max_val:.4f}]"
    )
    print(f"Risk baseline: w_tech={w_tech}, w_fin={w_fin} | theta_cap={theta_cap}")
    if funding_issues:
        print(f"[WARN] Funding sums != 1.0 for {len(funding_issues)} project(s). See step1_summary.json")
    if theta_issues:
        print(f"[WARN] Theta cap violated for {len(theta_issues)} project(s). See step1_summary.json")
    print(f"Artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
