from __future__ import annotations  # HARUS baris pertama setelah komentar/docstring

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np

# --- Make root importable: .../optimize_project
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import config  # default parameters
import common_ifpom_final as ifpom


# -----------------------------
# Step 1 loader
# -----------------------------
def load_step1(step1_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    projects_path = step1_dir / "projects.json"
    delta_path = step1_dir / "delta_matrix.npy"
    summary_path = step1_dir / "step1_summary.json"

    projects = json.loads(projects_path.read_text(encoding="utf-8"))
    delta = np.load(delta_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return projects, delta, summary


# -----------------------------
# Pareto utilities (Z1 max, Z2 min, Z3 max)
# -----------------------------
def dominates(a: List[float], b: List[float]) -> bool:
    """
    Return True if a dominates b under mixed directions:
      - maximize Z1, Z3
      - minimize Z2
    a dominates b if:
      a is no worse than b on all objectives, and strictly better in at least one.
    """
    if a is None or b is None:
        return False

    no_worse = (a[0] >= b[0]) and (a[1] <= b[1]) and (a[2] >= b[2])
    strictly_better = (a[0] > b[0]) or (a[1] < b[1]) or (a[2] > b[2])
    return no_worse and strictly_better


def extract_pareto_front(population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract non-dominated individuals from population.
    Keeps the full individual dict (x + funding + Z) for traceability.
    """
    pareto: List[Dict[str, Any]] = []
    for i, ind in enumerate(population):
        Zi = ind.get("Z")
        if Zi is None:
            continue
        dominated_flag = False
        for j, other in enumerate(population):
            if i == j:
                continue
            Zj = other.get("Z")
            if Zj is None:
                continue
            if dominates(Zj, Zi):  # other dominates current
                dominated_flag = True
                break
        if not dominated_flag:
            pareto.append(ind)
    return pareto


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=None, help="Population size (default from config)")
    ap.add_argument("--gen", type=int, default=None, help="Generations (default from config)")
    ap.add_argument("--neighbors", type=int, default=None, help="Neighborhood size T (default from config)")
    ap.add_argument("--theta_cap", type=float, default=None, help="Theta cap (default from config)")
    ap.add_argument("--step1", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # ---- defaults from config, overridable by CLI
    pop_size = args.pop if args.pop is not None else config.POPULATION_SIZE
    max_gen = args.gen if args.gen is not None else config.NUM_GENERATIONS
    T = args.neighbors if args.neighbors is not None else config.NEIGHBORHOOD_SIZE
    theta_cap = args.theta_cap if args.theta_cap is not None else config.THETA_CAP
    outdir = Path(args.outdir if args.outdir else config.DEFAULT_OUTDIR).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    step1_dir = Path(args.step1).expanduser().resolve()
    if not step1_dir.is_absolute():
        step1_dir = (ROOT / step1_dir).resolve()

    outdir = Path(args.outdir if args.outdir else config.DEFAULT_OUTDIR).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)


    # 1) Load prepared inputs
    projects, delta, step1_summary = load_step1(step1_dir)
    n_projects = len(projects)

    # 2) Initialize MOEA/D state
    population, weights, neighborhoods = ifpom.initialize_ifpom(
        pop_size=pop_size,
        num_projects=n_projects,
        num_neighbors=T,
        theta_cap=theta_cap,
    )

    # Ideal point for mixed directions: Z1 max, Z2 min, Z3 max
    ideal_point = [-float("inf"), float("inf"), -float("inf")]
    z_min = [float("inf"), float("inf"), float("inf")]
    z_max = [-float("inf"), -float("inf"), -float("inf")]

    # 3) Evaluate initial population + set initial ideal/z_min/z_max
    for ind in population:
        ind["Z"] = ifpom.evaluate_individual(ind, projects, delta)
        ifpom.update_ideal_point(ind["Z"], ideal_point)

        for k in range(3):
            z_min[k] = min(z_min[k], ind["Z"][k])
            z_max[k] = max(z_max[k], ind["Z"][k])

    # 4) Run MOEA/D
    log: List[Dict[str, Any]] = []
    for gen in range(max_gen):
        population, ideal_point, z_min, z_max = ifpom.moead_generation(
            population=population,
            projects=projects,
            delta_matrix=delta,
            weight_vectors=weights,
            neighborhoods=neighborhoods,
            ideal_point=ideal_point,
            gen=gen,
            max_gen=max_gen,
            z_min=z_min,
            z_max=z_max,
            theta_cap=theta_cap,
        )

        if gen % 10 == 0 or gen == max_gen - 1:
            log.append({"gen": gen, "ideal_point": ideal_point, "z_min": z_min, "z_max": z_max})
            print(f"Gen {gen:03d} | ideal(Z1,Z2,Z3)=({ideal_point[0]:.3f},{ideal_point[1]:.3f},{ideal_point[2]:.3f})")

    # 5) Extract Pareto
    pareto = extract_pareto_front(population)

    # 6) Export artifacts (Step-2 reproducible outputs)
    (outdir / "final_population.json").write_text(json.dumps(population, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "pareto_solutions.json").write_text(json.dumps(pareto, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "convergence_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "step1_summary_copy.json").write_text(json.dumps(step1_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    step2_summary = {
        "algorithm": "MOEA/D",
        "pop_size": pop_size,
        "generations": max_gen,
        "neighbors_T": T,
        "theta_cap": theta_cap,
        "final_ideal_point": ideal_point,
        "z_min_final": z_min,
        "z_max_final": z_max,
        "n_projects": n_projects,
        "n_population": len(population),
        "n_pareto": len(pareto),
        "source_step1_dir": str(step1_dir),
        "saved_files": {
            "final_population": "final_population.json",
            "pareto_solutions": "pareto_solutions.json",
            "convergence_log": "convergence_log.json",
            "step1_summary_copy": "step1_summary_copy.json",
        },
    }
    (outdir / "step2_summary.json").write_text(json.dumps(step2_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Step 2 Completed: MOEA/D finished ===")
    print(f"Saved: {outdir}")
    print(f"Pareto solutions: {len(pareto)}  -> {outdir / 'pareto_solutions.json'}")


if __name__ == "__main__":
    main()
