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


def load_step1(step1_dir: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    projects_path = step1_dir / "projects.json"
    delta_path = step1_dir / "delta_matrix.npy"
    summary_path = step1_dir / "step1_summary.json"

    projects = json.loads(projects_path.read_text(encoding="utf-8"))
    delta = np.load(delta_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return projects, delta, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1", type=str, required=True, help="Path to Step1 output dir (contains projects.json + delta_matrix.npy)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default from config)")
    ap.add_argument("--pop", type=int, default=None, help="Population size (default from config)")
    ap.add_argument("--gen", type=int, default=None, help="Generations (default from config)")
    ap.add_argument("--neighbors", type=int, default=None, help="Neighborhood size T (default from config)")
    ap.add_argument("--theta_cap", type=float, default=None, help="Theta cap (default from config)")
    args = ap.parse_args()

    # ---- defaults from config, overridable by CLI
    pop_size = args.pop if args.pop is not None else config.POPULATION_SIZE
    max_gen = args.gen if args.gen is not None else config.NUM_GENERATIONS
    T = args.neighbors if args.neighbors is not None else config.NEIGHBORHOOD_SIZE
    theta_cap = args.theta_cap if args.theta_cap is not None else config.THETA_CAP
    outdir = Path(args.outdir if args.outdir is not None else config.DEFAULT_OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    step1_dir = Path(args.step1).resolve()

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
        ifpom.update_ideal_point(ind["Z"], ideal_point)  # <-- FIX ORDER

        for k in range(3):
            z_min[k] = min(z_min[k], ind["Z"][k])
            z_max[k] = max(z_max[k], ind["Z"][k])

    # 4) Run MOEA/D
    log = []
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

        # light logging every 10 gens
        if gen % 10 == 0 or gen == max_gen - 1:
            log.append({"gen": gen, "ideal_point": ideal_point, "z_min": z_min, "z_max": z_max})
            print(f"Gen {gen:03d} | ideal(Z1,Z2,Z3)=({ideal_point[0]:.3f},{ideal_point[1]:.3f},{ideal_point[2]:.3f})")

    # 5) Export
    (outdir / "final_population.json").write_text(json.dumps(population, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "convergence_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "step1_summary_copy.json").write_text(json.dumps(step1_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Step 2 Completed: MOEA/D finished ===")
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()

# === SAVE STEP 2 ARTIFACTS ===

outdir.mkdir(parents=True, exist_ok=True)

# 1) Final population
with (outdir / "final_population.json").open("w") as f:
    json.dump(population, f, indent=2)

# 2) Pareto solutions
pareto = ifpom.extract_pareto_front(population)
with (outdir / "pareto_solutions.json").open("w") as f:
    json.dump(pareto, f, indent=2)

# 3) Convergence log
with (outdir / "convergence_log.csv").open("w") as f:
    f.write("generation,Z1_best,Z2_best,Z3_best\n")
    for g, (z1, z2, z3) in convergence_log:
        f.write(f"{g},{z1},{z2},{z3}\n")

# 4) Summary
summary = {
    "algorithm": "MOEA/D",
    "population": pop_size,
    "generations": max_gen,
    "neighbors": n_neighbors,
    "risk_weights": {"w_tech": w_tech, "w_fin": w_fin},
    "theta_cap": theta_cap,
    "best_ideal_point": ideal_point,
    "source_step1": str(step1_dir)
}

#Extractb Paretto


with (outdir / "step2_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
