"""
common_ifpom_final.py  (FIXED)
Core optimization logic for IFPOM (Integrated Flexible Portfolio Optimization Model)
used by MOEA/D experiments.

Key FIX (critical):
- Funding penalty in Z2 must use INDIVIDUAL funding shares (ind['alpha'][i]..ind['delta'][i]),
  NOT project-level p.get('alpha'...) values. Otherwise all individuals can collapse to identical Z.

Assumptions:
- projects is a list of dicts aligned with delta_matrix order.
- ind contains:
    ind['x']      : list[int] length N (0/1 selection)
    ind['alpha']  : list[float] length N
    ind['beta']   : list[float] length N
    ind['theta']  : list[float] length N
    ind['gamma']  : list[float] length N
    ind['delta']  : list[float] length N
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import random

from config import (
    INITIAL_MUTATION_RATE,
    MIN_MUTATION_RATE,
    DIVERSITY_THRESHOLD,
    DIVERSITY_ACCEPTANCE_PROB,
)

from tchebycheff_utils import tchebycheff_eval


# ---------------------------------------------------------------------
# Objective evaluation (Z1, Z2, Z3)
# ---------------------------------------------------------------------
def evaluate_individual(
    ind: Dict[str, Any],
    projects: List[Dict[str, Any]],
    delta_matrix: np.ndarray,
    n_samples: int = 10,
) -> List[float]:
    """
    Evaluate one candidate solution.

    Objectives:
      Z1 (maximize): Adjusted Strategic Value + pairwise synergy δ_ij
      Z2 (minimize): Risk-adjusted effective cost (fuzzy sampling + funding penalty)
      Z3 (maximize): Total synergy proxy (same + cross)

    IMPORTANT:
    - Funding penalty MUST come from *ind* (decision variables), not from *projects*.
      Using p.get("alpha"... ) makes Z2 identical across population when projects carry fixed values.
    """
    n = len(projects)
    Z1, Z2, Z3 = 0.0, 0.0, 0.0

    # Benefit-aware multipliers (λ_b): set to 1.0 if group is unknown.
    benefit_lambda = {
        "Operational Efficiency": 1.50,
        "Customer Experience": 1.43,
        "Business Culture": 1.00,
    }

    # Funding penalty multipliers (relative cost/friction encoded in Z2)
    # Align with your manuscript "Funding penalty in Z2" table.
    funding_penalty_weights = np.array([0.9, 1.0, 1.3, 1.1, 1.2], dtype=float)  # α,β,θ,γ,δ

    # ---- Project-level contributions
    for i in range(n):
        if ind["x"][i] != 1:
            continue

        p = projects[i]
        λ_b = float(benefit_lambda.get(p.get("benefit_group", "Business Culture"), 1.0))

        svs = float(p["svs"])
        risk = float(p["risk"])

        synergy_score = float(p["synergy_same"]) + float(p["synergy_cross"])

        # Z1: Strategic value adjusted by risk, plus benefit-aware synergy proxy
        Z1 += (svs + λ_b * synergy_score) * (1.0 - risk)

        # Z3: Total synergy proxy (reported separately)
        Z3 += synergy_score

        # ---- Z2: fuzzy-cost sampling (triangular) with synergy discount + funding penalty
        c1, c2, c3 = p["fuzzy_cost"]
        samples = np.random.triangular(float(c1), float(c2), float(c3), int(n_samples))

        # ✅ FIX: Funding mix vector must come from INDIVIDUAL decision variables
        f = np.array(
            [
                float(ind["alpha"][i]),
                float(ind["beta"][i]),
                float(ind["theta"][i]),
                float(ind["gamma"][i]),
                float(ind["delta"][i]),
            ],
            dtype=float,
        )

        # Weighted cost-of-capital / contractual friction proxy
        penalty = float(np.dot(f, funding_penalty_weights))

        # Compute average effective cost over samples with guardrails
        eff_cost_sum = 0.0
        for sc in samples:
            # synergy discount (do not go below 1.0)
            base_cost = max(1.0, float(sc) - synergy_score)
            effective_cost = base_cost * penalty

            # Safety cap to prevent unrealistic explosion
            effective_cost = min(effective_cost, base_cost * 1.5)
            eff_cost_sum += effective_cost

        avg_cost = eff_cost_sum / float(n_samples)

        # Risk-adjusted cost contribution
        Z2 += avg_cost * risk

    # ---- Pairwise inter-project synergy δ_ij added to Z1
    for i in range(n):
        if ind["x"][i] != 1:
            continue
        λ_i = float(benefit_lambda.get(projects[i].get("benefit_group", "Business Culture"), 1.0))

        for j in range(i + 1, n):
            if ind["x"][j] != 1:
                continue
            λ_j = float(benefit_lambda.get(projects[j].get("benefit_group", "Business Culture"), 1.0))
            avg_λ = 0.5 * (λ_i + λ_j)
            Z1 += avg_λ * float(delta_matrix[i, j])

    return [Z1, Z2, Z3]


def update_ideal_point(Z: List[float], ideal: List[float]) -> None:
    """Update ideal point for mixed-direction objectives: Z1 max, Z2 min, Z3 max."""
    ideal[0] = max(ideal[0], Z[0])
    ideal[1] = min(ideal[1], Z[1])
    ideal[2] = max(ideal[2], Z[2])


# ---------------------------------------------------------------------
# Initialization (population, weights, neighborhoods)
# ---------------------------------------------------------------------
def initialize_ifpom(
    pop_size: int,
    num_projects: int,
    num_neighbors: int = 20,
    theta_cap: float = 0.4,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Create an initial population, weight vectors, and neighborhoods for MOEA/D.

    - x initialized Bernoulli(0.5), repaired to ensure at least one project selected.
    - Funding per project is Dirichlet then repaired to enforce theta <= theta_cap.
    """
    population: List[Dict[str, Any]] = []

    for _ in range(pop_size):
        x = [1 if random.random() < 0.5 else 0 for _ in range(num_projects)]
        if sum(x) == 0:
            x[random.randint(0, num_projects - 1)] = 1

        funding = []
        for _p in range(num_projects):
            f = np.random.dirichlet([1, 1, 1, 1, 1]).astype(float)
            if f[2] > theta_cap:
                f[2] = theta_cap
                f = f / f.sum()
            funding.append(f)

        F = np.array(funding, dtype=float)

        ind = {
            "x": x,
            "alpha": F[:, 0].tolist(),
            "beta": F[:, 1].tolist(),
            "theta": F[:, 2].tolist(),
            "gamma": F[:, 3].tolist(),
            "delta": F[:, 4].tolist(),
            "Z": [None, None, None],
        }
        population.append(ind)

    def uniform_weight_vectors(n_objs: int, divisions: int) -> np.ndarray:
        import itertools

        vectors = []
        for partition in itertools.combinations_with_replacement(range(divisions + 1), n_objs):
            if sum(partition) == divisions:
                vectors.append(np.array(partition, dtype=float) / float(divisions))
        return np.array(vectors, dtype=float)

    raw_vectors = uniform_weight_vectors(n_objs=3, divisions=13)

    if len(raw_vectors) < pop_size:
        extra = np.random.dirichlet(np.ones(3), size=pop_size - len(raw_vectors))
        weight_vectors = np.vstack([raw_vectors, extra])
    else:
        weight_vectors = raw_vectors[:pop_size]

    distances = np.linalg.norm(weight_vectors[:, None, :] - weight_vectors[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, : min(num_neighbors, pop_size)]

    assert len(population) == len(weight_vectors) == len(neighborhoods), (
        f"Mismatch: population={len(population)}, weights={len(weight_vectors)}, neighbors={len(neighborhoods)}"
    )

    return population, weight_vectors, neighborhoods


# ---------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------
def crossover(parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
    """Uniform crossover across all decision components."""
    child = {
        "x": [],
        "alpha": [],
        "beta": [],
        "theta": [],
        "gamma": [],
        "delta": [],
        "Z": [None, None, None],
    }

    for key in ["x", "alpha", "beta", "theta", "gamma", "delta"]:
        for i in range(len(parent1[key])):
            child[key].append(parent1[key][i] if random.random() < 0.5 else parent2[key][i])

    return child


# ---------------------------------------------------------------------
# One MOEA/D generation update
# ---------------------------------------------------------------------
def moead_generation(
    population: List[Dict[str, Any]],
    projects: List[Dict[str, Any]],
    delta_matrix: np.ndarray,
    weight_vectors: np.ndarray,
    neighborhoods: np.ndarray,
    ideal_point: List[float],
    gen: int,
    max_gen: int,
    z_min: List[float],
    z_max: List[float],
    theta_cap: float = 0.4,
) -> Tuple[List[Dict[str, Any]], List[float], List[float], List[float]]:
    """
    Execute one generation of MOEA/D updates.

    - Mutation rate annealed linearly from INITIAL_MUTATION_RATE to MIN_MUTATION_RATE.
    - x mutation: bit-flip.
    - funding mutation: Gaussian noise + simplex repair + theta cap.
    - replacement: Tchebycheff scalarization with optional diversity acceptance.
    """
    mutation_prob = INITIAL_MUTATION_RATE * (1.0 - gen / float(max_gen)) + MIN_MUTATION_RATE
    n = len(projects)

    for i in range(len(population)):
        neighbors = neighborhoods[i]
        p1, p2 = random.sample(list(neighbors), 2)

        child = crossover(population[p1], population[p2])

        # --- Mutate x
        for m in range(n):
            if random.random() < mutation_prob:
                child["x"][m] = 1 - child["x"][m]

        if sum(child["x"]) == 0:
            child["x"][random.randint(0, n - 1)] = 1

        # --- Mutate funding ratios for selected projects only
        for m in range(n):
            if child["x"][m] != 1:
                continue
            if random.random() >= mutation_prob:
                continue

            ratios = np.array(
                [
                    float(child["alpha"][m]),
                    float(child["beta"][m]),
                    float(child["theta"][m]),
                    float(child["gamma"][m]),
                    float(child["delta"][m]),
                ],
                dtype=float,
            )

            noise = np.random.normal(0.0, 0.02, size=5)
            mutated = np.maximum(ratios + noise, 1e-4)
            mutated = mutated / mutated.sum()

            if mutated[2] > theta_cap:
                excess = mutated[2] - theta_cap
                mutated[2] = theta_cap

                redistribute_idx = [0, 1, 3, 4]
                remaining = mutated[redistribute_idx]
                remaining_sum = remaining.sum()
                if remaining_sum <= 0:
                    mutated[0] = 1.0 - theta_cap
                    mutated[1] = mutated[3] = mutated[4] = 0.0
                else:
                    mutated[redistribute_idx] = remaining + (excess * remaining / remaining_sum)

            child["alpha"][m] = float(mutated[0])
            child["beta"][m] = float(mutated[1])
            child["theta"][m] = float(mutated[2])
            child["gamma"][m] = float(mutated[3])
            child["delta"][m] = float(mutated[4])

        # --- Evaluate child
        child["Z"] = evaluate_individual(child, projects, delta_matrix)

        # --- Neighbor replacement
        for j in neighbors:
            old_fit = tchebycheff_eval(population[j]["Z"], weight_vectors[j], ideal_point, z_min, z_max)
            new_fit = tchebycheff_eval(child["Z"], weight_vectors[j], ideal_point, z_min, z_max)

            diversity_x = sum(child["x"][k] != population[j]["x"][k] for k in range(len(child["x"])))

            if (new_fit < old_fit) or (
                diversity_x >= DIVERSITY_THRESHOLD and random.random() < DIVERSITY_ACCEPTANCE_PROB
            ):
                population[j] = deepcopy(child)

        update_ideal_point(child["Z"], ideal_point)
        for k in range(3):
            z_min[k] = min(z_min[k], child["Z"][k])
            z_max[k] = max(z_max[k], child["Z"][k])

    return population, ideal_point, z_min, z_max
