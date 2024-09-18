import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, LpSolverDefault

# Step 1: Define Projects and Criteria
projects = {
    "Project A": {
        "base_cost": 100000,
        "financing_costs": {
            "RBF": 0.12,
            "Bonds": 0.08,
            "Equity": 0.15
        },
        "risk_adjustment": 0.10,
        "AHP_score": 0.6  # Example AHP score
    },
    "Project B": {
        "base_cost": 150000,
        "financing_costs": {
            "RBF": 0.18,
            "Bonds": 0.12,
            "Equity": 0.23
        },
        "risk_adjustment": 0.15,
        "AHP_score": 0.3
    },
    "Project C": {
        "base_cost": 80000,
        "financing_costs": {
            "RBF": 0.06,
            "Bonds": 0.04,
            "Equity": 0.08
        },
        "risk_adjustment": 0.05,
        "AHP_score": 0.1
    }
}

# Step 2: AHP Pairwise Comparison Matrix (Sample Data)
criteria_matrix = np.array([
    [1, 3, 0.5],  # Project A vs B and C
    [1/3, 1, 0.25],  # Project B vs A and C
    [2, 4, 1]  # Project C vs A and B
])

# Step 3: Calculate AHP Scores
def ahp_scores(matrix):
    normalized_matrix = matrix / matrix.sum(axis=0)
    scores = normalized_matrix.mean(axis=1)
    return scores

# Get AHP scores for projects
scores = ahp_scores(criteria_matrix)
project_scores = {f"Project {chr(65+i)}": score for i, score in enumerate(scores)}

# Step 4: Define Selected Financial Instruments
selected_instruments = {
    "Project A": "Bonds",
    "Project B": "Equity",
    "Project C": "RBF"
}

# Step 5: Calculate Total Cost
def calculate_total_cost(projects, selected_instruments):
    total_costs = {}
    for project, params in projects.items():
        base_cost = params["base_cost"]
        financing_cost = params["financing_costs"][selected_instruments[project]]
        risk_adjustment = params["risk_adjustment"]
        
        total_cost = base_cost + financing_cost + risk_adjustment
        total_costs[project] = total_cost
    
    return total_costs

# Calculate total costs
total_costs = calculate_total_cost(projects, selected_instruments)

# Step 6: Linear Programming Optimization
def optimize_projects(projects, total_costs, budget, max_risk):
    # Create a LP problem
    prob = LpProblem("Project_Selection", LpMaximize)

    # Create decision variables for each project
    project_vars = {project: LpVariable(project, cat='Binary') for project in projects.keys()}

    # Objective function: Maximize total AHP scores
    prob += lpSum([projects[project]["AHP_score"] * project_vars[project] for project in projects]), "Total_Benefit"

    # Budget constraint
    prob += lpSum([total_costs[project] * project_vars[project] for project in projects]) <= budget, "Budget_Constraint"

    # Risk constraint (assuming risk is the risk adjustment for simplicity)
    prob += lpSum([projects[project]["risk_adjustment"] * project_vars[project] for project in projects]) <= max_risk, "Risk_Constraint"

    # Solve the problem
    prob.solve()

    # Display the results
    selected_projects = [project for project in projects.keys() if project_vars[project].value() == 1]
    return selected_projects, LpStatus[prob.status], prob.objective.value()

# Define budget and maximum risk
budget = 250000
max_risk = 0.25

# Optimize project selection
selected_projects, status, total_benefit = optimize_projects(projects, total_costs, budget, max_risk)

# Output results
print("AHP Scores for Projects:")
for project, score in project_scores.items():
    print(f"{project}: {score:.4f}")

print("\nTotal Costs for Selected Projects:")
for project, cost in total_costs.items():
    print(f"{project}: ${cost:.2f}")

print(f"\nOptimization Status: {status}")
print(f"Selected Projects: {selected_projects}")
print(f"Total AHP Score of Selected Projects: {total_benefit:.4f}")