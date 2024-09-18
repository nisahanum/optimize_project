import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

# Define Projects Data
projects = {
    "Project A": {
        "base_cost": 100000,
        "financing_cost": 12000,
        "risk_adjustment": 0.10,
        "AHP_score": 0.5
    },
    "Project B": {
        "base_cost": 150000,
        "financing_cost": 18000,
        "risk_adjustment": 0.15,
        "AHP_score": 0.3
    },
    "Project C": {
        "base_cost": 80000,
        "financing_cost": 6000,
        "risk_adjustment": 0.05,
        "AHP_score": 0.2
    }
}

# Define total budget and maximum risk
budget = 250000
max_risk = 0.25

# Calculate Total Costs
def calculate_total_cost(project):
    return project["base_cost"] + project["financing_cost"] + project["risk_adjustment"]

total_costs = {name: calculate_total_cost(details) for name, details in projects.items()}

# Linear Programming Optimization
def optimize_projects(projects, total_costs, budget, max_risk):
    # Create a LP problem
    prob = LpProblem("Project_Selection", LpMaximize)

    # Create decision variables for each project
    project_vars = {project: LpVariable(project, cat='Binary') for project in projects.keys()}

    # Objective function: Maximize total AHP scores
    prob += lpSum([projects[project]["AHP_score"] * project_vars[project] for project in projects]), "Total_Benefit"

    # Budget constraint
    prob += lpSum([total_costs[project] * project_vars[project] for project in projects]) <= budget, "Budget_Constraint"

    # Risk constraint
    prob += lpSum([projects[project]["risk_adjustment"] * project_vars[project] for project in projects]) <= max_risk, "Risk_Constraint"

    # Solve the problem
    prob.solve()

    # Return selected projects and status
    selected_projects = [project for project in projects.keys() if project_vars[project].value() == 1]
    return selected_projects, LpStatus[prob.status], prob.objective.value()

# Optimize project selection
selected_projects, status, total_benefit = optimize_projects(projects, total_costs, budget, max_risk)

# Output results
print("Total Costs for Each Project:")
for project, cost in total_costs.items():
    print(f"{project}: ${cost:.2f}")

print(f"\nOptimization Status: {status}")
print(f"Selected Projects: {selected_projects}")
print(f"Total AHP Score of Selected Projects: {total_benefit:.4f}")