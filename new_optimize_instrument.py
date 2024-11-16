import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value

# Step 1: Define Projects and Criteria with realistic values
projects = {
    "Project A": {
        "base_cost": 100000,
        "financing_costs": {
            "RBF": 0.12,  # 12% financing cost
            "Bonds": 0.08,
            "Equity": 0.15
        },
        "risk_adjustment": 0.10,  # 10% risk adjustment
    },
    "Project B": {
        "base_cost": 150000,
        "financing_costs": {
            "RBF": 0.18,  # 18% financing cost
            "Bonds": 0.12,
            "Equity": 0.23
        },
        "risk_adjustment": 0.15,  # 15% risk adjustment
    },
    "Project C": {
        "base_cost": 80000,
        "financing_costs": {
            "RBF": 0.06,  # 6% financing cost
            "Bonds": 0.04,
            "Equity": 0.08
        },
        "risk_adjustment": 0.05,  # 5% risk adjustment
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

# Step 4: Linear Programming Optimization with Mixed Financing
def optimize_projects_and_financing(projects, project_scores, budget, max_risk):
    # Create an LP problem to maximize AHP scores
    prob = LpProblem("Project_Selection", LpMaximize)

    # Create decision variables for each project and continuous financing options
    project_vars = {project: LpVariable(project, cat='Binary') for project in projects.keys()}
    financing_vars = {f"{project}_{financing}": LpVariable(f"{project}_{financing}", lowBound=0, upBound=1)
                      for project in projects.keys()
                      for financing in projects[project]["financing_costs"].keys()}
    
    # Add positive and negative deviation variables to linearize abs()
    penalty_weight = 0.05  # A small penalty factor for unbalanced financing
    positive_dev_vars = {}
    negative_dev_vars = {}

    for project in projects.keys():
        for financing in projects[project]["financing_costs"].keys():
            positive_dev_vars[f"{project}_{financing}"] = LpVariable(f"positive_dev_{project}_{financing}", lowBound=0)
            negative_dev_vars[f"{project}_{financing}"] = LpVariable(f"negative_dev_{project}_{financing}", lowBound=0)

    # Objective function: Maximize total AHP scores for selected projects
    prob += lpSum([project_scores[project] * project_vars[project]
                   - penalty_weight * lpSum([positive_dev_vars[f"{project}_{financing}"] + negative_dev_vars[f"{project}_{financing}"]
                                             for financing in projects[project]["financing_costs"].keys()])
                   for project in projects]), "Total_AHP_Score"

    # Constraints
    # Budget Constraint: Ensure total project cost (base + financing cost) is within the available budget
    prob += lpSum([
        projects[project]["base_cost"] * project_vars[project] +
        lpSum([projects[project]["base_cost"] * projects[project]["financing_costs"][financing] * financing_vars[f"{project}_{financing}"]
               for financing in projects[project]["financing_costs"].keys()])
        for project in projects]) <= budget, "Budget_Constraint"
    
    # Risk Constraint: Ensure total risk adjustment does not exceed the maximum allowable risk
    prob += lpSum([projects[project]["risk_adjustment"] * projects[project]["base_cost"] * project_vars[project] for project in projects]) <= max_risk * budget, "Risk_Constraint"

    # Ensure that the sum of financing proportions for each project equals 1 (mixed financing)
    for project in projects.keys():
        prob += lpSum([financing_vars[f"{project}_{financing}"] for financing in projects[project]["financing_costs"].keys()]) == project_vars[project], f"Financing_Proportion_{project}"

    # Add constraints for positive and negative deviation variables
    for project in projects.keys():
        for financing in projects[project]["financing_costs"].keys():
            prob += financing_vars[f"{project}_{financing}"] - (1/len(projects[project]["financing_costs"])) == positive_dev_vars[f"{project}_{financing}"] - negative_dev_vars[f"{project}_{financing}"]

    # Solve the problem
    prob.solve()

    # Debugging outputs
    print("Objective Value:", value(prob.objective))
    print("Project Variables:")
    for project in project_vars:
        print(f"{project}: {project_vars[project].value()}")
    print("Financing Variables:")
    for financing in financing_vars:
        print(f"{financing}: {financing_vars[financing].value()}")

    # Calculate total cost of selected projects
    total_cost = 0
    for project in projects:
        if project_vars[project].value() == 1:  # Project is selected
            project_base_cost = projects[project]["base_cost"]
            selected_financing_cost = sum([project_base_cost * projects[project]["financing_costs"][financing] * financing_vars[f"{project}_{financing}"].value()
                                           for financing in projects[project]["financing_costs"].keys()])
            total_cost += project_base_cost + selected_financing_cost

    # Collect selected projects and their financing options
    selected_projects = []
    selected_financing = {}
    for project in projects.keys():
        if project_vars[project].value() == 1:
            selected_projects.append(project)
            selected_financing[project] = {financing: financing_vars[f"{project}_{financing}"].value()
                                           for financing in projects[project]["financing_costs"].keys() if financing_vars[f"{project}_{financing}"].value() > 0}

    return selected_projects, selected_financing, LpStatus[prob.status], total_cost

# Define a higher budget and risk threshold for more flexibility
budget = 350000  # Increased budget
max_risk = 0.40  # Loosened risk constraint

# Optimize project selection and financial instruments
selected_projects, selected_financing, status, total_cost = optimize_projects_and_financing(projects, project_scores, budget, max_risk)

# Output results
print(f"Optimization Status: {status}")
print(f"Selected Projects: {selected_projects}")
print(f"Selected Financing Instruments: {selected_financing}")
print(f"Total Cost of Selected Projects: ${total_cost:.2f}")
