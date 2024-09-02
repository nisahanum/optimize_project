import pulp
import numpy as np

# Define project data
projects = ['P1', 'P2', 'P3', 'P4']
costs = [1800000, 1300000, 2200000, 1600000]  # Expected costs of projects
benefits = [500000, 300000, 600000, 400000]  # Expected financial benefits (NPV)
risk_adjusted_costs = [200000, 150000, 300000, 250000]  # Risk-adjusted costs
AHP_weights = [0.35, 0.25, 0.20, 0.20]  # AHP-derived weights for projects

# Define financial instruments
financial_instruments = ['Loan', 'Bond', 'Equity']
interest_rates = [0.05, 0.03, 0.08]  # Cost of loans, bonds, and equity (returns)

# Budget and risk parameters
B = 4500000  # Total available budget
mu = 1000000  # Maximum acceptable risk level
lambda_param = 0.5  # Risk aversion parameter
num_simulations = 10000  # Number of simulations for VaR analysis

# Simulated probabilities of default for loans and bonds
default_probabilities = [0.02, 0.01]  # Loan and Bond default probabilities
loss_given_default = [0.6, 0.5]  # Loss given default for Loan and Bond

# Create the optimization problem
model = pulp.LpProblem("IT_Portfolio_Optimization_With_VaR", pulp.LpMaximize)

# Create decision variables
x = pulp.LpVariable.dicts("Project", projects, cat='Binary')
y = pulp.LpVariable.dicts("FinancialInstrument", financial_instruments, lowBound=0)

# Calculate expected loss for loans and bonds
expected_losses = [default_probabilities[j] * loss_given_default[j] * B for j in range(len(default_probabilities))]

# Risk contribution from financial instruments
risk_contribution = np.sum(expected_losses)

# Define the objective function
model += pulp.lpSum([AHP_weights[i] * benefits[i] * x[projects[i]] for i in range(len(projects))]), "Total_Benefit"

# Define constraints
# Budget constraint
model += (pulp.lpSum([costs[i] * x[projects[i]] for i in range(len(projects))]) +
            interest_rates[0] * y['Loan'] +  # Loan cost
            interest_rates[1] * y['Bond'] +  # Bond cost
            interest_rates[2] * y['Equity'] * pulp.lpSum([1 for _ in projects]) <= B,  # Equity cost as expected return
            "Budget_Constraint")

# Total Risk Constraint
model += (pulp.lpSum([risk_adjusted_costs[i] * x[projects[i]] for i in range(len(projects))]) + 
            risk_contribution <= mu, 
            "Total_Risk_Constraint")

# Solve the problem
model.solve()

# Output the results
print("Status:", pulp.LpStatus[model.status])
print("Selected Projects:")
for project in projects:
    if x[project].varValue == 1:
        print(f"- {project} (Cost: {costs[projects.index(project)]}, Benefit: {benefits[projects.index(project)]}, Risk Adjusted Cost: {risk_adjusted_costs[projects.index(project)]})")

print("Total Benefit:", pulp.value(model.objective))

# Financial Instruments Used
print("\nFinancial Instruments Used:")
print (y)
for instrument in financial_instruments:
    if y[instrument].varValue > 0:
        print(f"- {instrument}: Amount Used = {y[instrument].varValue}, Interest Rate = {interest_rates[financial_instruments.index(instrument)]}")

# VaR Analysis
# Simulate interest rate changes and calculate impact on cash flows
simulated_interest_rates = np.random.normal(loc=0, scale=0.01, size=num_simulations)  # Simulating rate changes
initial_cash_flows = np.array([y['Loan'].varValue * interest_rates[0], 
                                y['Bond'].varValue * interest_rates[1]])
simulated_cash_flows = []

for rate_change in simulated_interest_rates:
    new_cash_flows = initial_cash_flows * (1 + rate_change)  # Adjust cash flows by simulated rate change
    simulated_cash_flows.append(sum(new_cash_flows))

# Calculate VaR at 95% confidence level
VaR = np.percentile(simulated_cash_flows, 5)  # 5th percentile
print("\nValue-at-Risk (VaR) at 95% confidence level:", VaR)