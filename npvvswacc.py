import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
num_simulations = 10000

# Base parameters (mean and standard deviation) for distributions of variables
cost_of_equity_mean = 0.12  # Scenario 1: Change equity cost (default)
cost_of_equity_std = 0.02

cost_of_debt_mean = 0.05    # Scenario 2: Change debt cost (default)
cost_of_debt_std = 0.01

tax_rate_mean = 0.30        # Scenario 3: Change tax rate (default)
tax_rate_std = 0.05

rbf_cost_mean = 0.10        # Scenario 5: Revenue-based financing (RBF)
rbf_cost_std = 0.015

# Proportions for each financing source
equity_proportion_mean = 0.40  # Scenario 4: Adjust financing proportion
equity_proportion_std = 0.1

debt_proportion_mean = 0.40    # Scenario 4: Adjust financing proportion
debt_proportion_std = 0.1

rbf_proportion_mean = 0.20     # RBF proportion
rbf_proportion_std = 0.05

# Monte Carlo simulation for all scenarios
def monte_carlo_simulation(num_simulations, cost_of_equity_mean, cost_of_debt_mean, tax_rate_mean, rbf_cost_mean):
    total_wacc_values = []
    npv_values = []
    
    for _ in range(num_simulations):
        # Generate random samples for each parameter from their respective distributions
        cost_of_equity = np.random.normal(cost_of_equity_mean, cost_of_equity_std)
        cost_of_debt = np.random.normal(cost_of_debt_mean, cost_of_debt_std)
        rbf_cost = np.random.normal(rbf_cost_mean, rbf_cost_std)
        tax_rate = np.random.normal(tax_rate_mean, tax_rate_std)

        # Proportions for each financing source
        equity_proportion = np.clip(np.random.normal(equity_proportion_mean, equity_proportion_std), 0, 1)
        debt_proportion = np.clip(np.random.normal(debt_proportion_mean, debt_proportion_std), 0, 1)
        rbf_proportion = np.clip(np.random.normal(rbf_proportion_mean, rbf_proportion_std), 0, 1)

        # Normalize proportions so they sum to 1
        total_proportion = equity_proportion + debt_proportion + rbf_proportion
        equity_proportion /= total_proportion
        debt_proportion /= total_proportion
        rbf_proportion /= total_proportion

        # Calculate individual contributions to WACC
        equity_contribution = equity_proportion * cost_of_equity
        debt_contribution = debt_proportion * cost_of_debt * (1 - tax_rate)
        rbf_contribution = rbf_proportion * rbf_cost

        # Total WACC formula: WACC = E/V * re + D/V * rd * (1-T) + RBF/V * r_rbf
        total_wacc = equity_contribution + debt_contribution + rbf_contribution
        total_wacc_values.append(total_wacc)

        # NPV calculation with a discount rate of WACC and constant cash flow
        cash_flow = 1000000  # Example cash flow in currency
        project_life = 10    # Project life in years
        npv = np.sum([cash_flow / (1 + total_wacc)**t for t in range(1, project_life + 1)])
        npv_values.append(npv)

    return total_wacc_values, npv_values

# Run the simulation
total_wacc_simulations, npv_simulations = monte_carlo_simulation(
    num_simulations, cost_of_equity_mean, cost_of_debt_mean, tax_rate_mean, rbf_cost_mean)

# Create scatter plot of WACC vs NPV
plt.figure(figsize=(10, 6))
plt.scatter(total_wacc_simulations, npv_simulations, alpha=0.5, color='purple', edgecolors='black')
plt.title("NPV vs WACC")
plt.xlabel("WACC")
plt.ylabel("NPV")
plt.grid(True)
plt.show()
