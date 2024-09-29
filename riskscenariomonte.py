import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
num_simulations = 10000

# Parameters (mean and standard deviation) for distributions of variables
cost_of_equity_mean = 0.12
cost_of_equity_std = 0.02

cost_of_debt_mean = 0.05
cost_of_debt_std = 0.01

rbf_cost_mean = 0.10
rbf_cost_std = 0.015

tax_rate_mean = 0.30
tax_rate_std = 0.05

# Proportions of each financing source (mean and std)
equity_proportion_mean = 0.40
equity_proportion_std = 0.1

debt_proportion_mean = 0.40
debt_proportion_std = 0.1

rbf_proportion_mean = 0.20
rbf_proportion_std = 0.05

# Monte Carlo simulation
def monte_carlo_simulation(num_simulations):
    wacc_values = []
    
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

        # WACC formula: WACC = E/V * re + D/V * rd * (1-T) + RBF/V * r_rbf
        wacc = (equity_proportion * cost_of_equity) + \
               (debt_proportion * cost_of_debt * (1 - tax_rate)) + \
               (rbf_proportion * rbf_cost)

        wacc_values.append(wacc)

    return wacc_values

# Run simulation
wacc_simulations = monte_carlo_simulation(num_simulations)

# Plot the distribution of WACC values from the Monte Carlo simulation
plt.hist(wacc_simulations, bins=50, edgecolor='k', alpha=0.7, color='purple')
plt.title('Monte Carlo Simulation: Impact of Multiple Financing Sources on WACC')
plt.xlabel('WACC')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Summary statistics
mean_wacc = np.mean(wacc_simulations)
std_wacc = np.std(wacc_simulations)

print(f"Mean WACC: {mean_wacc:.4f}")
print(f"Standard Deviation of WACC: {std_wacc:.4f}")
