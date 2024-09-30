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
    equity_wacc = []
    debt_wacc = []
    rbf_wacc = []
    
    total_wacc_values = []
    
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

        # Append to lists
        equity_wacc.append(equity_contribution)
        debt_wacc.append(debt_contribution)
        rbf_wacc.append(rbf_contribution)
        
        # Total WACC formula: WACC = E/V * re + D/V * rd * (1-T) + RBF/V * r_rbf
        total_wacc = equity_contribution + debt_contribution + rbf_contribution
        total_wacc_values.append(total_wacc)

    return total_wacc_values, equity_wacc, debt_wacc, rbf_wacc

# Run simulation
total_wacc_simulations, equity_wacc_simulations, debt_wacc_simulations, rbf_wacc_simulations = monte_carlo_simulation(num_simulations)

# Plot the distribution of total WACC values from the Monte Carlo simulation
plt.figure(figsize=(12, 8))

# Plot Total WACC distribution
plt.subplot(2, 2, 1)
plt.hist(total_wacc_simulations, bins=50, edgecolor='k', alpha=0.7, color='purple')
plt.title('Total WACC')
plt.xlabel('WACC')
plt.ylabel('Frequency')

# Plot Equity Contribution distribution
plt.subplot(2, 2, 2)
plt.hist(equity_wacc_simulations, bins=50, edgecolor='k', alpha=0.7, color='blue')
plt.title('Equity Contribution to WACC')
plt.xlabel('Equity WACC')
plt.ylabel('Frequency')

# Plot Debt Contribution distribution
plt.subplot(2, 2, 3)
plt.hist(debt_wacc_simulations, bins=50, edgecolor='k', alpha=0.7, color='green')
plt.title('Debt Contribution to WACC')
plt.xlabel('Debt WACC')
plt.ylabel('Frequency')

# Plot RBF Contribution distribution
plt.subplot(2, 2, 4)
plt.hist(rbf_wacc_simulations, bins=50, edgecolor='k', alpha=0.7, color='orange')
plt.title('RBF Contribution to WACC')
plt.xlabel('RBF WACC')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Summary statistics
mean_total_wacc = np.mean(total_wacc_simulations)
std_total_wacc = np.std(total_wacc_simulations)

mean_equity_wacc = np.mean(equity_wacc_simulations)
std_equity_wacc = np.std(equity_wacc_simulations)

mean_debt_wacc = np.mean(debt_wacc_simulations)
std_debt_wacc = np.std(debt_wacc_simulations)

mean_rbf_wacc = np.mean(rbf_wacc_simulations)
std_rbf_wacc = np.std(rbf_wacc_simulations)

print(f"Mean Total WACC: {mean_total_wacc:.4f}")
print(f"Standard Deviation of Total WACC: {std_total_wacc:.4f}")

print(f"Mean Equity WACC: {mean_equity_wacc:.4f}")
print(f"Mean Debt WACC: {mean_debt_wacc:.4f}")
print(f"Mean RBF WACC: {mean_rbf_wacc:.4f}")
