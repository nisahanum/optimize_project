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
    equity_wacc = []
    debt_wacc = []
    rbf_wacc = []
    
    total_wacc_values = []
    npv_values_equity = []
    npv_values_debt = []
    npv_values_rbf = []
    
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

<<<<<<< HEAD
        # Simulating revenue growth and calculating NPV with flexible RBF payments
        initial_revenue = 1000000  # Initial revenue in year 1
        npv = 0
        for year in range(1, 11):  # Assuming a 10-year project
            growth_rate = np.random.normal(growth_rate_mean, growth_rate_std)
            revenue = initial_revenue * ((1 + growth_rate) ** year)
            discount_factor = (1 + total_wacc) ** year
            npv += revenue / discount_factor

        total_npv_values.append(npv)

    return total_wacc_values, total_npv_values

# Running the simulation
wacc_simulations, npv_simulations = simulate_rbf_flexibility(num_simulations)

# Plot the results for WACC
plt.figure(figsize=(10, 6))
plt.hist(wacc_simulations, bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.title('WACC Distribution - RBF vs. Equity vs. Debt')
plt.xlabel('WACC')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot the results for NPV
plt.figure(figsize=(10, 6))
plt.hist(npv_simulations, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('NPV Distribution - RBF vs. Equity vs. Debt')
plt.xlabel('NPV')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Summary statistics
mean_wacc = np.mean(wacc_simulations)
std_wacc = np.std(wacc_simulations)
mean_npv = np.mean(npv_simulations)
std_npv = np.std(npv_simulations)

print(f"Mean WACC: {mean_wacc:.4f}")
print(f"Standard Deviation of WACC: {std_wacc:.4f}")
print(f"Mean NPV: {mean_npv:.2f}")
print(f"Standard Deviation of NPV: {std_npv:.2f}")
 
=======
        # NPV calculation with a discount rate of WACC and constant cash flow
        cash_flow = 1000000  # Example cash flow in currency
        project_life = 10    # Project life in years
        
        npv_equity = np.sum([cash_flow / (1 + equity_contribution)**t for t in range(1, project_life + 1)])
        npv_debt = np.sum([cash_flow / (1 + debt_contribution)**t for t in range(1, project_life + 1)])
        npv_rbf = np.sum([cash_flow / (1 + rbf_contribution)**t for t in range(1, project_life + 1)])
        
        npv_values_equity.append(npv_equity)
        npv_values_debt.append(npv_debt)
        npv_values_rbf.append(npv_rbf)

    return total_wacc_values, npv_values_equity, npv_values_debt, npv_values_rbf, equity_wacc, debt_wacc, rbf_wacc

# Function to run and plot separate visualizations for each component
def run_scenario_separate(title, cost_of_equity_mean, cost_of_debt_mean, tax_rate_mean, rbf_cost_mean):
    total_wacc_simulations, npv_equity, npv_debt, npv_rbf, equity_wacc_simulations, debt_wacc_simulations, rbf_wacc_simulations = monte_carlo_simulation(
        num_simulations, cost_of_equity_mean, cost_of_debt_mean, tax_rate_mean, rbf_cost_mean)

    # Plot the results for each WACC component
    plt.figure(figsize=(10, 6))
    plt.hist(equity_wacc_simulations, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Equity WACC')
    plt.hist(debt_wacc_simulations, bins=50, alpha=0.7, color='green', edgecolor='black', label='Debt WACC')
    plt.hist(rbf_wacc_simulations, bins=50, alpha=0.7, color='orange', edgecolor='black', label='RBF WACC')
    plt.title(f"WACC Components Distribution - {title}")
    plt.xlabel('WACC')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the results for NPV comparison with different colors
    plt.figure(figsize=(10, 6))
    plt.hist(npv_equity, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Equity NPV')
    plt.hist(npv_debt, bins=50, alpha=0.7, color='green', edgecolor='black', label='Debt NPV')
    plt.hist(npv_rbf, bins=50, alpha=0.7, color='orange', edgecolor='black', label='RBF NPV')
    plt.title(f"NPV Distribution - {title}")
    plt.xlabel('NPV')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print summary statistics
    mean_wacc = np.mean(total_wacc_simulations)
    std_wacc = np.std(total_wacc_simulations)
    mean_npv_equity = np.mean(npv_equity)
    std_npv_equity = np.std(npv_equity)
    mean_npv_debt = np.mean(npv_debt)
    std_npv_debt = np.std(npv_debt)
    mean_npv_rbf = np.mean(npv_rbf)
    std_npv_rbf = np.std(npv_rbf)
    
    print(f"Scenario: {title}")
    print(f"Mean WACC: {mean_wacc:.4f}")
    print(f"Standard Deviation of WACC: {std_wacc:.4f}")
    print(f"Mean NPV (Equity): {mean_npv_equity:.2f}, Std Dev: {std_npv_equity:.2f}")
    print(f"Mean NPV (Debt): {mean_npv_debt:.2f}, Std Dev: {std_npv_debt:.2f}")
    print(f"Mean NPV (RBF): {mean_npv_rbf:.2f}, Std Dev: {std_npv_rbf:.2f}")
    print("-" * 50)

# Scenario 1: Separate visualization for WACC components and NPV comparison with different colors
run_scenario_separate("Debt, Equity and RBF Comparison", cost_of_equity_mean=0.12, cost_of_debt_mean=0.05, tax_rate_mean=0.30, rbf_cost_mean=0.15)
>>>>>>> dc4be83c376b53b4bd7ee7d724132fc1948c93fe
