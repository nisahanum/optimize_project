import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of simulations
n_simulations = 10000

# Function to calculate WACC
def calculate_wacc(r_e, r_d, T, omega_e, omega_d, omega_rbf, r_rbf):
    return (omega_e * r_e) + (omega_d * r_d * (1 - T)) + (omega_rbf * r_rbf)

# Function to calculate NPV
def calculate_npv(cash_flows, wacc, initial_investment):
    return np.sum(cash_flows / (1 + wacc) ** np.arange(1, len(cash_flows) + 1)) - initial_investment

# Set parameters
initial_investment = 500000
cash_flows = np.array([120000, 140000, 160000, 180000, 200000])

# Define distributions for Basic WACC parameters
base_case = {
    "r_e": 0.08,  # Cost of equity
    "r_d": 0.05,  # Cost of debt
    "T": 0.30,    # Tax rate
    "omega_e": 0.50,  # Weight of equity
    "omega_d": 0.50,  # Weight of debt
    "omega_rbf": 0.0,  # No RBF in Basic WACC
    "r_rbf": 0.10   # Assume fixed cost for RBF in New WACC
}

# Define distributions for New WACC parameters
new_case = {
    "r_e": 0.08,  # Cost of equity
    "r_d": 0.05,  # Cost of debt
    "T": 0.30,    # Tax rate
    "omega_e": 0.45,  # Weight of equity
    "omega_d": 0.45,  # Weight of debt
    "omega_rbf": 0.10, # Weight of RBF
    "r_rbf": 0.10   # Cost of RBF
}

# Lists to hold results
basic_wacc_results = []
new_wacc_results = []
npv_basic_results = []
npv_new_results = []

# Monte Carlo simulation for Basic and New WACC
for _ in range(n_simulations):
    # Basic WACC parameters (random sampling for demonstration)
    r_e_basic = np.random.normal(base_case["r_e"], 0.01)  # Cost of equity
    r_d_basic = np.random.normal(base_case["r_d"], 0.005)  # Cost of debt
    T_basic = np.random.uniform(0.25, 0.35)  # Tax rate

    omega_e_basic = base_case["omega_e"]
    omega_d_basic = base_case["omega_d"]
    omega_rbf_basic = base_case["omega_rbf"]

    # Calculate Basic WACC and NPV
    basic_wacc = calculate_wacc(r_e_basic, r_d_basic, T_basic, 
                                 omega_e_basic, omega_d_basic, omega_rbf_basic, base_case["r_rbf"])
    npv_basic = calculate_npv(cash_flows, basic_wacc, initial_investment)

    # Store results for Basic WACC
    basic_wacc_results.append(basic_wacc)
    npv_basic_results.append(npv_basic)

    # New WACC parameters (random sampling for demonstration)
    r_e_new = np.random.normal(new_case["r_e"], 0.01)  # Cost of equity
    r_d_new = np.random.normal(new_case["r_d"], 0.005)  # Cost of debt
    T_new = np.random.uniform(0.25, 0.35)  # Tax rate

    omega_e_new = new_case["omega_e"]
    omega_d_new = new_case["omega_d"]
    omega_rbf_new = new_case["omega_rbf"]

    # Calculate New WACC and NPV
    new_wacc = calculate_wacc(r_e_new, r_d_new, T_new, 
                              omega_e_new, omega_d_new, omega_rbf_new, new_case["r_rbf"])
    npv_new = calculate_npv(cash_flows, new_wacc, initial_investment)

    # Store results for New WACC
    new_wacc_results.append(new_wacc)
    npv_new_results.append(npv_new)

# Convert results to DataFrame for analysis
results_df = pd.DataFrame({
    'Basic WACC': basic_wacc_results,
    'New WACC': new_wacc_results,
    'NPV Basic': npv_basic_results,
    'NPV New': npv_new_results
})

# Analyze results
mean_basic_wacc = results_df['Basic WACC'].mean()
mean_new_wacc = results_df['New WACC'].mean()
mean_npv_basic = results_df['NPV Basic'].mean()
mean_npv_new = results_df['NPV New'].mean()

std_basic_wacc = results_df['Basic WACC'].std()
std_new_wacc = results_df['New WACC'].std()
std_npv_basic = results_df['NPV Basic'].std()
std_npv_new = results_df['NPV New'].std()

# Print summary statistics
print(f"Basic WACC: Mean = {mean_basic_wacc:.4%}, Std Dev = {std_basic_wacc:.4%}")
print(f"New WACC: Mean = {mean_new_wacc:.4%}, Std Dev = {std_new_wacc:.4%}")
print(f"NPV Basic: Mean = ${mean_npv_basic:,.2f}, Std Dev = ${std_npv_basic:,.2f}")
print(f"NPV New: Mean = ${mean_npv_new:,.2f}, Std Dev = ${std_npv_new:,.2f}")

# Optionally, you can plot histograms of the results
plt.figure(figsize=(12, 6))

# WACC Distribution
plt.subplot(1, 2, 1)
plt.hist(results_df['Basic WACC'], bins=50, color='blue', alpha=0.7, label='Basic WACC')
plt.hist(results_df['New WACC'], bins=50, color='orange', alpha=0.7, label='New WACC')
plt.title('Distribution of WACC')
plt.xlabel('WACC')
plt.ylabel('Frequency')
plt.legend()

# NPV Distribution
plt.subplot(1, 2, 2)
plt.hist(results_df['NPV Basic'], bins=50, color='green', alpha=0.7, label='NPV Basic')
plt.hist(results_df['NPV New'], bins=50, color='red', alpha=0.7, label='NPV New')
plt.title('Distribution of NPV')
plt.xlabel('NPV')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()