# Re-importing necessary libraries and reloading the dataset
import pandas as pd

# Load the dataset
file_path = "rbf_financing_summary_tuned.csv"  # Replace with the correct path if needed
rbf_data = pd.read_csv(file_path)

# Define the hybrid financing mix proportions
financing_mix = {
    "Debt (%)": 40,  # 40% debt
    "Equity (%)": 40,  # 40% equity
    "RBF (%)": 5,  # 20% RBF
}

# Define financing costs for each instrument
cost_of_debt = 5  # 5% annual interest rate for debt
cost_of_equity = 12  # 12% expected return for equity
cost_of_rbf = 1.5  # 10% implied cost for RBF

# Calculate Blended WACC for each project
rbf_data["Blended WACC (%)"] = (
    (financing_mix["Debt (%)"] / 100) * cost_of_debt +
    (financing_mix["Equity (%)"] / 100) * cost_of_equity +
    (financing_mix["RBF (%)"] / 100) * cost_of_rbf
)

# Calculate hybrid net cash flow after financing costs
rbf_data["Hybrid Financing Cost (USD)"] = (
    rbf_data["Initial Investment (USD)"] * rbf_data["Blended WACC (%)"] / 100
)
rbf_data["Net Cash Flow (Hybrid) (USD)"] = (
    rbf_data["Net Cash Flow (RBF) (USD)"] - rbf_data["Hybrid Financing Cost (USD)"]
)

# Calculate ROI for hybrid financing
rbf_data["ROI (Hybrid) (%)"] = (
    rbf_data["Net Cash Flow (Hybrid) (USD)"] / rbf_data["Initial Investment (USD)"]
) * 100

# Display updated dataset with hybrid financing metrics
# Print the DataFrame
print("Hybrid Financing Scheme Summary:")
print(rbf_data)

# Save to a CSV file (optional)
#rbf_data.to_csv("hybrid_financing_summary.csv", index=False)
#print("Results saved to hybrid_financing_summary.csv")
