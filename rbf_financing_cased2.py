import pandas as pd
import matplotlib.pyplot as plt

# Define RBF parameters
revenue_share_percentage = 15  # Increased from 10% to 15%
target_repayment_multiple = 1.2  # Reduced from 1.5x to 1.2x
max_years = 15  # Maximum repayment period (capped at 10 years)


# Load the dataset
file_path = "loan1_project_portfolio_summary.csv"  # Update with the correct path if needed
loan1_data = pd.read_csv(file_path)

# Create a copy of the dataset for RBF calculations
rbf_data = loan1_data.copy()

# Calculate RBF-specific fields
rbf_data["Revenue Share (USD)"] = rbf_data["Annual Revenue (USD)"] * (revenue_share_percentage / 100)
rbf_data["Target Repayment (USD)"] = rbf_data["Initial Investment (USD)"] * target_repayment_multiple

# Calculate the required years to repay based on revenue share, capped at max_years
rbf_data["Required Years"] = (
    rbf_data["Target Repayment (USD)"] / rbf_data["Revenue Share (USD)"]
).apply(lambda x: min(round(x), max_years))

# Calculate the total repayment period cash inflows
rbf_data["Total Revenue (RBF Period) (USD)"] = rbf_data["Annual Revenue (USD)"] * rbf_data["Required Years"]

# Calculate the Net Cash Flow under RBF
rbf_data["Net Cash Flow (RBF) (USD)"] = rbf_data["Total Revenue (RBF Period) (USD)"] - rbf_data["Target Repayment (USD)"]

# Compute ROI for RBF
rbf_data["ROI (RBF) (%)"] = (rbf_data["Net Cash Flow (RBF) (USD)"] / rbf_data["Initial Investment (USD)"]) * 100

# Display the RBF Financing Summary
print("RBF Financing Simulation Summary:")
print(rbf_data)

# Optionally save the results to a CSV file
output_file = "rbf_financing_summary_tuned.csv"
rbf_data.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

# Visualization of repayment years
plt.bar(rbf_data["Project"], rbf_data["Required Years"])
plt.xlabel("Projects")
plt.ylabel("Repayment Years")
plt.title("Repayment Years for RBF Financing (Tuned)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 1. ROI: Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(rbf_data["Project"], rbf_data["ROI (RBF) (%)"])
plt.title("ROI by Project (RBF Financing)")
plt.xlabel("Projects")
plt.ylabel("ROI (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Adjust Revenue Share Impact: Sensitivity Line Plot with Capped Years
share_percentages = [5, 10, 15, 20, 25]  # Different revenue share percentages
max_years = 20  # Cap repayment period at 20 years

plt.figure(figsize=(10, 6))
for share in share_percentages:
    repayment_years = rbf_data["Target Repayment (USD)"] / (rbf_data["Annual Revenue (USD)"] * (share / 100))
    repayment_years = repayment_years.apply(lambda x: min(x, max_years))  # Cap repayment period
    plt.plot(rbf_data["Project"], repayment_years, label=f"{share}% Revenue Share")

plt.title("Impact of Revenue Share on Repayment Period (Capped)")
plt.xlabel("Projects")
plt.ylabel("Repayment Period (Years)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Revenue vs. Repayment: Cumulative Line Chart
plt.figure(figsize=(10, 6))
for project in rbf_data["Project"]:
    project_data = rbf_data[rbf_data["Project"] == project]
    years = list(range(1, int(project_data["Required Years"].values[0]) + 1))
    cumulative_revenue = [project_data["Annual Revenue (USD)"].values[0] * year for year in years]
    cumulative_repayment = [
        min(cumulative_revenue[i], project_data["Target Repayment (USD)"].values[0])
        for i in range(len(years))
    ]
    plt.plot(years, cumulative_revenue, label=f"{project} - Revenue", linestyle="--")
    plt.plot(years, cumulative_repayment, label=f"{project} - Repayment")
plt.title("Revenue vs. Repayment (Cumulative)")
plt.xlabel("Years")
plt.ylabel("USD")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Cumulative Revenue Split: Stacked Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(
    rbf_data["Project"],
    rbf_data["Total Revenue (RBF Period) (USD)"] - rbf_data["Target Repayment (USD)"],
    label="Net Revenue After Repayment",
)
plt.bar(rbf_data["Project"], rbf_data["Target Repayment (USD)"], label="Repayment Target")
plt.title("Cumulative Revenue Split by Project")
plt.xlabel("Projects")
plt.ylabel("USD")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()