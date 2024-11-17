import pandas as pd
# Simulation for ICT Project Milestones with Hybrid Financing

# Project Details
total_funding = 5_000_000  # Total funding in USD
loan_1_percentage = 0.30  # Loan 1 share
direct_investment_percentage = 0.50  # Direct Investment share
rbf_percentage = 0.20  # RBF share

# Allocate funds based on percentages
loan_1_funding = total_funding * loan_1_percentage
direct_investment_funding = total_funding * direct_investment_percentage
rbf_funding = total_funding * rbf_percentage

# Milestones and funding requirements
milestones = [
    {"milestone": "Prototype Development", "timeframe": "Month 0-6", "source": "Direct Investment", "amount": 1_000_000},
    {"milestone": "Beta Launch", "timeframe": "Month 6-12", "source": "Loan 1", "amount": 500_000},
    {"milestone": "Product Launch", "timeframe": "Month 12-18", "source": "Direct Investment", "amount": 1_000_000},
    {"milestone": "Revenue Growth Phase", "timeframe": "Month 18-24", "source": "RBF", "amount": 1_000_000},
    {"milestone": "Expansion", "timeframe": "Month 24-36", "source": "Loan 1 + Direct Investment", "amount": 1_500_000},
]

# Generate repayment and cash flow projections
loan_interest_rate = 0.08  # Annual interest rate
loan_term = 5  # Loan term in years
rbf_revenue_share = 0.10  # Revenue-based financing repayment share
projected_annual_revenue = [3_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000]  # Projected revenue over 5 years

# Simulate cash flow impact
loan_1_annual_repayment = loan_1_funding / loan_term + (loan_1_funding * loan_interest_rate)
rbf_annual_payment = [revenue * rbf_revenue_share for revenue in projected_annual_revenue]

# Create milestone DataFrame
milestones_df = pd.DataFrame(milestones)

# Cash Flow Projections
cash_flow_data = {
    "Year": [1, 2, 3, 4, 5],
    "Loan_1_Repayment": [loan_1_annual_repayment] * loan_term,
    "RBF_Payment": rbf_annual_payment,
    "Direct_Investment_Return": [
        0, 
        direct_investment_funding * 0.2, 
        direct_investment_funding * 0.4, 
        direct_investment_funding * 0.6, 
        direct_investment_funding * 0.8
    ],  # Incremental ROI-based returns
}

cash_flow_df = pd.DataFrame(cash_flow_data)
cash_flow_df["Total_Cash_Outflow"] = (
    cash_flow_df["Loan_1_Repayment"] + cash_flow_df["RBF_Payment"] + cash_flow_df["Direct_Investment_Return"]
)

# Display dataframes directly in the terminal/console
print("=== ICT Project Milestones ===")
print(milestones_df)

print("\n=== ICT Project Cash Flow Projections ===")
print(cash_flow_df)

# Save dataframes to CSV files for further analysis
milestones_df.to_csv("ict_project_milestones.csv", index=False)
cash_flow_df.to_csv("ict_project_cash_flow_projections.csv", index=False)

print("\nData saved to 'ict_project_milestones.csv' and 'ict_project_cash_flow_projections.csv'.")
