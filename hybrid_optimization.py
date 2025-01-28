import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define global financing attributes
loan_interest_rate = 5  # Cost of loan in percentage
equity_cost = 8         # Cost of equity in percentage
rbf_cost = 12           # Cost of RBF in percentage
loan_term_years = 5     # Loan repayment term in years

# Define or load the sample_projects dataset
sample_projects = pd.DataFrame({
    'Project': ['Sample Project 1', 'Sample Project 2', 'Sample Project 3'],
    'Initial Investment (USD)': [1000000, 2000000, 3000000],
    'Annual Revenue (USD)': [300000, 600000, 900000],
    'Total Outflows (USD)': [1500000, 2800000, 4000000],
})

# Example analysis loop
for _, project in sample_projects.iterrows():
    print(f"Analyzing project: {project['Project']}")


# Define a function to analyze scenarios including equity and RBF
def analyze_equity_rbf_mix(project):
    equity_rbf_results = []
    
    # Iterate over possible equity and RBF proportions
    for equity_percent in range(0, 101, 10):  # Equity from 0% to 100%
        for rbf_percent in range(0, 101 - equity_percent, 10):  # RBF from 0% to remaining
            debt_percent = 100 - equity_percent - rbf_percent

            # Calculate Blended WACC
            blended_wacc = (
                (debt_percent / 100 * loan_interest_rate) +
                (equity_percent / 100 * equity_cost) +
                (rbf_percent / 100 * rbf_cost)
            )

            # Calculate Loan Repayment
            loan_amount = project['Initial Investment (USD)'] * (debt_percent / 100)
            annual_loan_repayment = loan_amount * (1 + (loan_interest_rate / 100)) / loan_term_years

            # Calculate Adjusted Outflows
            adjusted_outflows = (
                annual_loan_repayment +
                (project['Total Outflows (USD)'] * (equity_percent / 100 * (1 + equity_cost / 100))) +
                (project['Annual Revenue (USD)'] * (rbf_percent / 100))
            )

            # Calculate Net Cash Flow
            net_cash_flow = project['Annual Revenue (USD)'] - adjusted_outflows

            # Calculate ROI
            roi = (net_cash_flow / project['Initial Investment (USD)']) * 100

            # Store results
            equity_rbf_results.append({
                'Equity (%)': equity_percent,
                'RBF (%)': rbf_percent,
                'Debt (%)': debt_percent,
                'Blended WACC (%)': blended_wacc,
                'Net Cash Flow (USD)': net_cash_flow,
                'ROI (%)': roi,
            })

    return equity_rbf_results

# Analyze equity and RBF mix for all projects in the sample
equity_rbf_analysis_results = []
for _, project in sample_projects.iterrows():
    equity_rbf_analysis = analyze_equity_rbf_mix(project)
    for result in equity_rbf_analysis:
        result['Project'] = project['Project']
    equity_rbf_analysis_results.extend(equity_rbf_analysis)

# Convert equity and RBF analysis results to DataFrame
equity_rbf_analysis_df = pd.DataFrame(equity_rbf_analysis_results)

# Display the equity and RBF analysis results
print(equity_rbf_analysis_df)

#equity_rbf_analysis_df.to_csv("equity_rbf_analysis.csv", index=False)
#print("Equity and RBF mix analysis saved to 'equity_rbf_analysis.csv'")

# Visualization: Impact of Financing Mix on ROI
plt.figure(figsize=(12, 6))

# Line plot for ROI against Debt, Equity, and RBF proportions
palette = ["blue", "green", "orange"]
sns.lineplot(data=equity_rbf_analysis_df, x='Debt (%)', y='ROI (%)', label='Debt (%) Impact', marker='o',color=palette[0])
sns.lineplot(data=equity_rbf_analysis_df, x='Equity (%)', y='ROI (%)', label='Equity (%) Impact', marker='o',color=palette[1])
sns.lineplot(data=equity_rbf_analysis_df, x='RBF (%)', y='ROI (%)', label='RBF (%) Impact', marker='o',color=palette[2])

# Add chart details
plt.title('Impact of Financing Mix on ROI')
plt.xlabel('Financing Proportion (%)')
plt.ylabel('ROI (%)')
plt.legend(title='Financing Type')
plt.grid(True)
plt.tight_layout()
plt.show()
