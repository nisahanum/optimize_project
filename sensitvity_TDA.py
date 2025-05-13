import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define risk factors
financial_risks = ["Funding Sources", "Leverage Ratio", "Negative Cash Flow"]
technical_risks = ["Tech Readiness", "Complexity", "Failure Risk"]

# Assign corrected impact values
impact_financial = [0.25, 0.40, 0.30]  # Only financial risks
impact_technical = [0.40, 0.35, 0.30]  # Only technical risks

# Normalize scores
total_financial = sum(impact_financial)
total_technical = sum(impact_technical)

normalized_financial = [x / total_financial for x in impact_financial]
normalized_technical = [x / total_technical for x in impact_technical]

# Create DataFrames
df_financial = pd.DataFrame({
    "Risk Factor": financial_risks,
    "Impact on R_financial": impact_financial,
    "Normalized Financial Score": normalized_financial
}).sort_values(by="Impact on R_financial", ascending=True)

df_technical = pd.DataFrame({
    "Risk Factor": technical_risks,
    "Impact on R_technical": impact_technical,
    "Normalized Technical Score": normalized_technical
}).sort_values(by="Impact on R_technical", ascending=True)

# Plot Tornado Diagram for Financial Risks
plt.figure(figsize=(8, 6))
plt.barh(df_financial["Risk Factor"], df_financial["Impact on R_financial"], 
         color="lightcoral", edgecolor="black")
plt.xlabel("Impact on Financial Risk")
plt.ylabel("Financial Risk Factors")
plt.title("Corrected Tornado Diagram for Financial Risks")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Plot Tornado Diagram for Technical Risks
plt.figure(figsize=(8, 6))
plt.barh(df_technical["Risk Factor"], df_technical["Impact on R_technical"], 
         color="royalblue", edgecolor="black")
plt.xlabel("Impact on Technical Risk")
plt.ylabel("Technical Risk Factors")
plt.title("Corrected Tornado Diagram for Technical Risks")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()
