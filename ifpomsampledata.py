import pandas as pd

# Contoh data
data = {
    "Project_ID": [1, 2, 3, 4, 5],
    "Expected_Return (%)": [20, 25, 30, 18, 22],
    "Risk (%)": [15, 20, 25, 12, 18],
    "Funding_Needed (USD)": [100000, 150000, 200000, 120000, 80000],
    "Direct_Funding (%)": [50, 30, 20, 40, 60],
    "Joint_Venture (%)": [30, 40, 50, 30, 20],
    "Loan (%)": [20, 30, 30, 30, 20],
    "Synergy_Score": [0.8, 0.7, 0.9, 0.6, 0.5],
    "Social_Impact_Score": [70, 60, 80, 50, 55],
    "Dependency": [0, 1, 0, 0, 1],
}

df = pd.DataFrame(data)

# Menampilkan DataFrame secara langsung
print(df)

# Save to a CSV file
df.to_csv("ifpomdataset.csv", index=False)
print("Results saved to ifpomdataset.csv")