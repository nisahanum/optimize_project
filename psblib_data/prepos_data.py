import pandas as pd
import numpy as np
import ast

# Load your preprocessed CSV file (replace with your actual path if running locally)
csv_path = "C:/Users/nisahanum/Documents/cobagit/optimize_project/psblib_data/data/j30.mm/ifpom_projects_50_60.csv"
df = pd.read_csv(csv_path)

# Convert stringified lists in 'resources' and 'successors' to actual lists
df['resources'] = df['resources'].apply(ast.literal_eval)
df['successors'] = df['successors'].apply(ast.literal_eval)

# Filter to only Mode 1 per job
df_mode1 = df[df['mode_id'] == 1].copy()

# Generate fuzzy cost based on duration
df_mode1["Fuzzy_Cost_Min"] = (df_mode1["duration"] * 0.9).astype(int)
df_mode1["Fuzzy_Cost_Likely"] = df_mode1["duration"]
df_mode1["Fuzzy_Cost_Max"] = (df_mode1["duration"] * 1.2).astype(int)

# Synthesize SVS and Risk
np.random.seed(42)
df_mode1["SVS"] = np.random.randint(50, 101, size=len(df_mode1))
df_mode1["Risk"] = np.round(np.random.uniform(0.1, 0.4, size=len(df_mode1)), 2)

# Generate funding proportions: α, β, θ, γ, δ with θ capped at 0.4
funding = [np.random.dirichlet([1, 1, 1, 1, 1]) for _ in range(len(df_mode1))]
funding = np.array([[a, b, min(t, 0.4), g, d] for a, b, t, g, d in funding])
df_mode1["Alpha"] = np.round(funding[:, 0], 2)
df_mode1["Beta"] = np.round(funding[:, 1], 2)
df_mode1["Theta"] = np.round(funding[:, 2], 2)
df_mode1["Gamma"] = np.round(funding[:, 3], 2)
df_mode1["Delta"] = np.round(funding[:, 4], 2)

#print(df_mode1.head())

df_mode1.to_csv("ifpom_dataset_mode1.csv", index=False)
print("✅ IFPOM dataset saved to 'ifpom_dataset_mode1.csv'")


