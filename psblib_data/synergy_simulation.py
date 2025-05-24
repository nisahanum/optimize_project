import pandas as pd
import numpy as np

# Use the existing df_mode1 from your IFPOM-enhanced data
csv_path = "C:/Users/nisahanum/Documents/cobagit/optimize_project/psblib_data/data/j30.mm/ifpom_projects_50_60.csv"
df = pd.read_csv(csv_path)

# Ensure 'successors' column contains actual lists
df['successors'] = df['successors'].apply(eval if isinstance(df['successors'].iloc[0], str) else lambda x: x)

# Build synergy matrix δᵢⱼ based on Jaccard similarity of successors
n = len(df)
synergy_matrix = np.zeros((n, n))

for i in range(n):
    succ_i = set(df.loc[i, 'successors'])
    for j in range(i + 1, n):
        succ_j = set(df.loc[j, 'successors'])
        intersection = len(succ_i & succ_j)
        union = len(succ_i | succ_j)
        score = intersection / union if union > 0 else 0
        synergy_matrix[i, j] = synergy_matrix[j, i] = score

# Display as DataFrame
job_ids = df['job_id'].astype(str).tolist()
synergy_df = pd.DataFrame(synergy_matrix, index=job_ids, columns=job_ids)

# Save or view
synergy_df.to_csv("synergy_matrix.csv", index=True)
print("✅ Synergy matrix saved to 'synergy_matrix.csv'")
