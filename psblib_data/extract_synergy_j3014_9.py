
import pandas as pd

# Load the synergy matrix
synergy_df = pd.read_csv("C:/Users/nisahanum/Documents/cobagit/optimize_project/synergy_matrix.csv", index_col=0)
synergy_df.columns = synergy_df.columns.astype(str)
synergy_df.index = synergy_df.index.astype(str)

# Load the IFPOM dataset
df_ifpom = pd.read_csv("ifpom_dataset_mode1.csv")

# Filter for project j3014_9 and Mode 1 only
df_j3014_9_mode1 = df_ifpom[(df_ifpom['project_id'] == 'j3014_9') & (df_ifpom['mode_id'] == 1)]

# Extract job_ids and match against synergy matrix
job_ids_mode1 = df_j3014_9_mode1['job_id'].astype(str).tolist()
usable_keys = [jid for jid in job_ids_mode1 if jid in synergy_df.columns and jid in synergy_df.index]

# Extract the synergy submatrix
synergy_submatrix_mode1 = synergy_df.loc[usable_keys, usable_keys]

# Save the result
synergy_submatrix_mode1.to_csv("synergy_matrix_j3014_9_mode1.csv")
print("✅ Saved synergy_matrix_j3014_9_mode1.csv")
