import pandas as pd

# Load the job-level dataset
df_jobs = pd.read_csv("C:/Users/nisahanum/Documents/cobagit/optimize_project/ifpom_dataset_mode1.csv")

# Group by project_id and compute aggregated values
# - Average for SVS, Risk, Fuzzy costs, and funding
# - Count of jobs per project

project_agg = df_jobs.groupby("project_id").agg({
    "job_id": "count",
    "SVS": "mean",
    "Risk": "mean",
    "Fuzzy_Cost_Min": "mean",
    "Fuzzy_Cost_Likely": "mean",
    "Fuzzy_Cost_Max": "mean",
    "Alpha": "mean",
    "Beta": "mean",
    "Theta": "mean",
    "Gamma": "mean",
    "Delta": "mean"
}).reset_index()

project_agg.rename(columns={"job_id": "num_jobs"}, inplace=True)

#print(project_agg.head())

project_agg.to_csv("aggregated_ifpom_projects.csv", index=False)
print("✅ Saved to 'aggregated_ifpom_projects.csv'")

