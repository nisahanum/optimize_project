import pandas as pd

# Load the existing CSV
csv_path = "/Users/nisahanum/Documents/S3/simulationopt/optimize_project/s1_synergy_tuning_results.csv"
df = pd.read_csv(csv_path)

# Format Z2 to 3 decimal places
df["Z2"] = df["Z2"].round(3)

# Save the updated CSV
updated_path = "/Users/nisahanum/Documents/S3/simulationopt/optimize_project/s1_synergy_tuning_comma.csv"
df.to_csv(updated_path, index=False)

updated_path
