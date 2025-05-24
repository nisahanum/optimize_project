# Re-import necessary libraries due to kernel reset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Reload the CSV file containing project-level features
csv_path = "C:/Users/nisahanum/Documents/cobagit/optimize_project/aggregated_ifpom_projects.csv"  # Assumed correct file from earlier step
df_project = pd.read_csv(csv_path)

# Define the feature columns used to compute project similarity
feature_cols = ['SVS', 'Risk', 'Fuzzy_Cost_Likely', 'Alpha', 'Beta', 'Theta', 'Gamma', 'Delta']
feature_matrix = df_project[feature_cols]



# Normalize the features to range [0, 1]
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(feature_matrix)

# Calculate cosine similarity matrix on normalized data
cosine_sim_matrix = cosine_similarity(normalized_features)


# Create a DataFrame for the synergy matrix
project_ids = df_project['project_id'].tolist()
synergy_df = pd.DataFrame(cosine_sim_matrix, index=project_ids, columns=project_ids)

# Display the synergy matrix
print(synergy_df.round(3))

# Save to CSV
synergy_df.to_csv("synergy_matrix_cosine_normalized.csv")
print("\n📁 Saved to 'synergy_matrix_cosine_normalized.csv'")
