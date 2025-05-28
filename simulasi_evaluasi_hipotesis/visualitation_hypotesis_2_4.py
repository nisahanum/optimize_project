# visualitation_hypotesis_2_4_small.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
csv_path = "/Users/nisahanum/Documents/S3/simulationopt/optimize_project/s2_4_monte_carlo_results_2.csv"
df = pd.read_csv(csv_path)

# Convert axis values to string
df['ThetaCap'] = df['ThetaCap'].astype(str)
df['SynergyWeight'] = df['SynergyWeight'].astype(str)

# Create pivot tables
pivot_z1 = df.pivot(index="SynergyWeight", columns="ThetaCap", values="Z1")
pivot_z2 = df.pivot(index="SynergyWeight", columns="ThetaCap", values="Z2")
pivot_z3 = df.pivot(index="SynergyWeight", columns="ThetaCap", values="Z3")

# Plot smaller heatmaps
plt.figure(figsize=(10, 9))  # Smaller overall figure

plt.subplot(3, 1, 1)
sns.heatmap(pivot_z1, annot=True, cmap="Blues", fmt=".1f", annot_kws={"size": 8})
plt.title("Z₁ (Strategic Value) vs θ and λ", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(3, 1, 2)
sns.heatmap(pivot_z2, annot=True, cmap="Reds", fmt=".2f", annot_kws={"size": 8})
plt.title("Z₂ (Risk-Adjusted Cost) vs θ and λ", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(3, 1, 3)
sns.heatmap(pivot_z3, annot=True, cmap="Greens", fmt=".1f", annot_kws={"size": 8})
plt.title("Z₃ (Synergy Value) vs θ and λ", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.tight_layout()
plt.savefig("heatmaps_s2_4_small.png", dpi=300)
plt.show()
