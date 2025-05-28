import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load the logged data from MOEA/D run ===
df_logbook = pd.read_csv("moead_logbook_output.csv")

# === Normalize each Z column using Min-Max scaling ===
df_logbook["Z1"] = (df_logbook["Z1"] - df_logbook["Z1"].min()) / (df_logbook["Z1"].max() - df_logbook["Z1"].min())
df_logbook["Z2"] = (df_logbook["Z2"] - df_logbook["Z2"].min()) / (df_logbook["Z2"].max() - df_logbook["Z2"].min())
df_logbook["Z3"] = (df_logbook["Z3"] - df_logbook["Z3"].min()) / (df_logbook["Z3"].max() - df_logbook["Z3"].min())

# === Reshape to long format for Seaborn ===
df_violin = pd.melt(
    df_logbook,
    id_vars=["Generation"],
    value_vars=["Z1", "Z2", "Z3"],
    var_name="Objective",
    value_name="Normalized Value"
)

# === Optional: format generation labels for x-axis clarity ===
df_violin["Generation"] = df_violin["Generation"].apply(lambda g: f"Gen {g}")

# === Plot violin with box inside ===
plt.figure(figsize=(16, 6))
sns.violinplot(
    x="Generation",
    y="Normalized Value",
    hue="Objective",
    data=df_violin,
    inner="box",
    palette="Set2"
)
plt.title("Normalized Distribution of Z₁, Z₂, and Z₃ Across Generations")
plt.ylabel("Normalized Objective Value (0–1)")
plt.xlabel("Generation")
plt.grid(True)
plt.legend(title="Objective")
plt.tight_layout()
plt.show()
