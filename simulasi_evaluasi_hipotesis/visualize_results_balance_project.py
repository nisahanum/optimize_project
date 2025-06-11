import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("selected_projects_summary.csv")

# === Ambil 3 Solusi Balanced yang Unik (non-duplicate) ===
top_3_unique = df.sort_values(
    by=['Z1 (Strategic Value)', 'Z3 (Synergy)', 'Z2 (Financial Cost)'],
    ascending=[False, False, True]
).drop_duplicates(subset=['Z1 (Strategic Value)', 'Z2 (Financial Cost)', 'Z3 (Synergy)'])

top_3_unique = top_3_unique.head(3).reset_index(drop=True)

# === Normalisasi nilai untuk Radar Chart ===
top_3_unique['Z2 (Normalized)'] = top_3_unique['Z2 (Financial Cost)'].max() - top_3_unique['Z2 (Financial Cost)']
top_3_unique['Z2 (Normalized)'] = top_3_unique['Z2 (Normalized)'] / top_3_unique['Z2 (Normalized)'].max()
top_3_unique['Z1 (Normalized)'] = top_3_unique['Z1 (Strategic Value)'] / top_3_unique['Z1 (Strategic Value)'].max()
top_3_unique['Z3 (Normalized)'] = top_3_unique['Z3 (Synergy)'] / top_3_unique['Z3 (Synergy)'].max()

# === Radar Chart Setup ===
radar_data_unique = top_3_unique[['Z1 (Normalized)', 'Z2 (Normalized)', 'Z3 (Normalized)']].values
labels_unique = top_3_unique['Solution'].tolist()
categories = ['Z1 (Strategic Value)', 'Z2 (Financial Cost)', 'Z3 (Synergy)']

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# === Plot Radar Chart ===
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
for idx, row in enumerate(radar_data_unique):
    values = row.tolist() + [row[0]]
    ax.plot(angles, values, label=labels_unique[idx], marker='o')
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
plt.title("3 Solusi seimbang terbaik")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=1)
plt.tight_layout()
plt.show()
