import matplotlib.pyplot as plt
import numpy as np

# Radar chart labels
labels = ["Cost Impact", "Control", "Risk Exposure"]
num_vars = len(labels)

# Funding type scores: [Cost Impact, Control, Risk Exposure]
data = {
    "α (Equity)": [1, 5, 1],        # Marginal cost, full control, zero liability
    "β (Soft Loan)": [2, 3, 3],     # Preferential cost, shared control, partial liability
    "θ (Vendor)": [5, 1, 5],        # Locked-up cost, third-party control, full liability
    "γ (Grant)": [2, 3, 1],         # Preferential cost, shared control, zero liability
    "δ (PPP)": [4, 2, 5]            # Locked-up cost, partial control, high risk
}

# Convert to angles for radar
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Set up the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each funding type
for label, scores in data.items():
    values = scores + scores[:1]
    ax.plot(angles, values, label=label)
    ax.fill(angles, values, alpha=0.1)

# Set the chart labels and layout
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 5)
ax.set_title("Jenis Pendanaan Berdasarkan Biaya, Pengendalian, dan Risiko", size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
