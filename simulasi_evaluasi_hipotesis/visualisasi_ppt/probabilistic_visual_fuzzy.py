import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Flowchart components: (title, description, x_position)
components = [
    ("Project Cost Estimation", "Biaya proyek sebagai\nFuzzy Triangular Number\n(min, mean, max)", 0.5),
    ("Expected Value Calculation", "E(C̃) = (min + 2×mean + max) / 4", 3.5),
    ("Deviation Measurement", "Deviasi dari nilai fuzzy\nmenggunakan ξ, η₁, η₂", 6.5),
    ("RPP Objective Construction", "Z₂ = E(C̃) + penalty deviation", 9.5),
    ("IFPOM Optimization", "Digunakan dalam portofolio:\nZ₁, Z₂, Z₃", 12.5)
]

# Draw flowchart boxes
for title, desc, x in components:
    ax.add_patch(patches.FancyBboxPatch(
        (x, 2.2), 2.8, 2.0,
        boxstyle="round,pad=0.03",
        edgecolor='black', facecolor='#e8f4fa'
    ))
    ax.text(x + 1.4, 3.8, title, ha='center', fontsize=11, weight='bold')
    ax.text(x + 1.4, 3.1, desc, ha='center', fontsize=9)

# Arrows between boxes
for i in range(len(components) - 1):
    x_start = components[i][2] + 2.8
    x_end = components[i + 1][2]
    ax.annotate('', xy=(x_end, 3.2), xytext=(x_start, 3.2),
                arrowprops=dict(arrowstyle='->', lw=2))

# Title
ax.text(7.5, 5.2, "Flowchart: Robust Possibilistic Programming (RPP) untuk Ketidakpastian Biaya di IFPOM",
        ha='center', fontsize=14, weight='bold')

# Show the chart
plt.tight_layout()
plt.show()
