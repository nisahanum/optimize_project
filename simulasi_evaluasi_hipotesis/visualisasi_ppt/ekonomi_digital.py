import matplotlib.pyplot as plt

# Data and sources (only 2 as requested)
labels = ['USD 130M', 'USD 146M']
values = [130, 146]
colors = ['#1f77b4', '#2ca02c']
sources = ['pwc-finance-tranformation', 'neotri']

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
bars = ax.bar(labels, values, color=colors)

# Add data labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2, f'{height}', ha='center', va='bottom', fontsize=11)

# Title and labels
ax.set_title('Proyeksi Ekonomi Digital Tahun 2025', fontsize=14, weight='bold')
ax.set_ylabel('Nilai (dalam miliar USD)')
ax.set_ylim(0, 160)

# Add simplified legend for 2 sources
custom_legend = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(colors))]
ax.legend(custom_legend, sources, title='Sumber Data', loc='upper left')

plt.tight_layout()
plt.show()
