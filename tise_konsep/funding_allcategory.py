import matplotlib.pyplot as plt
import numpy as np

# Categories and funding types
benefit_groups = ['Operational\nEfficiency', 'Customer\nExperience', 'Business\nCulture']
funding_types = ['Direct\nInvestment', 'Soft\nLoan', 'Vendor\nFinancing', 'Balanced\nMix']

# Simulated data for Z1 (Strategic Value) and Z2 (Cost)
Z1_scores = np.array([
    [80, 75, 50, 85],  # Operational Efficiency
    [70, 72, 48, 88],  # Customer Experience
    [68, 70, 45, 82]   # Business Culture
])

Z2_costs = np.array([
    [2.0, 2.2, 3.2, 2.5],
    [2.1, 2.3, 3.3, 2.4],
    [2.2, 2.4, 3.4, 2.6]
])

x = np.arange(len(benefit_groups))
width = 0.18

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Z1 (Strategic Value)
for i in range(len(funding_types)):
    ax1.bar(x + (i - 1.5)*width, Z1_scores[:, i], width, label=funding_types[i])

ax1.set_ylabel('Strategic Value (Z1)')
ax1.set_title('Comparative Performance of Funding Types across Project Benefit Groups')
ax1.set_xticks(x)
ax1.set_xticklabels(benefit_groups)
ax1.legend(title='Funding Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
