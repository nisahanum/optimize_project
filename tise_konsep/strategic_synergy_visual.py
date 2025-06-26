import matplotlib.pyplot as plt
import numpy as np

# Data from the synergy simulation table
scenarios = ['S1.1\nNo Synergy', 'S1.2\nSame-Period', 'S1.3\nCross-Period', 'S1.4\nFull Synergy']
Z1_values = [513.1, 916.1, 513.1, 1200.0]  # Value Gain
Z2_costs = [2.6, 2.3, 2.6, 3.0]           # Cost (scaled)
Z3_synergy = [0.0, 539.0, 80.0, 759.6]    # Collaboration (Synergy)

# Scale costs to match visual comparison
Z2_scaled = [z2 * 100 for z2 in Z2_costs]  # scale for plotting alongside other metrics

x = np.arange(len(scenarios))  # label locations
width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width, Z1_values, width, label='Value Gain', color='royalblue')
bar2 = ax.bar(x, Z2_scaled, width, label='Cost (scaled)', color='salmon')
bar3 = ax.bar(x + width, Z3_synergy, width, label='Synergy', color='gold')

# Add labels and title
ax.set_xlabel('Synergy Scenario')
ax.set_ylabel('Score / Scaled Cost')
ax.set_title('Comparative Simulation Results under Varying Synergy Scenarios (λ = 7.5)')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()

# Add value labels on top
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
