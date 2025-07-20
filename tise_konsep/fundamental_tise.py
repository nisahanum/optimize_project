import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 8))

# === Triune Intelligence Main Box ===
ax.add_patch(patches.Rectangle((2, 7), 6, 1, facecolor='lightgrey', edgecolor='black'))
ax.text(5, 7.45, 'Triune Intelligence', ha='center', va='center', fontsize=12, weight='bold')

# Sub-boxes
ax.add_patch(patches.Rectangle((2.2, 7.1), 1.6, 0.6, facecolor='#f4cccc', edgecolor='black'))
ax.text(3, 7.4, 'Human\n(Ethos)', ha='center', va='center', fontsize=10)

ax.add_patch(patches.Rectangle((4, 7.1), 1.6, 0.6, facecolor='#c9daf8', edgecolor='black'))
ax.text(4.8, 7.4, 'AI\n(Logic)', ha='center', va='center', fontsize=10)

ax.add_patch(patches.Rectangle((5.8, 7.1), 1.6, 0.6, facecolor='#d9ead3', edgecolor='black'))
ax.text(6.6, 7.4, 'Natural\n(Ecology)', ha='center', va='center', fontsize=10)

# === PSKVE Box ===
ax.add_patch(patches.Rectangle((2.5, 5.5), 5, 1, facecolor='#fce5cd', edgecolor='black'))
ax.text(5, 6.0, 'PSKVE Value Engine\n(P, S, K, V, E)', ha='center', va='center', fontsize=11, weight='bold')

# Arrow from Triune to PSKVE
ax.annotate('', xy=(5, 7), xytext=(5, 6.5), arrowprops=dict(facecolor='black', arrowstyle='->'))

# === SFE Box ===
ax.add_patch(patches.Rectangle((3, 3.5), 4, 1, facecolor='#d0e0e3', edgecolor='black'))
ax.text(5, 4.0, 'Smart Financing Engine\n(SFE ➝ IFPOM)', ha='center', va='center', fontsize=11, weight='bold')

# Arrow from PSKVE to SFE
ax.annotate('', xy=(5, 5.5), xytext=(5, 4.5), arrowprops=dict(facecolor='black', arrowstyle='->'))

# === Outputs ===
ax.text(5, 2.5, 'Outputs: Z₁ (Strategic Value), Z₂ (Risk-Adjusted Cost), Z₃ (Synergy)', ha='center', fontsize=10, style='italic')

# Set limits and hide axes
ax.set_xlim(1, 9)
ax.set_ylim(2, 9)
ax.axis('off')

plt.show()
