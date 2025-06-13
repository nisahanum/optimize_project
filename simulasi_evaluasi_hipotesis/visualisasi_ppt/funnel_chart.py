import matplotlib.pyplot as plt

# Data funnel
stages = ['Total Proyek', 'Proyek Sesuai Tujuan', 'Proyek Gagal']
values_before = [100, 38, 62]  # Sebelum integrasi PMO strategis (%)
values_after = [100, 85, 5]    # Setelah integrasi PMO strategis (%)

fig, ax = plt.subplots(figsize=(9,6))

# Bar atas - Sebelum PMO Strategis
bars1 = ax.barh(0.7, values_before[1], height=0.15, color='#FF7043', label='Proyek Sesuai Tujuan (Sebelum)')
bars2 = ax.barh(0.7, values_before[2], height=0.15, left=values_before[1], color='#E64A19', label='Proyek Gagal (Sebelum)')

# Bar bawah - Setelah PMO Strategis
bars3 = ax.barh(0.4, values_after[1], height=0.15, color='#4CAF50', label='Proyek Sesuai Tujuan (Setelah)')
bars4 = ax.barh(0.4, values_after[2], height=0.15, left=values_after[1], color='#1B5E20', label='Proyek Gagal (Setelah)')

# Label dan ticks
ax.set_yticks([0.4, 0.7])
ax.set_yticklabels(['Setelah PMO Strategis', 'Sebelum PMO Strategis'], fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_xlabel('Persentase Proyek (%)', fontsize=12)
ax.set_title('Dampak Project Management Office (PMO) Strategis terhadap Keberhasilan Proyek', fontsize=16, fontweight='bold')

# Menambahkan nilai persentase di dalam bar dengan font lebih besar dan bold
def autolabel(bars):
    for bar in bars:
        width = bar.get_width()
        ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)
autolabel(bars4)

# Legend di bawah dengan frame dan font lebih besar
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, fontsize=12)
frame = leg.get_frame()
frame.set_edgecolor('black')
frame.set_linewidth(1)

plt.tight_layout()
plt.show()
