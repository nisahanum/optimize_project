import matplotlib.pyplot as plt
import numpy as np

labels = [
    'Komunikasi Efektif',
    'Keterlibatan Sponsor\n/Manajemen Senior',
    'Estimasi Akurat',
    'Kepemilikan Proyek\nyang Jelas',
    'Keselarasan PMO\ndengan Strategi'
]

# Nilai keberhasilan
success_factors = [68, 77, 60, 77, 60]
# Nilai kegagalan sebagai komplemen (100 - keberhasilan)
failure_factors = [32, 23, 40, 23, 40]

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

success_factors += success_factors[:1]
failure_factors += failure_factors[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.plot(angles, success_factors, color='green', linewidth=2, label='Faktor Keberhasilan')
ax.fill(angles, success_factors, color='green', alpha=0.25)

ax.plot(angles, failure_factors, color='red', linewidth=2, label='Faktor Kegagalan')
ax.fill(angles, failure_factors, color='red', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)

# Menambahkan label nilai pada setiap titik data untuk kedua garis
for i in range(num_vars):
    ax.text(angles[i], success_factors[i] + 3, f"{success_factors[i]}%", color='green', fontsize=10, fontweight='bold', ha='center')
    ax.text(angles[i], failure_factors[i] - 7, f"{failure_factors[i]}%", color='red', fontsize=10, fontweight='bold', ha='center')

plt.title('Radar Chart: Faktor Keberhasilan vs. Kegagalan Proyek ICT', fontsize=14, fontweight='bold', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()
