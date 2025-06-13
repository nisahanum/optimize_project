import matplotlib.pyplot as plt

# Data roadmap
years = ['2023', '2024', '2025']
stages = ['Studi Literatur', 'Pengumpulan Data', 'Pengembangan Model', 'Eksperimen & Simulasi', 'Publikasi & Luaran']

# Aktivitas tiap tahun (dalam bentuk teks ringkas)
activities = {
    '2023': [
        'Kajian teori project interdependency, synergy, multi-objective optimization',
        'Kumpulkan dataset proyek ICT dan scoring expert',
        'Formulasi fungsi objektif IFPOM, implementasi fuzzy logic',
        'Preliminary experiment dengan data statis',
        'Proposal disertasi, paper konferensi'
    ],
    '2024': [
        'Review algoritma MOEA/D, penyempurnaan teori',
        'Pra-pemrosesan data, analisis korelasi variabel',
        'Pengembangan algoritma MOEA/D hybrid, simulasi awal',
        'Simulasi multi-skenario, evaluasi kinerja',
        'Paper jurnal Q1 IEEE Access, workshop'
    ],
    '2025': [
        'Integrasi studi kasus dan validasi literatur',
        'Update dataset dinamis',
        'Optimasi dan tuning model',
        'Finalisasi eksperimen, validasi expert',
        'Laporan akhir disertasi, publikasi, seminar'
    ]
}

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Posisi vertikal tiap stage
y_positions = range(len(stages), 0, -1)

for i, stage in enumerate(stages):
    ax.text(0.05, y_positions[i], stage, fontsize=12, fontweight='bold', ha='left', va='center')

# Menambahkan aktivitas untuk setiap tahun
x_offsets = [0.3, 0.6, 0.9]
for j, year in enumerate(years):
    for i, activity in enumerate(activities[year]):
        ax.text(x_offsets[j], y_positions[i], activity, fontsize=10, ha='left', va='center', wrap=True)

# Menambahkan header tahun
for j, year in enumerate(years):
    ax.text(x_offsets[j], len(stages) + 0.5, year, fontsize=14, fontweight='bold', ha='center')

plt.title('Peta Jalan Penelitian IFPOM (2023-2025)', fontsize=16, fontweight='bold', pad=20)
plt.show()
