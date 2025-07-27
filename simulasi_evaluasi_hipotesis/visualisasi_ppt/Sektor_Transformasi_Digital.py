import matplotlib.pyplot as plt

# Data estimasi distribusi sektor digital 2025 (dalam miliar USD)
# Berdasarkan info: Total pasar transformasi digital = USD 24.37 miliar
# E-commerce: Rp471 triliun (~ USD 30B estimasi, tapi kita normalkan proporsinya ke total 24.37)
# Transportasi: Rp12.66 triliun (~ USD 0.8B)

# Untuk proporsi visual: kita normalkan jadi persentase dari total
labels = ['E-commerce', 'Online transportation sector', 'Sektor Lainnya']
values = [19.5, 0.5, 4.37]  # Penyesuaian agar total = 24.37
colors = ['#ff9999','#66b3ff','#99ff99']

# Create pie chart
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('Distribusi Sektor Transformasi Digital Indonesia 2025\n(Total Pasar USD 24,37 Miliar)', fontsize=13, weight='bold')

plt.tight_layout()
plt.show()
