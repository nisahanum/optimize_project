import matplotlib.pyplot as plt

# Data simulasi hasil evolusi per generasi
generations = [0, 10, 40, 80, 90]
Z1_values = [1452.98, 1748.46, 1748.46, 1748.46, 1748.46]
Z2_values = [55.87, 43.08, 27.31, 23.65, 23.65]
Z3_values = [1008.49, 1185.44, 1185.44, 1185.44, 1185.44]

# Visualisasi garis per fungsi objektif
plt.figure(figsize=(10, 6))
plt.plot(generations, Z1_values, marker='o', label='Z1 - Strategic Value', color='skyblue')
plt.plot(generations, Z2_values, marker='o', label='Z2 - Risk-Adjusted Cost', color='salmon')
plt.plot(generations, Z3_values, marker='o', label='Z3 - Total Synergy', color='lightgreen')

plt.title("Evolusi Nilai Fungsi Objektif IFPOM per Generasi (MOEA/D)", fontsize=14)
plt.xlabel("Generasi", fontsize=12)
plt.ylabel("Nilai Fungsi Objektif", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
