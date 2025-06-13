import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Senior Management Support']
high_performance = [77]
low_performance = [44]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(6,5))
rects1 = ax.bar(x - width/2, high_performance, width, label='Kinerja Tinggi', color="#6BD3E8")
rects2 = ax.bar(x + width/2, low_performance, width, label='Kinerja Rendah', color='#FF5722')

# Label dan judul
ax.set_ylabel('Persentase (%)')
ax.set_title('Perbandingan Keterlibatan Senior Eksekutif')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Tambahkan label nilai pada bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 100)
plt.show()
