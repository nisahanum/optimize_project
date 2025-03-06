import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Membuat dataset proyek dengan variabel state dan pendanaan awal
np.random.seed(42)
n_projects = 50

projects_funding = pd.DataFrame({
    'Project_ID': np.arange(1, n_projects + 1),
    'Current_State': np.random.choice(['Initial Planning', 'Execution', 'Scaling Up', 'Restructuring', 'Exit/Maturity'], n_projects),
    'Performance_Score': np.random.uniform(0.5, 1.5, n_projects),  # Indikator keberhasilan proyek
    'Financial_Health': np.random.uniform(0, 1, n_projects),  # Indikator stabilitas keuangan proyek
    'Risk_Level': np.random.uniform(0, 1, n_projects),  # Risiko proyek
    'Funding_Type': np.random.choice(['Venture Capital', 'Debt Financing', 'Revenue-Based Financing', 'Public-Private Partnerships', 'Equity Investment'], n_projects)
})

# Aturan perubahan strategi pendanaan berdasarkan transisi state proyek
def determine_funding_strategy(row):
    if row['Current_State'] == 'Initial Planning':
        if row['Performance_Score'] > 1.2 and row['Financial_Health'] > 0.6:
            return 'Debt Financing'
        else:
            return 'Venture Capital'
    
    elif row['Current_State'] == 'Execution':
        if row['Performance_Score'] > 1.3 and row['Financial_Health'] > 0.7:
            return 'Revenue-Based Financing'
        elif row['Risk_Level'] > 0.8:
            return 'Public-Private Partnerships'
        else:
            return 'Debt Financing'
    
    elif row['Current_State'] == 'Scaling Up':
        if row['Performance_Score'] > 1.4 and row['Financial_Health'] > 0.8:
            return 'Equity Investment'
        elif row['Risk_Level'] > 0.7:
            return 'Public-Private Partnerships'
        else:
            return 'Revenue-Based Financing'
    
    elif row['Current_State'] == 'Restructuring':
        if row['Performance_Score'] > 1.1 and row['Financial_Health'] > 0.5:
            return 'Debt Financing'
        else:
            return 'Government Grants'
    
    elif row['Current_State'] == 'Exit/Maturity':
        return 'IPO'
    
    else:
        return 'Unknown'

# Menentukan strategi pendanaan baru untuk setiap proyek
projects_funding['New_Funding_Strategy'] = projects_funding.apply(determine_funding_strategy, axis=1)

# Menentukan tipe pendanaan yang digunakan (Hybrid atau Fixed)
projects_funding['Financing_Type'] = np.where(
    projects_funding['Funding_Type'] == projects_funding['New_Funding_Strategy'], 
    'Fixed Financing', 'Hybrid Financing'
)

# Menghitung rata-rata ROI dan Efisiensi Finansial untuk kedua kelompok
roi_comparison = projects_funding.groupby('Financing_Type')['Financial_Health'].mean().rename("Avg_ROI")
efficiency_comparison = projects_funding.groupby('Financing_Type')['Performance_Score'].mean().rename("Avg_Efficiency")

# Menggabungkan hasil perbandingan dalam satu tabel
comparison_results = pd.concat([roi_comparison, efficiency_comparison], axis=1)

# Visualisasi perbandingan ROI dan Efisiensi Finansial
plt.figure(figsize=(10, 5))
sns.barplot(x=comparison_results.index, y=comparison_results['Avg_ROI'], palette="Blues")
plt.title("Perbandingan Rata-rata ROI antara Hybrid vs Fixed Financing")
plt.ylabel("Average ROI")
plt.xlabel("Financing Type")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=comparison_results.index, y=comparison_results['Avg_Efficiency'], palette="Greens")
plt.title("Perbandingan Efisiensi Finansial antara Hybrid vs Fixed Financing")
plt.ylabel("Average Efficiency Score")
plt.xlabel("Financing Type")
plt.show()

# Uji statistik untuk melihat perbedaan signifikan antara kedua kelompok
hybrid_group = projects_funding[projects_funding['Financing_Type'] == 'Hybrid Financing']['Financial_Health']
fixed_group = projects_funding[projects_funding['Financing_Type'] == 'Fixed Financing']['Financial_Health']

t_stat, p_value = ttest_ind(hybrid_group, fixed_group, equal_var=False)

# Menampilkan hasil uji statistik
t_stat, p_value
