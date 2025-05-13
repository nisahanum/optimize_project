import numpy as np
import pandas as pd
from IPython.display import display
# Simulasi Pendanaan dalam Transisi State Proyek dengan Data Nyata

# Membuat dataset proyek dengan variabel state dan pendanaan awal
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
            return 'Debt Financing'  # Proyek yang kuat akan mendapat utang
        else:
            return 'Venture Capital'  # Proyek awal yang masih lemah lebih cocok dengan modal ventura
    
    elif row['Current_State'] == 'Execution':
        if row['Performance_Score'] > 1.3 and row['Financial_Health'] > 0.7:
            return 'Revenue-Based Financing'  # Proyek dengan kinerja tinggi mulai berbasis pendapatan
        elif row['Risk_Level'] > 0.8:
            return 'Public-Private Partnerships'  # Proyek dengan risiko tinggi membutuhkan dukungan pemerintah
        else:
            return 'Debt Financing'
    
    elif row['Current_State'] == 'Scaling Up':
        if row['Performance_Score'] > 1.4 and row['Financial_Health'] > 0.8:
            return 'Equity Investment'  # Proyek sukses bisa mencari investasi ekuitas besar
        elif row['Risk_Level'] > 0.7:
            return 'Public-Private Partnerships'  # Risiko tinggi tetap butuh jaminan dari sektor publik
        else:
            return 'Revenue-Based Financing'
    
    elif row['Current_State'] == 'Restructuring':
        if row['Performance_Score'] > 1.1 and row['Financial_Health'] > 0.5:
            return 'Debt Financing'  # Restrukturisasi dengan utang jika ada potensi pemulihan
        else:
            return 'Government Grants'  # Proyek yang sulit pulih butuh hibah pemerintah
    
    elif row['Current_State'] == 'Exit/Maturity':
        return 'IPO'  # Proyek yang sukses bisa dilepas ke publik melalui IPO
    
    else:
        return 'Unknown'

# Menentukan strategi pendanaan baru untuk setiap proyek
projects_funding['New_Funding_Strategy'] = projects_funding.apply(determine_funding_strategy, axis=1)

# Menampilkan hasil simulasi pendanaan dalam transisi state proyek
display(projects_funding)