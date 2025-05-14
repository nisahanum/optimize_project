import numpy as np

def load_synergy_matrix(num_projects=10, seed=42):
    """
    Membuat matriks sinergi simetris antar proyek (delta_ij).
    """
    np.random.seed(seed)
    matrix = np.random.uniform(0, 100, size=(num_projects, num_projects))
    
    # Buat simetris dan nol-kan diagonal
    for i in range(num_projects):
        for j in range(num_projects):
            if i == j:
                matrix[i][j] = 0
            elif i < j:
                matrix[j][i] = matrix[i][j]
    
    return matrix
