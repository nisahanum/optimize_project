import numpy as np

import pandas as pd

def load_synergy_matrix():
    df = pd.read_csv("synergy_matrix_cosine_normalized.csv", index_col=0)
    return df.values  # returns numpy matrix for optimization
