# === MOEA/D Algorithm Parameters ===
POPULATION_SIZE = 100
NUM_GENERATIONS = 300
NEIGHBORHOOD_SIZE = 20

# Enhanced Diversity Settings
INITIAL_MUTATION_RATE = 0.3       # more aggressive mutation
MIN_MUTATION_RATE = 0.05           # keep mutation alive
DIVERSITY_THRESHOLD = 3           # lower barrier to allow more acceptance
DIVERSITY_ACCEPTANCE_PROB = 0.1   # higher chance of accepting diverse child
FUNDING_MUTATION_SIGMA = 0.02  # bisa diubah menjadi 0.05 untuk lebih agresif


# === Objective Normalization Bounds (Optional) ===
#Z1_MIN = 0
#Z1_MAX = 2000
#Z2_MIN = 0
#Z2_MAX = 1
#Z3_MIN = 0
#Z3_MAX = 1000
