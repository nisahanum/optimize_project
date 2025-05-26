# === MOEA/D Algorithm Parameters ===
POPULATION_SIZE = 100
NUM_GENERATIONS = 300
NEIGHBORHOOD_SIZE = 20

# Enhanced Diversity Settings
INITIAL_MUTATION_RATE = 0.4       # more aggressive mutation
MIN_MUTATION_RATE = 0.1           # keep mutation alive
DIVERSITY_THRESHOLD = 1           # lower barrier to allow more acceptance
DIVERSITY_ACCEPTANCE_PROB = 0.6   # higher chance of accepting diverse child

# === Objective Normalization Bounds (Optional) ===
#Z1_MIN = 0
#Z1_MAX = 2000
#Z2_MIN = 0
#Z2_MAX = 1
#Z3_MIN = 0
#Z3_MAX = 1000
