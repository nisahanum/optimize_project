def tchebycheff_eval(Z, weight, ideal, z_min, z_max):
    Z_norm = []
    for i in range(len(Z)):
        denom = z_max[i] - z_min[i]
        if denom < 1e-6:
            # Avoid divide-by-zero: assume no variation, set to 0.0 or skip normalization
            Z_norm.append(0.0)
        else:
            Z_norm.append((Z[i] - z_min[i]) / (denom + 1e-8))

    return max([weight[i] * abs(Z_norm[i] - ideal[i]) for i in range(len(Z))])
