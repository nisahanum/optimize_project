def evaluate(individual):
    total_roi = 0
    total_risk_adjusted_cost = 0
    total_synergy = 0

    for i in range(NUM_PROJECTS):
        x = int(round(individual[i * 5]))  # binary project selection
        a = individual[i * 5 + 1]
        b = individual[i * 5 + 2]
        c = individual[i * 5 + 3]
        s = individual[i * 5 + 4]

        if x == 0:
            continue

        proj = project_data[i]
        ROI = proj['roi']
        cost = proj['cost']
        risk = proj['risk']
        synergy = proj['synergy']

        # Objective 1: ROI adjusted by risk
        total_roi += ROI * (1 - risk)

        # Objective 2: Risk-adjusted cost
        total_risk_adjusted_cost += (a + b + c) * cost * (1 + risk)

        # Objective 3: Total synergy (maximize)
        total_synergy += synergy

    return -total_roi, total_risk_adjusted_cost, -total_synergy
