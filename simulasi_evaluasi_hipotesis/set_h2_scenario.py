def set_h2_scenario(scenario_code, projects):
    """
    Configure funding strategy for each scenario in Hypothesis 2.
    Each project will have a mix of funding types: α, β, θ, γ, δ,
    which must always sum to 1.0.
    """

    if scenario_code == "S2.1":
        # Fixed single funding: 100% Direct Investment
        for p in projects:
            p['alpha'] = 1.0
            p['beta'] = 0.0
            p['theta'] = 0.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.2":
        # Fixed single funding: 100% Soft Loan
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 1.0
            p['theta'] = 0.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.3":
        # Fixed single funding: 100% Vendor Financing
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 0.0
            p['theta'] = 1.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.4":
    # Adaptive Hybrid Funding: aggressive when SVS is high
      for p in projects:
        risk = p.get('risk', 0.5)
        svs = p.get('svs', 60)

        if svs > 80:
            # Strategic project → tolerate vendor
            alpha, beta, theta = 0.1, 0.2, 0.7
        elif risk > 0.6:
            # High risk → prefer equity
            alpha, beta, theta = 0.6, 0.3, 0.1
        else:
            # Default balanced
            alpha, beta, theta = 0.3, 0.3, 0.4

        # Cap θ and redistribute if needed
        if theta > 0.4:
            excess = theta - 0.4
            theta = 0.4
            alpha += 0.6 * excess
            beta += 0.4 * excess

        total = alpha + beta + theta
        alpha /= total
        beta /= total
        theta /= total

        p['alpha'], p['beta'], p['theta'] = alpha, beta, theta
        p['gamma'] = 0.0
        p['delta'] = 0.0

    else:
        raise ValueError(f"Unknown scenario code: {scenario_code}")
