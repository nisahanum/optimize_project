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
    # Adaptive Hybrid Funding: relaxed θ cap for strategic, low-risk projects
      for p in projects:
        risk = p.get('risk', 0.5)
        svs = p.get('svs', 60)
        fuzzy = p.get('fuzzy_cost', (2.0, 2.5, 3.0))
        cost_est = (fuzzy[0] + 2 * fuzzy[1] + fuzzy[2]) / 4

        if svs > 85 and risk < 0.4:
            # Very strategic and low-risk → allow high vendor
            alpha, beta, theta = 0.1, 0.2, 0.7
        elif risk > 0.6:
            # High risk → equity-heavy
            alpha, beta, theta = 0.6, 0.3, 0.1
        else:
            # Balanced fallback
            alpha, beta, theta = 0.4, 0.3, 0.3

        # Only cap θ for non-strategic or risky projects
        if theta > 0.4 and not (svs > 85 and risk < 0.4):
            excess = theta - 0.4
            theta = 0.4
            alpha += excess * 0.6
            beta += excess * 0.4

        # Normalize
        total = alpha + beta + theta
        p['alpha'] = alpha / total
        p['beta'] = beta / total
        p['theta'] = theta / total
        p['gamma'] = 0.0
        p['delta'] = 0.0


    else:
        raise ValueError(f"Unknown scenario code: {scenario_code}")
