def set_h2_scenario(scenario_code, projects):
    """
    Configure funding strategy for each scenario in Hypothesis 2.
    Each project will have a mix of funding types: α, β, θ, γ, δ,
    which must always sum to 1.0.
    """

    if scenario_code == "S2.1":
        # 100% Direct Investment
        for p in projects:
            p['alpha'] = 1.0
            p['beta'] = 0.0
            p['theta'] = 0.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.2":
        # 100% Soft Loan
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 1.0
            p['theta'] = 0.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.3":
        # 100% Vendor Financing
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 0.0
            p['theta'] = 1.0
            p['gamma'] = 0.0
            p['delta'] = 0.0

    elif scenario_code == "S2.4":
        # 100% Grant-based Funding
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 0.0
            p['theta'] = 0.0
            p['gamma'] = 1.0
            p['delta'] = 0.0

    elif scenario_code == "S2.5":
        # 100% PPP-based Funding
        for p in projects:
            p['alpha'] = 0.0
            p['beta'] = 0.0
            p['theta'] = 0.0
            p['gamma'] = 0.0
            p['delta'] = 1.0

    elif scenario_code == "S2.6":
        # Adaptive Hybrid Funding: uses all 5 funding types based on project characteristics
        for p in projects:
            risk = p.get('risk', 0.5)
            svs = p.get('svs', 60)
            group = p.get('benefit_group', '')
            
            # Base allocation
            if svs > 85 and risk < 0.4:
                alpha, beta, theta = 0.1, 0.2, 0.6
            elif risk > 0.6:
                alpha, beta, theta = 0.6, 0.3, 0.1
            else:
                alpha, beta, theta = 0.3, 0.3, 0.2

            # Adjust grant and PPP based on project classification
            gamma = 0.0
            delta = 0.0
            if group == "Business Culture":
                gamma = 0.2
            elif group == "Operational Efficiency":
                delta = 0.2

            # Normalize all five funding components
            total = alpha + beta + theta + gamma + delta
            p['alpha'] = alpha / total
            p['beta'] = beta / total
            p['theta'] = theta / total
            p['gamma'] = gamma / total
            p['delta'] = delta / total

    else:
        raise ValueError(f"Unknown scenario code: {scenario_code}")
