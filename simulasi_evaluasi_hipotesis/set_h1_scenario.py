
# set_h1_scenario.py

def set_h1_scenario(scenario_code, projects):
    """
    Menyesuaikan nilai synergy_same dan synergy_cross sesuai skenario H1:
    - S1.1: No synergy
    - S1.2: Only same-period synergy
    - S1.3: Only cross-period synergy
    - S1.4: Full synergy (same + cross)
    """
    for p in projects:
        if scenario_code == "S1.1":
            p['synergy_same'] = 0
            p['synergy_cross'] = 0
        elif scenario_code == "S1.2":
            # only use same-period synergy
            p['synergy_cross'] = 0
        elif scenario_code == "S1.3":
            # only use cross-period synergy
            p['synergy_same'] = 0
        elif scenario_code == "S1.4":
            # full synergy (default)
            pass
