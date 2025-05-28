# utils/risk_utils.py

def compute_risks(p, w_t=0.6, w_f=0.4):
    """
    Computes and assigns risk_tech, risk_fin, and overall risk to a project p.
    - w_t: weight for technical risk
    - w_f: weight for financial risk
    """
    p['risk_tech'] = ((9 - p['trl']) / 8) * p['complexity']
    p['risk_fin'] = (
        p['alpha'] * 0.0 +     # Equity
        p['beta'] * 0.3 +      # Soft Loan
        p['theta'] * 1.0 +     # Vendor Financing
        p['gamma'] * 0.1 +     # Grant
        p['delta'] * 0.6       # PPP
    )
    p['risk'] = max(0.05, w_t * p['risk_tech'] + w_f * p['risk_fin'])
