"""Business cost model and threshold optimization for the churn classifier."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Retention campaign economics -- see README for the derivation.
OFFER_COST = 50        # cost of extending a retention offer
LTV = 1400              # customer lifetime value lost if they churn
EFFECTIVENESS = 0.30    # fraction of would-be churners the offer actually saves


def expected_cost(y_true, proba, threshold,
                   offer_cost=OFFER_COST, ltv=LTV, eff=EFFECTIVENESS):
    """Total expected cost of operating the classifier at `threshold`."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cost_tp = offer_cost + (1 - eff) * ltv   # intervened churner: offer + residual loss
    cost_fn = ltv                            # missed churner: full LTV lost
    cost_fp = offer_cost                     # wasted offer on a loyal customer
    return tp * cost_tp + fn * cost_fn + fp * cost_fp


def threshold_sweep(y_true, proba, lo=0.01, hi=0.99, n=99, **cost_kwargs):
    rows = []
    for t in np.linspace(lo, hi, n):
        rows.append({"threshold": t, "cost": expected_cost(y_true, proba, t, **cost_kwargs)})
    return pd.DataFrame(rows)


def analytical_optimal_threshold(offer_cost=OFFER_COST, ltv=LTV, eff=EFFECTIVENESS):
    """Closed-form optimum: intervene when p * eff * ltv > offer_cost."""
    return offer_cost / (eff * ltv)
