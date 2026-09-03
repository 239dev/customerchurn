"""Cost model for turning a churn probability into a go/no-go decision."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# retention offer economics -- see README for where these come from
OFFER_COST = 50       # cost of extending the offer
LTV = 1400             # lifetime value lost if the customer leaves
EFFECTIVENESS = 0.30   # fraction of would-be churners the offer actually keeps


def expected_cost(y_true, proba, threshold, offer_cost=OFFER_COST, ltv=LTV, eff=EFFECTIVENESS):
    """Total cost of running the classifier at a given threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    cost_tp = offer_cost + (1 - eff) * ltv  # offered, but still lose them (1-eff) of the time
    cost_fn = ltv                            # missed a churner entirely
    cost_fp = offer_cost                     # offer wasted on someone who wasn't leaving

    return tp * cost_tp + fn * cost_fn + fp * cost_fp


def threshold_sweep(y_true, proba, lo=0.01, hi=0.99, n=99, **cost_kwargs):
    rows = [{"threshold": t, "cost": expected_cost(y_true, proba, t, **cost_kwargs)}
            for t in np.linspace(lo, hi, n)]
    return pd.DataFrame(rows)


def analytical_optimal_threshold(offer_cost=OFFER_COST, ltv=LTV, eff=EFFECTIVENESS):
    """p * eff * ltv > offer_cost  =>  p > offer_cost / (eff * ltv)"""
    return offer_cost / (eff * ltv)
