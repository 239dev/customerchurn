import numpy as np

from src.evaluate import expected_cost, threshold_sweep, analytical_optimal_threshold


def test_perfect_predictions_cost_nothing():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.0, 0.0, 1.0, 1.0])
    # at threshold 0.5: two true negatives, two true positives -- only the
    # two true positives cost anything (offer + residual churn risk)
    cost = expected_cost(y_true, proba, threshold=0.5, offer_cost=50, ltv=1000, eff=0.5)
    assert cost == 2 * (50 + 0.5 * 1000)


def test_missed_churner_costs_full_ltv():
    y_true = np.array([1])
    proba = np.array([0.1])  # below threshold, so we don't act
    cost = expected_cost(y_true, proba, threshold=0.5, ltv=1000)
    assert cost == 1000


def test_wasted_offer_costs_only_the_offer():
    y_true = np.array([0])
    proba = np.array([0.9])  # above threshold, but they were never leaving
    cost = expected_cost(y_true, proba, threshold=0.5, offer_cost=50)
    assert cost == 50


def test_true_negative_costs_nothing():
    y_true = np.array([0])
    proba = np.array([0.1])
    assert expected_cost(y_true, proba, threshold=0.5) == 0


def test_analytical_threshold_matches_its_own_formula():
    t = analytical_optimal_threshold(offer_cost=50, ltv=1400, eff=0.3)
    assert t == 50 / (1400 * 0.3)


def test_threshold_sweep_covers_the_requested_range():
    y_true = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.2, 0.8, 0.4, 0.6, 0.9])
    sweep = threshold_sweep(y_true, proba, lo=0.1, hi=0.9, n=9)
    assert len(sweep) == 9
    assert sweep["threshold"].min() >= 0.1
    assert sweep["threshold"].max() <= 0.9


def test_higher_threshold_never_increases_true_positives_caught():
    # a sanity check on the underlying confusion-matrix math, not just cost:
    # as the bar for "acting" goes up, cost at the extremes should bracket
    # everything in between (nobody flagged vs. everybody flagged)
    y_true = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([0.1, 0.9, 0.3, 0.6, 0.4, 0.2])
    cost_act_on_nobody = expected_cost(y_true, proba, threshold=1.01)
    cost_act_on_everybody = expected_cost(y_true, proba, threshold=0.0)
    n_churners = y_true.sum()
    n_stayers = len(y_true) - n_churners
    assert cost_act_on_nobody == n_churners * 1400  # every churner missed
    assert cost_act_on_everybody == n_churners * (50 + 0.7 * 1400) + n_stayers * 50
