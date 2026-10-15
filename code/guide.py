"""DebiasedDTA-style guide: cross-validated NN-similarity guide -> per-sample importance
coefficients, following literature_review.md item #1: a low-capacity guide is
cross-validated (R repeats of K-fold) on the training set; the per-sample importance
coefficient is the median out-of-fold squared error across repeats (compute budget: R=5
repeats of 5-fold CV, vs. the original paper's 10 repeats -- documented in RESULTS.md).
Samples the guide explains well (low error, i.e. the chemical-analog bias already predicts
them) get a LOW importance coefficient; samples it cannot explain get a HIGH one.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from datalib import tanimoto_max


def nn_sim_feature(ecfp_tr, y_tr, ecfp_target):
    """Max Tanimoto similarity of each row of ecfp_target to the positive rows of ecfp_tr."""
    pos = ecfp_tr[y_tr == 1]
    return tanimoto_max(ecfp_target, pos)


def guide_importance_weights(ecfp_tr, y_tr, n_repeats=5, n_folds=5, seed0=1000):
    """Returns (importance, nn_sim_full) both length len(y_tr).
    importance: median OOF squared-error of the 1-feature (NN-similarity) logistic guide,
    normalized to mean 1 across the training fold.
    nn_sim_full: NN-similarity feature computed using ALL training positives (for use as
    the baseline/guide score at test time, not used in the CV above to avoid leakage)."""
    n = len(y_tr)
    errs = np.full((n_repeats, n), np.nan)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed0 + r)
        for tr_idx, va_idx in skf.split(np.zeros(n), y_tr):
            feat_tr = nn_sim_feature(ecfp_tr[tr_idx], y_tr[tr_idx], ecfp_tr[tr_idx]).reshape(-1, 1)
            feat_va = nn_sim_feature(ecfp_tr[tr_idx], y_tr[tr_idx], ecfp_tr[va_idx]).reshape(-1, 1)
            if y_tr[tr_idx].sum() == 0 or y_tr[tr_idx].sum() == len(tr_idx):
                p_va = np.full(len(va_idx), y_tr[tr_idx].mean())
            else:
                lr = LogisticRegression(class_weight="balanced", max_iter=500)
                lr.fit(feat_tr, y_tr[tr_idx])
                p_va = lr.predict_proba(feat_va)[:, 1]
            errs[r, va_idx] = (p_va - y_tr[va_idx]) ** 2
    importance = np.median(errs, axis=0)
    importance = importance / (importance.mean() + 1e-8)
    nn_sim_full = nn_sim_feature(ecfp_tr, y_tr, ecfp_tr)
    return importance, nn_sim_full


def weight_schedule(importance, epoch, n_epochs):
    """Convex interpolation from uniform (epoch 0) to importance coefficients (epoch n_epochs-1),
    exactly the DebiasedDTA epoch-wise sample-weight schedule."""
    t = epoch / max(1, n_epochs - 1)
    return (1 - t) * np.ones_like(importance) + t * importance
