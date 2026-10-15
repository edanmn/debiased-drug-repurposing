"""Shared data loading, splits and metrics for idea1. Adapted from
ICHI2027_mgan/code/evallib.py, extended with the text (target-function) view
and its availability mask. All evaluation is at the compound level.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") + os.sep

NINE_CLASSES = [
    "epilepsy", "antibacterial", "antidiabetic", "antineoplastic", "antipsychotic",
    "betablocker", "depression", "lipid", "nsaid",
]


def load(pool="all", disease="epilepsy"):
    cmp = pd.read_parquet(D + "compounds.parquet")
    if disease != "epilepsy":
        lab = pd.read_csv(D + f"labels_{disease}.csv").set_index("ik14")
        cmp["approved"] = cmp.ik14.map(lab.positive).fillna(False).astype(bool).values
        cmp["clinical"] = cmp.ik14.map(lab.clinical).fillna(False).astype(bool).values
        cmp["max_phase"] = cmp.ik14.map(lab.max_phase).values
        _ay = D + f"approval_years_{disease}.csv"
        if os.path.exists(_ay):
            yr = pd.read_csv(_ay).dropna(subset=["ik14"]).set_index("ik14").first_approval
            cmp["first_approval"] = cmp.ik14.map(yr).values
        else:
            cmp["first_approval"] = np.nan
    keep = cmp.has_structure.values
    if pool == "drugs":
        ph = pd.read_csv(D + "chembl_max_phase_all.csv").set_index("ik14").chembl_max_phase
        keep = keep & ((cmp.ik14.map(ph).fillna(0).values >= 1) | cmp.approved.values)
    ecfp = np.load(D + "cmp_ecfp.npy")[keep].astype(np.float32)
    expr = np.load(D + "cmp_expr.npy")[keep].astype(np.float32)
    text = np.load(D + "cmp_text_emb.npy")[keep].astype(np.float32)
    text_mask = np.load(D + "cmp_text_mask.npy")[keep]
    cmp = cmp[keep].reset_index(drop=True)
    y = cmp.approved.values.astype(int)
    return cmp, ecfp, expr, text, text_mask, y


def splits(cmp, y, kind="scaffold", n_folds=5, seeds=(0, 1, 2, 3, 4)):
    """Yield (seed, fold, train_idx, test_idx). kind='scaffold' is the primary protocol
    (Murcko-scaffold-grouped stratified K-fold); kind='compound' is a simple stratified
    K-fold control with no scaffold grouping."""
    for s in seeds:
        if kind == "compound":
            it = StratifiedKFold(n_folds, shuffle=True, random_state=s).split(np.zeros(len(y)), y)
        else:
            g = pd.factorize(cmp.scaffold.replace("", np.nan).fillna(cmp.ik14))[0]
            it = StratifiedGroupKFold(n_folds, shuffle=True, random_state=s).split(np.zeros(len(y)), y, g)
        for f, (tr, te) in enumerate(it):
            yield s, f, tr, te


def metrics(y, s):
    o = np.argsort(-s, kind="stable")
    ys = y[o]
    P = y.sum()
    n = len(y)
    r = {"auroc": roc_auc_score(y, s) if 0 < P < n else float("nan"),
         "auprc": average_precision_score(y, s) if 0 < P < n else float("nan"),
         "n_pos": int(P), "n": n}
    k1 = max(1, int(round(0.01 * n)))
    r["recall@1%"] = float(ys[:k1].sum() / P) if P > 0 else float("nan")
    return r


def standardize_view(X_tr, X_te, mask_tr=None, mask_te=None):
    """Z-score a continuous view using TRAIN statistics only (no leakage). If a per-row
    availability mask is given (text view: many rows are an all-zero 'missing' sentinel),
    the mean/std are computed only from available train rows, and unavailable rows (train
    or test) are forced back to an exact all-zero sentinel AFTER scaling, so 'missing' stays
    a clean, fixed signal rather than an arbitrary large value induced by mean-subtraction."""
    ref = X_tr[mask_tr] if mask_tr is not None else X_tr
    if mask_tr is not None and ref.shape[0] < 2:
        ref = X_tr
    mu = ref.mean(0, keepdims=True)
    sigma = ref.std(0, keepdims=True) + 1e-6
    Xtr_s = (X_tr - mu) / sigma
    Xte_s = (X_te - mu) / sigma
    if mask_tr is not None:
        Xtr_s[~mask_tr] = 0.0
    if mask_te is not None:
        Xte_s[~mask_te] = 0.0
    return Xtr_s.astype(np.float32), Xte_s.astype(np.float32)


def tanimoto_max(A, B):
    """Max Tanimoto of each row of A to rows of B (binary float matrices). Returns -1 for
    rows of A with no B rows to compare (empty B)."""
    if len(B) == 0:
        return np.full(len(A), -1.0)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return (inter / np.maximum(union, 1)).max(1)


def tanimoto_matrix(A, B):
    if len(B) == 0:
        return np.zeros((len(A), 0))
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)
