"""Isolates whether the vanished epilepsy cold-bucket delta on the expanded pool
(run_pipeline_expanded.py, +0.116 -> ~-0.03 at the now-confirmed-correct lr=1e-3; see
results/lr_selection_validation.csv) is caused by the 25 new non-LINCS positives (which have
no real L1000 profile, only the expr_mask=False sentinel) rather than genuine
non-generalization of the debiasing effect.

Restricts the expanded epilepsy pool to expr_mask==True rows only (i.e. drops every
compound added by the LINCS-decoupling, keeping only the original LINCS-covered pool, but
pulled through load_expanded()'s data arrays rather than load()'s, as a consistency check
on the restriction itself), then reruns the same scaffold CV x 5 seeds x fusion_full vs
fusion_nodebias comparison at lr=1e-3, n_epochs=60.

If the delta reappears close to the committed +0.116 once non-LINCS compounds are excluded,
the non-replication is explained by that specific data-availability change, not by a general
failure of the debiasing effect to generalize. If it stays near zero, the missing-expression
explanation is ruled out too.

Writes results/oof_epilepsy_diag_lincsonly.parquet and prints the per-seed cold delta.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datalib import load_expanded, splits, standardize_view
from guide import guide_importance_weights, weight_schedule, nn_sim_feature
from fusion_model import train_fusion, predict_fusion

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
LOGF = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/logs/diagnose_expanded_lincsonly.log"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 60
LR = 1e-3


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def main():
    t0 = time.time()
    cmp, ecfp, expr, expr_mask, text, text_mask, y = load_expanded("epilepsy")
    keep = expr_mask.astype(bool)
    cmp, ecfp, expr, expr_mask, text, text_mask, y = (
        cmp[keep].reset_index(drop=True), ecfp[keep], expr[keep], expr_mask[keep],
        text[keep], text_mask[keep], y[keep],
    )
    log(f"LINCS-only subset of the expanded pool: n={len(y)} n_pos={y.sum()} "
        f"(original committed pool: n=9878 n_pos=31)")

    oof_rows = []
    for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=SEEDS):
        ytr, yte = y[tr], y[te]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        nn_sim_te = nn_sim_feature(ecfp[tr], ytr, ecfp[te])
        importance, _ = guide_importance_weights(ecfp[tr], ytr)

        expr_tr_s, expr_te_s = standardize_view(expr[tr], expr[te])
        text_tr_s, text_te_s = standardize_view(text[tr], text[te], text_mask[tr], text_mask[te])

        train_seed = seed * 1000 + fold

        m_nodebias = train_fusion(ecfp[tr], expr_tr_s, text_tr_s, text_mask[tr], ytr, np.ones(len(tr)),
                                   use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, lr=LR,
                                   seed=train_seed)
        s_nodebias = predict_fusion(m_nodebias, ecfp=ecfp[te], expr=expr_te_s, text=text_te_s,
                                     text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        sched = lambda ep: weight_schedule(importance, ep, N_EPOCHS)
        m_full = train_fusion(ecfp[tr], expr_tr_s, text_tr_s, text_mask[tr], ytr, sched,
                               use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, lr=LR,
                               seed=train_seed)
        s_full = predict_fusion(m_full, ecfp=ecfp[te], expr=expr_te_s, text=text_te_s,
                                 text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        row = pd.DataFrame({"seed": seed, "fold": fold, "idx": te, "y": yte, "nn_sim": nn_sim_te})
        row["fusion_full"] = s_full
        row["fusion_nodebias"] = s_nodebias
        oof_rows.append(row)

    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_parquet(f"{RESULTS}/oof_epilepsy_diag_lincsonly.parquet", index=False)

    nn_sim = oof.nn_sim.values
    valid = nn_sim >= 0
    terts = np.nanpercentile(nn_sim[valid], [33.33, 66.67])
    cold = valid & (nn_sim <= terts[0])
    sub = oof[cold]
    rows = []
    for seed, g in sub.groupby("seed"):
        yv = g.y.values
        if yv.sum() < 1 or yv.sum() == len(yv):
            continue
        a_full = roc_auc_score(yv, g.fusion_full.values)
        a_nodebias = roc_auc_score(yv, g.fusion_nodebias.values)
        rows.append(a_full - a_nodebias)
    rows = np.array(rows)
    log(f"[lincsonly_subset_of_expanded] lr={LR} n_epochs={N_EPOCHS} usable_seeds={len(rows)}/5 "
        f"mean_delta={rows.mean():+.3f} std={rows.std():.3f} per_seed={np.round(rows,3).tolist()}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
