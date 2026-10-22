"""Generalizes diagnose_expanded_lincs_only.py (epilepsy-only) to all nine classes: does the
cold-bucket debiasing delta hold on the LINCS-covered subset of the expanded pool but vanish
once non-LINCS (expr_mask=False) compounds are included, or was that an epilepsy-specific
coincidence?

For each class, restricts the expanded pool to expr_mask==True rows (same compounds as the
original committed pool, pulled through load_expanded()'s arrays), reruns fusion_full vs
fusion_nodebias at the validation-confirmed lr=1e-3 (results/lr_selection_validation.csv),
and reports the same pooled-cutoff per-seed cold delta used throughout
(analyze_expanded_cold_delta.py / cold_delta_per_seed_reliability*.csv), directly comparable
to the already-computed full-expanded-pool numbers in
results/cold_delta_per_seed_reliability_expanded.csv and the original committed numbers in
results/cold_delta_per_seed_reliability.csv.

Writes results/cold_delta_per_seed_reliability_expanded_lincsonly.csv.
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datalib import NINE_CLASSES, load_expanded, splits, standardize_view
from guide import guide_importance_weights, weight_schedule, nn_sim_feature
from fusion_model import train_fusion, predict_fusion

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
LOGF = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/logs/diagnose_expanded_lincsonly_all_classes.log"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 60
LR = 1e-3


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def run_disease(disease):
    cmp, ecfp, expr, expr_mask, text, text_mask, y = load_expanded(disease)
    keep = expr_mask.astype(bool)
    cmp, ecfp, expr, expr_mask, text, text_mask, y = (
        cmp[keep].reset_index(drop=True), ecfp[keep], expr[keep], expr_mask[keep],
        text[keep], text_mask[keep], y[keep],
    )
    log(f"[{disease}] LINCS-only subset of expanded pool: n={len(y)} n_pos={int(y.sum())}")
    if y.sum() < 6:
        log(f"[{disease}] too few positives, skipping")
        return None

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
    oof.to_parquet(f"{RESULTS}/oof_{disease}_diag_lincsonly.parquet", index=False)

    nn_sim = oof.nn_sim.values
    valid = nn_sim >= 0
    terts = np.nanpercentile(nn_sim[valid], [33.33, 66.67])
    cold = valid & (nn_sim <= terts[0])
    sub = oof[cold]
    rows = []
    for seed, g in sub.groupby("seed"):
        yv = g.y.values
        n_pos = int(yv.sum())
        if n_pos < 1 or n_pos == len(yv):
            rows.append(dict(seed=seed, n_pos=n_pos, delta=np.nan))
            continue
        a_full = roc_auc_score(yv, g.fusion_full.values)
        a_nodebias = roc_auc_score(yv, g.fusion_nodebias.values)
        rows.append(dict(seed=seed, n_pos=n_pos, delta=a_full - a_nodebias))
    per_seed = pd.DataFrame(rows)
    usable = per_seed.dropna(subset=["delta"])
    mean_n_pos = per_seed.n_pos.mean()
    mean_delta = usable.delta.mean() if len(usable) else np.nan
    std_delta = usable.delta.std() if len(usable) else np.nan
    log(f"[{disease}] mean cold n_pos/seed={mean_n_pos:.2f} usable_seeds={len(usable)}/5 "
        f"mean_delta={mean_delta:+.3f} std={std_delta:.3f}" if len(usable) else f"[{disease}] n/a")
    return dict(disease=disease, mean_cold_n_pos_per_seed=mean_n_pos, n_usable_seeds=len(usable),
                mean_delta=mean_delta, std_delta=std_delta, cold_cutoff=terts[0])


def main():
    t0 = time.time()
    summary = []
    for disease in NINE_CLASSES:
        r = run_disease(disease)
        if r is not None:
            summary.append(r)
    out = pd.DataFrame(summary)
    out.to_csv(f"{RESULTS}/cold_delta_per_seed_reliability_expanded_lincsonly.csv", index=False)
    log(f"\nWrote results/cold_delta_per_seed_reliability_expanded_lincsonly.csv")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
