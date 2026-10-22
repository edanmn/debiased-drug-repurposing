"""Principled, non-circular learning-rate selection for the fusion model.

diagnose_epilepsy_training_stability.py found that lr=3e-4 restores the epilepsy
cold-bucket delta on the expanded pool, and diagnose_epilepsy_lr_on_original.py found
lr=3e-4 also gives a bigger delta on the original pool. That is not, by itself, a valid
reason to adopt lr=3e-4: both checks pick the learning rate by which one makes the paper's
own test statistic (the cold-bucket delta) look best, which is exactly the kind of
hyperparameter-tuned-on-the-test-metric the reviewer flagged.

This script selects lr a different way, without touching the cold-bucket delta, nn_sim, or
any debiasing-related quantity at all. For every (disease, seed, fold) training set in the
ORIGINAL (committed, non-expanded) pool, it carves an inner validation split out of the
training indices (stratified, 80/20, a fixed sub-seed independent of the outer CV seed),
trains the plain uniform-weight fusion_nodebias architecture (no debiasing schedule) at each
candidate lr for the production n_epochs=60, and scores plain validation AUROC/AUPRC and
validation BCE loss. The outer test fold and every cold-bucket/nn_sim quantity are untouched.
Whichever lr wins on this ordinary supervised validation criterion, averaged across all nine
classes, is the principled choice -- independent of whether it happens to help the paper's
headline result.

Writes results/lr_selection_validation.csv (one row per disease/seed/fold/lr) and prints the
aggregate.
"""
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from datalib import load, splits, standardize_view, NINE_CLASSES
from fusion_model import train_fusion, predict_fusion

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
LOGF = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/logs/select_lr_by_validation.log"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 60
CANDIDATE_LRS = (1e-3, 3e-4, 1e-4)
INNER_VAL_SEED = 20261022  # fixed, independent of outer CV seed


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def main():
    t0 = time.time()
    rows = []
    for disease in NINE_CLASSES:
        cmp, ecfp, expr, text, text_mask, y = load("all", disease)
        for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=SEEDS):
            ytr_full = y[tr]
            if ytr_full.sum() < 6:
                continue  # need positives on both sides of the inner split

            inner_tr, inner_va = train_test_split(
                np.arange(len(tr)), test_size=0.2, stratify=ytr_full,
                random_state=INNER_VAL_SEED + seed * 1000 + fold,
            )
            itr, iva = tr[inner_tr], tr[inner_va]
            ytr, yva = y[itr], y[iva]
            if ytr.sum() < 3 or yva.sum() < 1 or yva.sum() == len(yva):
                continue

            expr_tr_s, expr_va_s = standardize_view(expr[itr], expr[iva])
            text_tr_s, text_va_s = standardize_view(text[itr], text[iva], text_mask[itr], text_mask[iva])

            for lr in CANDIDATE_LRS:
                train_seed = seed * 1000 + fold
                m = train_fusion(ecfp[itr], expr_tr_s, text_tr_s, text_mask[itr], ytr, np.ones(len(itr)),
                                  use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, lr=lr,
                                  seed=train_seed)
                p = predict_fusion(m, ecfp=ecfp[iva], expr=expr_va_s, text=text_va_s, text_mask=text_mask[iva],
                                    use_ecfp=True, use_expr=True, use_text=True)
                auroc = roc_auc_score(yva, p)
                auprc = average_precision_score(yva, p)
                bce = log_loss(yva, np.clip(p, 1e-7, 1 - 1e-7))
                rows.append(dict(disease=disease, seed=seed, fold=fold, lr=lr,
                                  n_tr=len(itr), n_va=len(iva), n_va_pos=int(yva.sum()),
                                  val_auroc=auroc, val_auprc=auprc, val_bce=bce))
            log(f"[{disease}] seed={seed} fold={fold} "
                f"{[(lr, round(r['val_auroc'], 3)) for r, lr in zip(rows[-len(CANDIDATE_LRS):], CANDIDATE_LRS)]}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/lr_selection_validation.csv", index=False)
    log(f"\nWrote results/lr_selection_validation.csv ({len(out)} rows)")

    log("\n=== Aggregate validation performance by lr (plain supervised task, no debiasing, no cold-bucket) ===")
    agg = out.groupby("lr")[["val_auroc", "val_auprc", "val_bce"]].mean()
    log(agg.to_string())

    log("\n=== Per-disease mean validation AUROC by lr ===")
    per_disease = out.groupby(["disease", "lr"])["val_auroc"].mean().unstack()
    log(per_disease.to_string())

    best_lr = agg.val_auroc.idxmax()
    log(f"\nBest lr by mean validation AUROC (plain task, independent of debiasing): {best_lr}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
