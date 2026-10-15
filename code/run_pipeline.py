"""Main evaluation pipeline for Idea 1. For each of the nine therapeutic classes, runs
scaffold-grouped CV (5 seeds x 5 folds) and evaluates:
  - random
  - nn_tanimoto          : zero-training NN-search baseline/guide
  - lr_ecfp, xgb_ecfp    : plain fingerprint baselines
  - text_only            : target-knowledge-only ablation (no chemistry)
  - fusion_struct        : structure-only fusion-model ablation
  - fusion_struct_expr   : structure+L1000 fusion-model ablation
  - fusion_full          : structure+L1000+text, DebiasedDTA-reweighted (PROPOSED METHOD)
  - fusion_nodebias      : same architecture as fusion_full but uniform sample weights
                           throughout training (debiasing-removed ablation)

Writes:
  results/oof_<class>.parquet  -- one row per (seed, fold, test_idx), all model scores + nn_sim + y
  results/fold_metrics_<class>.csv -- per (seed, fold, model) metrics
Logs timing to logs/run_pipeline.log
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/code")
from datalib import load, splits, metrics, tanimoto_max, standardize_view
from guide import guide_importance_weights, weight_schedule, nn_sim_feature
from fusion_model import train_fusion, predict_fusion

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
LOGF = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/logs/run_pipeline.log"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 60


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def run_class(disease, pool="all"):
    cmp, ecfp, expr, text, text_mask, y = load(pool, disease)
    log(f"[{disease}] n={len(y)} n_pos={y.sum()}")

    fold_rows = []
    oof_rows = []

    for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=SEEDS):
        t0 = time.time()
        ytr, yte = y[tr], y[te]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        rng = np.random.default_rng(seed * 1000 + fold)

        # ---- guide / NN baseline ----
        nn_sim_te = nn_sim_feature(ecfp[tr], ytr, ecfp[te])
        importance, nn_sim_tr = guide_importance_weights(ecfp[tr], ytr)

        scores = {"random": rng.random(len(te)), "nn_tanimoto": nn_sim_te}

        # ---- standardize continuous views using TRAIN-only statistics (fixes a real bug
        # found during development: unnormalized PubMedBERT text embeddings and L1000
        # z-scores at a different scale than the binary ECFP view caused the single-view
        # text_only ablation to produce inverted (AUROC<0.5) rankings; see RESULTS.md) ----
        expr_tr_s, expr_te_s = standardize_view(expr[tr], expr[te])
        text_tr_s, text_te_s = standardize_view(text[tr], text[te], text_mask[tr], text_mask[te])

        # ---- plain fingerprint baselines ----
        sc = StandardScaler().fit(ecfp[tr])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(ecfp[tr]), ytr)
        scores["lr_ecfp"] = lr.decision_function(sc.transform(ecfp[te]))

        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.3, scale_pos_weight=max(1, (ytr == 0).sum() / max(1, ytr.sum())),
                             tree_method="hist", random_state=seed, verbosity=0, n_jobs=10).fit(ecfp[tr], ytr)
        scores["xgb_ecfp"] = xgb.predict_proba(ecfp[te])[:, 1]

        # ---- text-only ablation (target-knowledge-only, no chemistry) ----
        m_text = train_fusion(None, None, text_tr_s, text_mask[tr], ytr, np.ones(len(tr)),
                               use_ecfp=False, use_expr=False, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        scores["text_only"] = predict_fusion(m_text, text=text_te_s, text_mask=text_mask[te],
                                              use_ecfp=False, use_expr=False, use_text=True)

        # ---- fusion ablations (view ablation, uniform weights) ----
        m_struct = train_fusion(ecfp[tr], None, None, None, ytr, np.ones(len(tr)),
                                 use_ecfp=True, use_expr=False, use_text=False, n_epochs=N_EPOCHS, seed=seed)
        scores["fusion_struct"] = predict_fusion(m_struct, ecfp=ecfp[te], use_ecfp=True, use_expr=False, use_text=False)

        m_se = train_fusion(ecfp[tr], expr_tr_s, None, None, ytr, np.ones(len(tr)),
                             use_ecfp=True, use_expr=True, use_text=False, n_epochs=N_EPOCHS, seed=seed)
        scores["fusion_struct_expr"] = predict_fusion(m_se, ecfp=ecfp[te], expr=expr_te_s, use_ecfp=True, use_expr=True, use_text=False)

        # ---- debiasing-removed ablation: full 3-view fusion, uniform weights ----
        m_nodebias = train_fusion(ecfp[tr], expr_tr_s, text_tr_s, text_mask[tr], ytr, np.ones(len(tr)),
                                   use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        scores["fusion_nodebias"] = predict_fusion(m_nodebias, ecfp=ecfp[te], expr=expr_te_s, text=text_te_s,
                                                    text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        # ---- PROPOSED METHOD: full 3-view fusion, DebiasedDTA importance-reweighted ----
        sched = lambda ep: weight_schedule(importance, ep, N_EPOCHS)
        m_full = train_fusion(ecfp[tr], expr_tr_s, text_tr_s, text_mask[tr], ytr, sched,
                               use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        scores["fusion_full"] = predict_fusion(m_full, ecfp=ecfp[te], expr=expr_te_s, text=text_te_s,
                                                text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        for name, s in scores.items():
            fold_rows.append(dict(disease=disease, seed=seed, fold=fold, model=name, **metrics(yte, s)))

        row = pd.DataFrame({"seed": seed, "fold": fold, "idx": te, "y": yte, "nn_sim": nn_sim_te})
        for name, s in scores.items():
            row[name] = s
        oof_rows.append(row)

        log(f"[{disease}] seed={seed} fold={fold} n_tr={len(tr)} n_te={len(te)} "
            f"auroc_full={metrics(yte, scores['fusion_full'])['auroc']:.3f} "
            f"auroc_nn={metrics(yte, scores['nn_tanimoto'])['auroc']:.3f} "
            f"({time.time()-t0:.1f}s)")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f"{RESULTS}/fold_metrics_{disease}.csv", index=False)
    oof_df = pd.concat(oof_rows, ignore_index=True)
    oof_df.to_parquet(f"{RESULTS}/oof_{disease}.parquet", index=False)
    log(f"[{disease}] DONE. wrote fold_metrics_{disease}.csv and oof_{disease}.parquet")
    return fold_df


if __name__ == "__main__":
    diseases = sys.argv[1:] or ["epilepsy"]
    for d in diseases:
        run_class(d)
