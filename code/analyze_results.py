"""Post-hoc analysis over the OOF prediction files written by run_pipeline.py:
  - per (class, model): mean +/- std AUROC across the 5 seeds (headline numbers)
  - bootstrap 95% CI on pooled OOF AUROC per (class, model)
  - warm (top-tertile NN-similarity) vs cold (bottom-tertile) AUROC per (class, model)
  - continuous similarity-stratified AUROC (quintile bins of NN-similarity)
Writes CSVs to results/.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
MODELS = ["random", "nn_tanimoto", "lr_ecfp", "xgb_ecfp", "text_only",
          "fusion_struct", "fusion_struct_expr", "fusion_nodebias", "fusion_full"]


def bootstrap_auroc_ci(y, s, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    if y.sum() == 0 or y.sum() == n:
        return np.nan, np.nan, np.nan
    aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or y[idx].sum() == n:
            continue
        aurocs.append(roc_auc_score(y[idx], s[idx]))
    aurocs = np.array(aurocs)
    return np.mean(aurocs), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def main():
    files = sorted(glob.glob(f"{RESULTS}/oof_*.parquet"))
    per_seed_rows, bootstrap_rows, warmcold_rows, quintile_rows = [], [], [], []

    for fp in files:
        disease = fp.split("oof_")[-1].replace(".parquet", "")
        df = pd.read_parquet(fp)
        for model in MODELS:
            if model not in df.columns:
                continue
            # per-seed AUROC (pooled across folds within a seed) -> mean/std across 5 seeds
            seed_aurocs = []
            for seed, g in df.groupby("seed"):
                y, s = g["y"].values, g[model].values
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                seed_aurocs.append(roc_auc_score(y, s))
            if seed_aurocs:
                per_seed_rows.append(dict(disease=disease, model=model,
                                           mean_auroc=np.mean(seed_aurocs),
                                           std_auroc=np.std(seed_aurocs),
                                           n_seeds=len(seed_aurocs)))

            # pooled bootstrap CI (all seeds/folds pooled -- note: not iid, reported descriptively)
            y, s = df["y"].values, df[model].values
            mean_b, lo, hi = bootstrap_auroc_ci(y, s)
            bootstrap_rows.append(dict(disease=disease, model=model, boot_mean_auroc=mean_b,
                                        ci_lo=lo, ci_hi=hi, n=len(y), n_pos=int(y.sum())))

            # warm/cold by NN-similarity tertile (computed within each test fold already; pool here)
            nn_sim = df["nn_sim"].values
            valid = nn_sim >= 0
            if valid.sum() > 10:
                terts = np.nanpercentile(nn_sim[valid], [33.33, 66.67])
                cold_mask = valid & (nn_sim <= terts[0])
                warm_mask = valid & (nn_sim >= terts[1])
                for tag, mask in [("cold", cold_mask), ("warm", warm_mask), ("all", valid)]:
                    yy, ss = y[mask], s[mask]
                    if yy.sum() > 0 and yy.sum() < len(yy):
                        warmcold_rows.append(dict(disease=disease, model=model, bucket=tag,
                                                   auroc=roc_auc_score(yy, ss), n=len(yy), n_pos=int(yy.sum())))

                # continuous: quintile bins of NN-similarity
                qs = np.nanpercentile(nn_sim[valid], [20, 40, 60, 80])
                bins = np.digitize(nn_sim, qs)
                for b in range(5):
                    mask_b = valid & (bins == b)
                    yy, ss = y[mask_b], s[mask_b]
                    if yy.sum() > 0 and yy.sum() < len(yy):
                        quintile_rows.append(dict(disease=disease, model=model, sim_quintile=b,
                                                   auroc=roc_auc_score(yy, ss), n=len(yy), n_pos=int(yy.sum())))

    pd.DataFrame(per_seed_rows).to_csv(f"{RESULTS}/headline_mean_std_auroc.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(f"{RESULTS}/bootstrap_ci.csv", index=False)
    pd.DataFrame(warmcold_rows).to_csv(f"{RESULTS}/warm_cold_auroc.csv", index=False)
    pd.DataFrame(quintile_rows).to_csv(f"{RESULTS}/similarity_quintile_auroc.csv", index=False)
    print("Wrote headline_mean_std_auroc.csv, bootstrap_ci.csv, warm_cold_auroc.csv, similarity_quintile_auroc.csv")


if __name__ == "__main__":
    main()
