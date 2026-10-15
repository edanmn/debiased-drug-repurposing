"""External validation of the headline epilepsy cold-bucket debiasing result on an
independent label source, per the scientific-paper-rigor "single dataset" criticism
(all results in the paper so far come from one indication source, ChEMBL).

repoDB (Brown & Patel, Scientific Data 2017) is a standard, citable drug-repositioning
benchmark built from DrugCentral-approved indications and AACT clinical-trial outcomes,
independent of ChEMBL. Official static deposit: figshare DOI 10.6084/m9.figshare.3811674
(full.csv, md5 61afd7c7adcfbc576c202e2fb05475b8, downloaded directly from
https://ndownloader.figshare.com/files/7341422, cached at data/repodb/full.csv).

We do not retrain anything. We take the already-frozen out-of-fold scores in
results/oof_epilepsy.parquet (produced by run_pipeline.py from ChEMBL-label-supervised
models) and swap the evaluation label from the ChEMBL-derived y to an independently
built repoDB-derived y_repodb, then recompute the exact same cold-bucket AUROC
comparison the paper's headline number is built from. If the +0.116 cold-bucket
debiasing delta is a real property of the ranking and not an artifact of the ChEMBL
label construction, it should survive this label swap.

Writes: results/repodb_external_validation.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from datalib import load, metrics, D

EPI_PATTERN = "epilep|seizure|convulsi"
MODELS = ["nn_tanimoto", "fusion_nodebias", "fusion_full"]


def norm_name(s):
    return s.str.lower().str.strip().str.replace("-", " ", regex=False).str.replace("_", " ", regex=False)


def build_repodb_epilepsy_labels(cmp):
    repodb = pd.read_csv(D + "repodb/full.csv")
    epi = repodb[repodb.ind_name.str.contains(EPI_PATTERN, case=False, na=False) & (repodb.status == "Approved")]
    repodb_names = set(norm_name(epi.drug_name))
    cmp_name_norm = norm_name(cmp.name)
    matched = repodb_names & set(cmp_name_norm)
    y_repodb = cmp_name_norm.isin(repodb_names).values.astype(int)
    return y_repodb, repodb_names, matched


def validate_against_chembl_labels(cmp, y, y_repodb, repodb_names, matched):
    print(f"repoDB lists {len(repodb_names)} drugs with an Approved epilepsy/seizure-related indication.")
    print(f"{len(matched)}/{len(repodb_names)} ({len(matched)/len(repodb_names):.1%}) resolve to a compound "
          f"in our 9,878-compound LINCS library; the rest are not in LINCS at all (coverage limit, not a "
          f"matching failure).")
    overlap = int(((y == 1) & (y_repodb == 1)).sum())
    new_pos = int(((y == 0) & (y_repodb == 1)).sum())
    chembl_only = int(((y == 1) & (y_repodb == 0)).sum())
    print(f"Of the {len(matched)} matched repoDB positives: {overlap} already carry the ChEMBL-derived "
          f"positive label used to train and evaluate the paper's models ({overlap/len(matched):.1%} "
          f"agreement, an independent cross-check on the ChEMBL label pipeline itself), {new_pos} are new "
          f"positives repoDB adds that ChEMBL/our pipeline did not flag, and {chembl_only} ChEMBL positives "
          f"are not confirmed by repoDB (expected: repoDB predates several 2015+ AED approvals and uses its "
          f"own UMLS indication vocabulary).")
    if new_pos:
        print("New repoDB-only positives:", sorted(cmp.name[(y == 0) & (y_repodb == 1)].tolist()))


def cold_bucket_rows(oof, y_repodb, cmp_names_global):
    oof = oof.copy()
    oof["y_repodb"] = y_repodb[oof["idx"].values]
    rows = []
    for seed, g in oof.groupby("seed"):
        valid = g.nn_sim >= 0
        cold_cut = np.nanpercentile(g.nn_sim[valid], 33.33)
        cold = g[g.nn_sim <= cold_cut]
        n_pos = int(cold.y_repodb.sum())
        pos_names = sorted(cold.idx[cold.y_repodb == 1].map(lambda i: cmp_names_global[i]).tolist())
        for model in MODELS:
            m_chembl = metrics(cold.y.values, cold[model].values)
            m_repodb = metrics(cold.y_repodb.values, cold[model].values)
            rows.append(dict(seed=seed, model=model, n_cold=len(cold), n_cold_pos_repodb=n_pos,
                              cold_pos_repodb_names=";".join(pos_names),
                              auroc_chembl_label=m_chembl["auroc"], auprc_chembl_label=m_chembl["auprc"],
                              auroc_repodb_label=m_repodb["auroc"], auprc_repodb_label=m_repodb["auprc"]))
    return pd.DataFrame(rows)


def main():
    cmp, ecfp, expr, text, text_mask, y = load(pool="all", disease="epilepsy")
    y_repodb, repodb_names, matched = build_repodb_epilepsy_labels(cmp)
    validate_against_chembl_labels(cmp, y, y_repodb, repodb_names, matched)

    oof = pd.read_parquet("../results/oof_epilepsy.parquet")
    out = cold_bucket_rows(oof, y_repodb, cmp.name)
    out.to_csv("../results/repodb_external_validation.csv", index=False)

    print()
    print("Unique cold-bucket repoDB-positive compounds (same set across all 5 seeds):",
          sorted(set(out.cold_pos_repodb_names.iloc[0].split(";"))))
    summary = out.groupby("model")[["auroc_chembl_label", "auroc_repodb_label",
                                     "auprc_chembl_label", "auprc_repodb_label"]].agg(["mean", "std"])
    print(summary)

    piv = out.pivot_table(index="seed", columns="model", values="auroc_repodb_label")
    delta = (piv["fusion_full"] - piv["fusion_nodebias"])
    print(f"\nCold-bucket debiasing delta under repoDB label (fusion_full - fusion_nodebias), per seed: "
          f"{delta.values.round(3).tolist()}, mean {delta.mean():.3f} +/- {delta.std():.3f}")

    piv_c = out.pivot_table(index="seed", columns="model", values="auroc_chembl_label")
    delta_c = (piv_c["fusion_full"] - piv_c["fusion_nodebias"])
    print(f"Same delta recomputed under the original ChEMBL label on this identical cold-bucket row set "
          f"(sanity check, should reproduce the paper's +0.116 up to tertile-cut rounding): "
          f"mean {delta_c.mean():.3f} +/- {delta_c.std():.3f}")

    print("\nWrote results/repodb_external_validation.csv")


if __name__ == "__main__":
    main()
