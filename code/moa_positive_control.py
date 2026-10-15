"""Biological-interpretation positive control (per scientific-paper-rigor review item: "no
biological interpretation" is a common RECOMB rejection reason). Checks whether debiasing's
reranking of cold-bucket (bottom-tertile NN-similarity), non-AED epilepsy candidates is enriched
for pharmacologically plausible antiseizure mechanisms, using the Broad Repurposing Hub's own
mechanism-of-action (MoA) field (data/targets.parquet, built from the Hub join, independent of
anything the model was trained on: the model never sees this field).

Validation of the keyword set itself (printed, not just asserted): of the 18 known AEDs
(data/known_aeds.csv) that resolve in the Hub MoA table, 14/18 (78%) match one of the keywords
below, against a 7.8% base rate in the full 2,290-compound MoA table (179/2290). This is reported
honestly as a validation step, not assumed.

Writes: results/moa_enrichment.csv (per-seed top-50 hit rate for nn_tanimoto, fusion_nodebias,
fusion_full, vs. the cold-bucket candidate-pool base rate).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from datalib import load, D

KEYWORDS = ["gaba", "sodium channel", "calcium channel", "glutamate", "potassium channel", "sv2a"]
TOPK = 50
MODELS = ["nn_tanimoto", "fusion_nodebias", "fusion_full"]


def moa_hit_mask(cmp):
    targets = pd.read_parquet(D + "targets.parquet")[["ik14", "moa"]]
    moa_by_ik = targets.set_index("ik14").moa.str.lower()
    moa = cmp.ik14.map(moa_by_ik)
    return moa.str.contains("|".join(KEYWORDS), na=False).values


def validate_keyword_set():
    targets = pd.read_parquet(D + "targets.parquet")
    aeds = pd.read_csv(D + "known_aeds.csv")
    targets["name_lc"] = targets.name.str.lower().str.strip()
    aeds["name_lc"] = aeds.drug_name.str.lower().str.strip()
    m = targets.merge(aeds, on="name_lc", how="inner")
    hit = m.moa.str.lower().str.contains("|".join(KEYWORDS), na=False)
    base = targets.moa.str.lower().str.contains("|".join(KEYWORDS), na=False)
    print(f"Keyword-set validation: {hit.sum()}/{len(m)} resolved known AEDs match "
          f"({hit.mean():.1%}) vs {base.sum()}/{len(targets)} base rate in full MoA table "
          f"({base.mean():.1%})")
    return hit.mean(), base.mean()


def main():
    validate_keyword_set()

    cmp, ecfp, expr, text, text_mask, y = load(pool="all", disease="epilepsy")
    moa_hit = moa_hit_mask(cmp)

    oof = pd.read_parquet("../results/oof_epilepsy.parquet")
    nn_sim = oof["nn_sim"].values
    valid = nn_sim >= 0
    terts = np.nanpercentile(nn_sim[valid], [33.33, 66.67])
    cold_cut = terts[0]

    rows = []
    for seed, g in oof.groupby("seed"):
        g = g[(g.nn_sim <= cold_cut) & (g.y == 0)]
        if len(g) < TOPK:
            continue
        cand_idx = g["idx"].values
        base_rate = moa_hit[cand_idx].mean()
        for model in MODELS:
            order = cand_idx[np.argsort(-g[model].values)]
            top = order[:TOPK]
            rows.append(dict(seed=seed, model=model, top50_moa_rate=moa_hit[top].mean(),
                              cold_base_rate=base_rate, n_cold_candidates=len(g)))

    out = pd.DataFrame(rows)
    out.to_csv("../results/moa_enrichment.csv", index=False)
    print(out.groupby("model")[["top50_moa_rate", "cold_base_rate"]].agg(["mean", "std"]))
    print("\nWrote results/moa_enrichment.csv")


if __name__ == "__main__":
    main()
