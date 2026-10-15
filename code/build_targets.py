"""Build data/targets.parquet: join Repurposing Hub target annotations onto compounds.parquet by name.

Join key: lowercased, whitespace-stripped pert_iname vs compounds.parquet `name`.
Keeps only compounds with a non-null `target` field in the Hub file.
"""
import pandas as pd
import re

HUB = "/mnt/c/Users/rishik/epilepsy_repurpose/PACT/PACT/data/repurposing_drugs_20200324.txt"
CMP = "/mnt/c/Users/rishik/epilepsy_repurpose/ICHI2027_mgan/data/compounds.parquet"
OUT = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/data/targets.parquet"


def norm(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    rh = pd.read_csv(HUB, sep="\t", skiprows=9)
    print("Repurposing Hub rows:", len(rh))
    rh = rh.dropna(subset=["target"]).copy()
    print("Repurposing Hub rows with non-null target:", len(rh))
    rh["name_norm"] = rh["pert_iname"].map(norm)
    # some drugs appear twice (duplicate pert_iname) with different annotations - keep first, merge targets
    rh_g = rh.groupby("name_norm").agg({
        "pert_iname": "first",
        "clinical_phase": "first",
        "moa": lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
        "target": lambda x: "|".join(sorted(set("|".join(x.dropna().astype(str)).split("|")))),
        "disease_area": "first",
        "indication": "first",
    }).reset_index()

    cmp = pd.read_parquet(CMP)
    print("compounds.parquet rows:", len(cmp))
    cmp["name_norm"] = cmp["name"].map(norm)

    merged = cmp.merge(rh_g, on="name_norm", how="inner", suffixes=("", "_hub"))
    print("Joined rows (compounds with Hub target annotation):", len(merged))
    print("Unique ik14 after join:", merged["ik14"].nunique())

    # dedupe in case of multiple name collisions mapping to same ik14 (unlikely) - keep first
    merged = merged.drop_duplicates(subset=["ik14"]).reset_index(drop=True)
    print("After dedup on ik14:", len(merged))

    out = merged[["ik14", "name", "target", "moa", "disease_area", "indication", "clinical_phase"]].copy()
    out = out.rename(columns={"target": "target_genes"})

    all_genes = set()
    for g in out["target_genes"].dropna():
        all_genes.update(g.split("|"))
    all_genes = sorted(all_genes)
    print("Unique target gene symbols:", len(all_genes))

    out.to_parquet(OUT, index=False)
    print("Wrote", OUT)

    with open("/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/data/unique_target_genes.txt", "w") as f:
        f.write("\n".join(all_genes))
    print("Wrote unique_target_genes.txt")


if __name__ == "__main__":
    main()
