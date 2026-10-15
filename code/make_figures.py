"""Produce the headline figures from results/ CSVs. Saved to figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
FIGS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/figures"

MODEL_LABELS = {
    "random": "Random", "nn_tanimoto": "NN-Tanimoto (zero-train)", "lr_ecfp": "LR (ECFP4)",
    "xgb_ecfp": "XGBoost (ECFP4)", "text_only": "Target-text only", "fusion_struct": "Fusion: struct only",
    "fusion_struct_expr": "Fusion: struct+L1000", "fusion_nodebias": "Fusion: full, no debias",
    "fusion_full": "Fusion: full, debiased (PROPOSED)",
}


def fig_warm_cold():
    df = pd.read_csv(f"{RESULTS}/warm_cold_auroc.csv")
    models = ["nn_tanimoto", "xgb_ecfp", "fusion_nodebias", "fusion_full"]
    diseases = sorted(df.disease.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, bucket in zip(axes, ["warm", "cold"]):
        sub = df[df.bucket == bucket]
        x = np.arange(len(diseases))
        width = 0.2
        for i, m in enumerate(models):
            vals = [sub[(sub.disease == d) & (sub.model == m)].auroc.mean() for d in diseases]
            ax.bar(x + i * width, vals, width, label=MODEL_LABELS.get(m, m))
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(diseases, rotation=45, ha="right")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{bucket.capitalize()} (NN-similarity tertile)")
        ax.set_ylabel("AUROC" if bucket == "warm" else "")
    axes[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/warm_cold_auroc.png", dpi=150)
    plt.close()


def fig_similarity_stratified():
    df = pd.read_csv(f"{RESULTS}/similarity_quintile_auroc.csv")
    models = ["nn_tanimoto", "xgb_ecfp", "fusion_nodebias", "fusion_full"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        sub = df[df.model == m].groupby("sim_quintile").auroc.mean()
        ax.plot(sub.index, sub.values, marker="o", label=MODEL_LABELS.get(m, m))
    ax.set_xlabel("NN-Tanimoto-similarity-to-train quintile (0=lowest/coldest)")
    ax.set_ylabel("AUROC (averaged over 9 classes)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Performance vs. continuous train-test similarity")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/similarity_stratified_auroc.png", dpi=150)
    plt.close()


def fig_headline_bar():
    df = pd.read_csv(f"{RESULTS}/headline_mean_std_auroc.csv")
    models = ["random", "nn_tanimoto", "lr_ecfp", "xgb_ecfp", "text_only",
              "fusion_struct", "fusion_struct_expr", "fusion_nodebias", "fusion_full"]
    agg = df.groupby("model").apply(lambda g: pd.Series({
        "mean": np.average(g.mean_auroc, weights=g.n_seeds),
        "std": np.sqrt(np.average(g.std_auroc ** 2, weights=g.n_seeds)),
    })).reindex(models)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(models)), agg["mean"], yerr=agg["std"], capsize=4,
           color=["#999" if m != "fusion_full" else "#d62728" for m in models])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean AUROC across 9 classes (+/- std across seeds)")
    ax.set_title("Headline comparison, averaged across all 9 therapeutic classes")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/headline_comparison.png", dpi=150)
    plt.close()


def fig_leakage():
    try:
        df = pd.read_csv(f"{RESULTS}/leakage_audit_summary.csv")
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(df.condition, df.auroc, color=["#d62728", "#1f77b4"])
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("5-fold CV AUROC: predicting AED label from pre-approval PubMed text")
    ax.set_title("PubMed-abstract label leakage audit")
    for i, v in enumerate(df.auroc):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIGS}/leakage_audit.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_warm_cold()
    fig_similarity_stratified()
    fig_headline_bar()
    fig_leakage()
    print("Wrote figures to", FIGS)
