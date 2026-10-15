"""Adapted baseline comparison against TAPB (Lin et al., Nature Communications 2025,
doi:10.1038/s41467-025-66915-1): an interventional/causal debiasing framework for
drug-target interaction prediction. TAPB identifies "target prior bias" (label-imbalance
specific to a target, i.e. how skewed its positive rate is) as a confound, builds a
dictionary of confounder clusters via K-means over target embeddings, and corrects
predictions with a backdoor adjustment, P(Y|D,do(T)) = sum_i P(c_i) Softmax(g_y(F_{c_i})):
a shared classifier g_y applied to confounder-cluster-aligned segments of the fused
representation, combined with fixed cluster-population weights P(c_i).

TAPB's own architecture (a SMILES-BERT drug encoder, frozen ESM-2 protein-target
embeddings, a K-means dictionary over those embeddings, and a multi-head cross-attention
Confounder Alignment Module, CAM) does not transfer to this paper's task: there is no
protein-target axis here, and TAPB's released code (github.com/GaomingL1n/TAPB) is built
entirely around protein-sequence input we do not have. We instead implement TAPB's
backdoor-adjustment *mechanism*, adapted so it applies to this paper's nine-indication
label structure, which is the closest available analogue to "multiple targets with
different prior tendencies":

  - Per indication T, compute TAPB's own prior-tendency score z_T (mean positive rate),
    exactly as defined in the TAPB paper. For epilepsy, z_T uses ONLY the training fold's
    labels (never the held-out test labels); for the other eight indications, which are
    not evaluated here, z_T uses the full label set as side information.
  - K-means (I=3) over the nine z_T values gives a small confounder dictionary, with
    P(c_i) the fraction of the nine indications assigned to cluster i (equivalent to
    population weighting, since every indication contributes the same number of drug
    pairs).
  - A softmin cluster-alignment weight a_i(T), the lightweight analogue of CAM's
    attention, replaces attention-over-target-embeddings with attention over the scalar
    prior-tendency score.
  - A single shared classifier g_y takes [drug_embedding, a_i(T)] for each cluster i and
    produces a per-cluster logit; the final backdoor-adjusted prediction sums
    P(c_i)*sigmoid(logit_i), mirroring the paper's Eq. 22 while using our existing
    three-view (ECFP/L1000/text) drug encoder in place of their SMILES-BERT branch.
  - The drug encoder and g_y are trained jointly across all nine indications' full
    (drug, label) pairs except epilepsy's held-out test rows, a standard multi-task
    side-information setup; the amino-acid randomization augmentation in the original
    paper is specific to protein-sequence input and has no analogue here, so it is
    omitted.

This tests TAPB's bias-correction principle on this paper's actual task and data, not a
reproduction of its reported DTI benchmark numbers.

Writes: results/tapb_baseline.csv (per-seed cold-bucket and overall AUROC/AUPRC, same
format as drdm_baseline.csv and repodb_external_validation.csv).
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

sys.path.insert(0, ".")
from datalib import load, metrics
from fusion_model import ViewEncoder, to_tensor, DEVICE

NINE_CLASSES = [
    "epilepsy", "antibacterial", "antidiabetic", "antineoplastic", "antipsychotic",
    "betablocker", "depression", "lipid", "nsaid",
]
EMB_DIM = 48
N_CLUSTERS = 3
N_EPOCHS = 150
SOFTMIN_TAU = 0.15


class DrugEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.e_ecfp = ViewEncoder(2048, 128, emb_dim)
        self.e_expr = ViewEncoder(978, 128, emb_dim)
        self.e_text = ViewEncoder(768, 128, emb_dim)
        self.proj = nn.Linear(emb_dim * 3, emb_dim)

    def forward(self, ecfp, expr, text, text_mask):
        e1 = self.e_ecfp(ecfp)
        e2 = self.e_expr(expr)
        e3 = self.e_text(text) * text_mask.unsqueeze(-1)
        return self.proj(torch.cat([e1, e2, e3], dim=1))


class TAPBModel(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, n_clusters=N_CLUSTERS):
        super().__init__()
        self.drug_enc = DrugEncoder(emb_dim)
        self.g_y = nn.Sequential(nn.Linear(emb_dim + 1, 32), nn.ReLU(), nn.Linear(32, 1))

    def cluster_logits(self, drug_emb, align_weights):
        """align_weights: (n_clusters,) alignment of ONE indication to each cluster.
        Returns per-cluster logits, shape (n_drug, n_clusters)."""
        n_drug, n_c = drug_emb.shape[0], align_weights.shape[0]
        a = align_weights.view(1, n_c, 1).expand(n_drug, n_c, 1)
        d = drug_emb.unsqueeze(1).expand(n_drug, n_c, drug_emb.shape[1])
        inp = torch.cat([d, a], dim=-1)
        return self.g_y(inp).squeeze(-1)

    def predict(self, drug_emb, align_weights, pop_weights):
        logits = self.cluster_logits(drug_emb, align_weights)
        probs = torch.sigmoid(logits)
        return (probs * pop_weights.view(1, -1)).sum(1)


def prior_tendency(y):
    return float(np.mean(y))


def build_confounder_dictionary(z_values):
    z = np.asarray(z_values, dtype=np.float64).reshape(-1, 1)
    zmin, zmax = z.min(), z.max()
    z_norm = (z - zmin) / (zmax - zmin + 1e-12)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit(z_norm)
    centers = km.cluster_centers_.ravel()
    pop = np.array([(km.labels_ == i).mean() for i in range(N_CLUSTERS)])
    return z_norm.ravel(), centers, pop


def alignment_weights(z_i, centers):
    d2 = (z_i - centers) ** 2
    w = np.exp(-d2 / SOFTMIN_TAU)
    return w / w.sum()


def run_fold(cmp, ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed):
    torch.manual_seed(seed)
    n_disease = Y.shape[1]
    z_raw = []
    for k in range(n_disease):
        if k == epi_col:
            z_raw.append(prior_tendency(Y[tr_idx, k]))
        else:
            z_raw.append(prior_tendency(Y[:, k]))
    z_norm, centers, pop = build_confounder_dictionary(z_raw)
    align = np.stack([alignment_weights(z_norm[k], centers) for k in range(n_disease)])

    model = TAPBModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ecfp_t, expr_t = to_tensor(ecfp), to_tensor(expr)
    text_t, mask_t = to_tensor(text), to_tensor(text_mask.astype(np.float32))
    pop_t = to_tensor(pop)
    align_t = to_tensor(align)

    for epoch in range(N_EPOCHS):
        model.train()
        drug_emb = model.drug_enc(ecfp_t[tr_idx], expr_t[tr_idx], text_t[tr_idx], mask_t[tr_idx])
        opt.zero_grad()
        total_loss = 0.0
        for k in range(n_disease):
            y_k = Y[tr_idx, k]
            pred = model.predict(drug_emb, align_t[k], pop_t)
            y_t = to_tensor(y_k)
            pos_frac = max(y_k.mean(), 1e-3)
            w = torch.where(y_t > 0.5, 1.0 / pos_frac, 1.0 / (1 - pos_frac))
            loss_k = (nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y_t, reduction="none") * w).mean()
            total_loss = total_loss + loss_k
        (total_loss / n_disease).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        drug_emb_te = model.drug_enc(ecfp_t[te_idx], expr_t[te_idx], text_t[te_idx], mask_t[te_idx])
        pred_te = model.predict(drug_emb_te, align_t[epi_col], pop_t)
    return pred_te.cpu().numpy()


def main():
    cmp, ecfp, expr, text, text_mask, y_epi = load(pool="all", disease="epilepsy")
    Y = np.zeros((len(cmp), len(NINE_CLASSES)), dtype=np.float32)
    Y[:, 0] = y_epi
    for k, d in enumerate(NINE_CLASSES[1:], start=1):
        _, _, _, _, _, yk = load(pool="all", disease=d)
        Y[:, k] = yk
    epi_col = 0

    oof = pd.read_parquet("../results/oof_epilepsy.parquet")
    rows = []
    for seed, g in oof.groupby("seed"):
        full_scores = np.full(len(cmp), np.nan)
        for fold, gf in g.groupby("fold"):
            te_idx = gf["idx"].values
            tr_mask = np.ones(len(cmp), dtype=bool)
            tr_mask[te_idx] = False
            tr_idx = np.nonzero(tr_mask)[0]
            s = run_fold(cmp, ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed=seed)
            full_scores[te_idx] = s
            print(f"seed={seed} fold={fold} n_te={len(te_idx)} auroc_fold="
                  f"{metrics(gf['y'].values, s)['auroc']:.3f}", flush=True)

        nn_sim = g.set_index("idx")["nn_sim"].reindex(np.arange(len(cmp))).values
        y_all = g.set_index("idx")["y"].reindex(np.arange(len(cmp))).values
        valid = ~np.isnan(nn_sim) & ~np.isnan(full_scores)
        cold_cut = np.nanpercentile(nn_sim[valid], 33.33)
        cold = valid & (nn_sim <= cold_cut)
        m_cold = metrics(y_all[cold].astype(int), full_scores[cold])
        m_all = metrics(y_all[valid].astype(int), full_scores[valid])
        rows.append(dict(seed=seed, model="tapb_adapted", n_cold=int(cold.sum()),
                          auroc_cold=m_cold["auroc"], auprc_cold=m_cold["auprc"],
                          auroc_all=m_all["auroc"], auprc_all=m_all["auprc"]))
        print(f"seed={seed} DONE cold n={cold.sum()} auroc_cold={m_cold['auroc']:.3f} "
              f"auroc_all={m_all['auroc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("../results/tapb_baseline.csv", index=False)
    print(out)
    print(f"\nMean cold-bucket AUROC: {out.auroc_cold.mean():.3f} +/- {out.auroc_cold.std():.3f}")
    print("Wrote results/tapb_baseline.csv")


if __name__ == "__main__":
    main()
