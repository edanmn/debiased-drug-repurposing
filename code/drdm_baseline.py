"""Adapted baseline comparison against DRDM (Meng et al., J. Biomedical Informatics 2025,
doi:10.1016/j.jbi.2025.104843): a GNN-based drug repositioning method that counters
node-degree ("popularity") bias in a drug-disease association graph by dynamically
reweighting edges during training, lifting the contribution of long-tail (low-degree)
entities.

DRDM's own architecture (dual-view contrastive GNN over curated drug-disease benchmarks
such as Gdataset/Cdataset/DRRS, which provide thousands of diseases) does not transfer
as-is to this paper's setting: a single disease axis per task and chemical/transcriptomic/
text drug features rather than a large pre-built association graph. We therefore implement
DRDM's core mechanism, not its literal architecture: a lightweight bipartite drug-disease
graph built from the nine indication label sets already used elsewhere in this project,
with a 2-hop graph-convolution embedding model trained under an annealed inverse-degree
edge reweighting (low-degree drugs/diseases get up-weighted over training, exactly DRDM's
stated goal of "enhancing long-tail entity representation"). This is a fair, clearly-scoped
test of whether DRDM's debiasing *principle* recovers the paper's epilepsy cold-bucket
signal, not a reproduction of their reported numbers on their own benchmarks.

Leakage control: the epilepsy test-fold edges are excluded from the graph for that fold
(so the epilepsy disease embedding is built only from train-fold positives); the other
eight diseases' full label sets are used as side information for every fold, since those
tasks are not being evaluated here. Drug embeddings come from a feature-based encoder
(same three views as the paper's main pipeline) run on train+test compounds alike; only
the epilepsy *labels* of test compounds are ever withheld.

Writes: results/drdm_baseline.csv (per-seed cold-bucket and overall AUROC/AUPRC, matching
the format of repodb_external_validation.csv so it can be merged into the same comparison).
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from datalib import load, metrics, D
from fusion_model import ViewEncoder, to_tensor, DEVICE

NINE_CLASSES = [
    "epilepsy", "antibacterial", "antidiabetic", "antineoplastic", "antipsychotic",
    "betablocker", "depression", "lipid", "nsaid",
]
EMB_DIM = 32
N_EPOCHS = 150
ALPHA_MAX = 0.5
NEG_PER_POS = 5


class DrugEncoder(nn.Module):
    """Simple concatenation of the three view encoders (no gating: this is a baseline
    being compared against, not the paper's proposed fusion architecture)."""

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


class DRDMGraph(nn.Module):
    """Two-hop bipartite graph convolution over a (n_drugs x n_diseases) association
    matrix, with disease embeddings as a learned table and drug embeddings produced by
    DrugEncoder. Scoring is a bilinear dot product."""

    def __init__(self, n_diseases, emb_dim=EMB_DIM):
        super().__init__()
        self.drug_enc = DrugEncoder(emb_dim)
        self.disease_emb = nn.Embedding(n_diseases, emb_dim)
        self.proj_d2i = nn.Linear(emb_dim, emb_dim)  # disease -> drug message
        self.proj_i2d = nn.Linear(emb_dim, emb_dim)  # drug -> disease message

    def propagate(self, drug0, disease0, A, w):
        """A: (n_drug, n_disease) binary adjacency restricted to visible edges.
        w: same shape, edge weight (0 where A==0)."""
        drug1 = drug0 + self.proj_d2i((A * w) @ disease0)
        disease1 = disease0 + self.proj_i2d((A * w).t() @ drug0)
        drug2 = drug1 + self.proj_d2i((A * w) @ disease1)
        disease2 = disease1 + self.proj_i2d((A * w).t() @ drug1)
        return drug2, disease2


def build_adjacency(Y_visible):
    """Y_visible: (n_drug, n_disease) {0,1} array with the held-out fold's epilepsy
    column already zeroed out. Returns torch tensors (A, deg_drug, deg_disease)."""
    A = torch.as_tensor(Y_visible, dtype=torch.float32, device=DEVICE)
    deg_drug = A.sum(1, keepdim=True).clamp(min=1.0)
    deg_disease = A.sum(0, keepdim=True).clamp(min=1.0)
    return A, deg_drug, deg_disease


def run_fold(cmp, ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed):
    torch.manual_seed(seed)
    n_disease = Y.shape[1]
    Y_visible = Y.copy()
    Y_visible[te_idx, epi_col] = 0  # exclude held-out epilepsy edges from the graph

    model = DRDMGraph(n_disease).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ecfp_t, expr_t = to_tensor(ecfp), to_tensor(expr)
    text_t, mask_t = to_tensor(text), to_tensor(text_mask.astype(np.float32))

    A, deg_drug, deg_disease = build_adjacency(Y_visible)
    pos_i, pos_j = torch.nonzero(A, as_tuple=True)
    n_pos = len(pos_i)
    rng = np.random.default_rng(seed)
    n_drug = Y.shape[0]

    for epoch in range(N_EPOCHS):
        model.train()
        alpha = ALPHA_MAX * epoch / max(1, N_EPOCHS - 1)  # annealed 0 -> 0.5
        w = (deg_drug * deg_disease).pow(-alpha)
        drug0 = model.drug_enc(ecfp_t, expr_t, text_t, mask_t)
        disease0 = model.disease_emb.weight
        drug2, disease2 = model.propagate(drug0, disease0, A, w)

        neg_i = torch.as_tensor(rng.integers(0, n_drug, size=n_pos * NEG_PER_POS), device=DEVICE)
        neg_j = torch.as_tensor(rng.integers(0, n_disease, size=n_pos * NEG_PER_POS), device=DEVICE)

        pos_score = (drug2[pos_i] * disease2[pos_j]).sum(1)
        neg_score = (drug2[neg_i] * disease2[neg_j]).sum(1)
        pos_w = w[pos_i, pos_j].detach()

        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss_pos = (bce(pos_score, torch.ones_like(pos_score)) * pos_w).sum() / pos_w.sum()
        loss_neg = bce(neg_score, torch.zeros_like(neg_score)).mean()
        loss = loss_pos + loss_neg
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        w_final = (deg_drug * deg_disease).pow(-ALPHA_MAX)
        drug0 = model.drug_enc(ecfp_t, expr_t, text_t, mask_t)
        disease0 = model.disease_emb.weight
        drug2, disease2 = model.propagate(drug0, disease0, A, w_final)
        scores = (drug2[te_idx] * disease2[epi_col].unsqueeze(0)).sum(1)
    return torch.sigmoid(scores).cpu().numpy()


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
        rows.append(dict(seed=seed, model="drdm_adapted", n_cold=int(cold.sum()),
                          auroc_cold=m_cold["auroc"], auprc_cold=m_cold["auprc"],
                          auroc_all=m_all["auroc"], auprc_all=m_all["auprc"]))
        print(f"seed={seed} DONE cold n={cold.sum()} auroc_cold={m_cold['auroc']:.3f} "
              f"auroc_all={m_all['auroc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("../results/drdm_baseline.csv", index=False)
    print(out)
    print(f"\nMean cold-bucket AUROC: {out.auroc_cold.mean():.3f} +/- {out.auroc_cold.std():.3f}")
    print("Wrote results/drdm_baseline.csv")


if __name__ == "__main__":
    main()
