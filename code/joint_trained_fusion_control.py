"""Second instance of the joint-training confound found in Section 5.7 (tapb_baseline.py /
multitask_control.py): does training jointly across all nine indications inflate cold-bucket
AUROC even with no correction mechanism at all, on the PAPER'S OWN undebiased architecture,
not just on TAPB's simpler drug encoder?

multitask_control.py cannot answer this cleanly: it reuses TAPB's DrugEncoder (concatenate-
then-project fusion), not this paper's real GatedFusion (learned per-sample gate), so a
comparison against fusion_nodebias confounds training regime with architecture. This script
removes that confound: JointGatedFusion below is fusion_model.GatedFusion's exact encoder and
gate, just with nine independent linear heads sharing one encoder instead of one head trained
on one indication at a time. Same emb_dim (64), same epoch count (60), same optimizer (Adam,
lr=1e-3, weight_decay=1e-4), same uniform (non-class-balanced) sample weights as
run_pipeline.py's fusion_nodebias, and the same train-fold-only standardization of the
expr/text views (see RESULTS.md Section 2's note on why unnormalized continuous views break
this architecture). Only the training regime (joint vs. per-task) differs.

Writes: results/joint_trained_fusion_control.csv
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from datalib import load, standardize_view, metrics
from fusion_model import ViewEncoder, to_tensor, DEVICE

NINE_CLASSES = [
    "epilepsy", "antibacterial", "antidiabetic", "antineoplastic", "antipsychotic",
    "betablocker", "depression", "lipid", "nsaid",
]
EMB_DIM = 64
N_EPOCHS = 60


class JointGatedFusion(nn.Module):
    """fusion_model.GatedFusion's encoder and gate, shared across indications, with one
    linear head per indication in place of GatedFusion's single head."""

    def __init__(self, n_disease, emb_dim=EMB_DIM):
        super().__init__()
        self.enc_ecfp = ViewEncoder(2048, 128, emb_dim)
        self.enc_expr = ViewEncoder(978, 128, emb_dim)
        self.enc_text = ViewEncoder(768, 128, emb_dim)
        self.gate = nn.Sequential(nn.Linear(emb_dim * 3, 32), nn.ReLU(), nn.Linear(32, 3))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(n_disease)
        ])

    def encode(self, ecfp, expr, text, text_mask):
        e1 = self.enc_ecfp(ecfp)
        e2 = self.enc_expr(expr)
        e3 = self.enc_text(text)
        cat = torch.cat([e1, e2, e3], dim=1)
        gate_logits = self.gate(cat)
        avail = torch.stack([torch.ones_like(text_mask), torch.ones_like(text_mask), text_mask], dim=1)
        gate_logits = gate_logits.masked_fill(avail == 0, -1e9)
        gate_w = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack([e1, e2, e3], dim=1)
        return (stacked * gate_w.unsqueeze(-1)).sum(dim=1)

    def forward(self, ecfp, expr, text, text_mask, k):
        fused = self.encode(ecfp, expr, text, text_mask)
        return self.heads[k](fused).squeeze(-1)


def run_fold(ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed):
    torch.manual_seed(seed)
    n_disease = Y.shape[1]
    model = JointGatedFusion(n_disease).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    expr_tr_s, expr_te_s = standardize_view(expr[tr_idx], expr[te_idx])
    text_tr_s, text_te_s = standardize_view(text[tr_idx], text[te_idx], text_mask[tr_idx], text_mask[te_idx])

    ecfp_t = to_tensor(ecfp[tr_idx])
    expr_t = to_tensor(expr_tr_s)
    text_t = to_tensor(text_tr_s)
    mask_t = to_tensor(text_mask[tr_idx].astype(np.float32))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for k in range(n_disease):
            y_k = Y[tr_idx, k]
            y_t = to_tensor(y_k)
            logit = model(ecfp_t, expr_t, text_t, mask_t, k)
            w_t = torch.ones_like(y_t)  # uniform weights, matching fusion_nodebias (no debiasing)
            loss_k = (bce(logit, y_t) * w_t).mean()
            total_loss = total_loss + loss_k
        (total_loss / n_disease).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ecfp_te_t = to_tensor(ecfp[te_idx])
        expr_te_t = to_tensor(expr_te_s)
        text_te_t = to_tensor(text_te_s)
        mask_te_t = to_tensor(text_mask[te_idx].astype(np.float32))
        logit_te = model(ecfp_te_t, expr_te_t, text_te_t, mask_te_t, epi_col)
    return torch.sigmoid(logit_te).cpu().numpy()


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
            s = run_fold(ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed=seed)
            full_scores[te_idx] = s
            print(f"seed={seed} fold={fold} auroc_fold={metrics(gf['y'].values, s)['auroc']:.3f}", flush=True)

        nn_sim = g.set_index("idx")["nn_sim"].reindex(np.arange(len(cmp))).values
        y_all = g.set_index("idx")["y"].reindex(np.arange(len(cmp))).values
        valid = ~np.isnan(nn_sim) & ~np.isnan(full_scores)
        cold_cut = np.nanpercentile(nn_sim[valid], 33.33)
        cold = valid & (nn_sim <= cold_cut)
        m_cold = metrics(y_all[cold].astype(int), full_scores[cold])
        m_all = metrics(y_all[valid].astype(int), full_scores[valid])
        rows.append(dict(seed=seed, model="joint_trained_fusion_nodebias", n_cold=int(cold.sum()),
                          auroc_cold=m_cold["auroc"], auprc_cold=m_cold["auprc"],
                          auroc_all=m_all["auroc"], auprc_all=m_all["auprc"]))
        print(f"seed={seed} DONE cold auroc_cold={m_cold['auroc']:.3f} auroc_all={m_all['auroc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("../results/joint_trained_fusion_control.csv", index=False)
    print(out)
    print(f"\nMean cold-bucket AUROC: {out.auroc_cold.mean():.3f} +/- {out.auroc_cold.std():.3f}")


if __name__ == "__main__":
    main()
