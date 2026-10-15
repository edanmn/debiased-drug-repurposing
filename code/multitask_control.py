"""Control for tapb_baseline.py: the TAPB-adapted model's strong cold-bucket AUROC could
come from TAPB's confounder-cluster backdoor adjustment, or simply from training the drug
encoder jointly across all nine indications (extra shared-representation signal that our
paper's own models never get, since run_pipeline.py trains each indication independently).
This script isolates that: same drug encoder, same nine-indication joint training loop,
same per-epoch optimization, but a single plain linear head per indication instead of
TAPB's confounder-mixture correction. If this control matches tapb_baseline.py's AUROC,
the strong result there is a multi-task-learning effect, not evidence that TAPB's
debiasing mechanism specifically helps.

Writes: results/multitask_control.csv (same format as tapb_baseline.csv/drdm_baseline.csv).
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from datalib import load, metrics
from fusion_model import to_tensor, DEVICE
from tapb_baseline import NINE_CLASSES, N_EPOCHS, DrugEncoder


class PlainMultitask(nn.Module):
    def __init__(self, emb_dim, n_disease):
        super().__init__()
        self.drug_enc = DrugEncoder(emb_dim)
        self.heads = nn.ModuleList([nn.Linear(emb_dim, 1) for _ in range(n_disease)])

    def forward(self, ecfp, expr, text, text_mask, k):
        emb = self.drug_enc(ecfp, expr, text, text_mask)
        return self.heads[k](emb).squeeze(-1)


def run_fold(cmp, ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed):
    torch.manual_seed(seed)
    n_disease = Y.shape[1]
    model = PlainMultitask(48, n_disease).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ecfp_t, expr_t = to_tensor(ecfp), to_tensor(expr)
    text_t, mask_t = to_tensor(text), to_tensor(text_mask.astype(np.float32))

    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for k in range(n_disease):
            y_k = Y[tr_idx, k]
            logit = model(ecfp_t[tr_idx], expr_t[tr_idx], text_t[tr_idx], mask_t[tr_idx], k)
            y_t = to_tensor(y_k)
            pos_frac = max(y_k.mean(), 1e-3)
            w = torch.where(y_t > 0.5, 1.0 / pos_frac, 1.0 / (1 - pos_frac))
            loss_k = (nn.functional.binary_cross_entropy_with_logits(logit, y_t, reduction="none") * w).mean()
            total_loss = total_loss + loss_k
        (total_loss / n_disease).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logit_te = model(ecfp_t[te_idx], expr_t[te_idx], text_t[te_idx], mask_t[te_idx], epi_col)
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
            s = run_fold(cmp, ecfp, expr, text, text_mask, Y, epi_col, tr_idx, te_idx, seed=seed)
            full_scores[te_idx] = s
            print(f"seed={seed} fold={fold} auroc_fold={metrics(gf['y'].values, s)['auroc']:.3f}", flush=True)

        nn_sim = g.set_index("idx")["nn_sim"].reindex(np.arange(len(cmp))).values
        y_all = g.set_index("idx")["y"].reindex(np.arange(len(cmp))).values
        valid = ~np.isnan(nn_sim) & ~np.isnan(full_scores)
        cold_cut = np.nanpercentile(nn_sim[valid], 33.33)
        cold = valid & (nn_sim <= cold_cut)
        m_cold = metrics(y_all[cold].astype(int), full_scores[cold])
        m_all = metrics(y_all[valid].astype(int), full_scores[valid])
        rows.append(dict(seed=seed, model="multitask_control", n_cold=int(cold.sum()),
                          auroc_cold=m_cold["auroc"], auprc_cold=m_cold["auprc"],
                          auroc_all=m_all["auroc"], auprc_all=m_all["auprc"]))
        print(f"seed={seed} DONE cold auroc_cold={m_cold['auroc']:.3f} auroc_all={m_all['auroc']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("../results/multitask_control.csv", index=False)
    print(out)
    print(f"\nMean cold-bucket AUROC: {out.auroc_cold.mean():.3f} +/- {out.auroc_cold.std():.3f}")


if __name__ == "__main__":
    main()
