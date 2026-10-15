"""Gated multi-view fusion model: structure (ECFP4), L1000 expression, and target-function
text views, combined via a learned per-sample gate. Trained with DebiasedDTA-style
importance-reweighted BCE loss (see guide.py for the weight schedule).
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ViewEncoder(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, use_ecfp=True, use_expr=True, use_text=True, emb_dim=64):
        super().__init__()
        self.use_ecfp, self.use_expr, self.use_text = use_ecfp, use_expr, use_text
        self.emb_dim = emb_dim
        self.n_views = sum([use_ecfp, use_expr, use_text])
        if use_ecfp:
            self.enc_ecfp = ViewEncoder(2048, 128, emb_dim)
        if use_expr:
            self.enc_expr = ViewEncoder(978, 128, emb_dim)
        if use_text:
            self.enc_text = ViewEncoder(768, 128, emb_dim)
        if self.n_views > 1:
            self.gate = nn.Sequential(
                nn.Linear(emb_dim * self.n_views, 32), nn.ReLU(), nn.Linear(32, self.n_views),
            )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, ecfp=None, expr=None, text=None, text_mask=None):
        embs, avail = [], []
        if self.use_ecfp:
            embs.append(self.enc_ecfp(ecfp))
            avail.append(torch.ones(ecfp.shape[0], device=ecfp.device))
        if self.use_expr:
            embs.append(self.enc_expr(expr))
            avail.append(torch.ones(expr.shape[0], device=expr.device))
        if self.use_text:
            e = self.enc_text(text)
            embs.append(e)
            avail.append(text_mask.float())

        if self.n_views == 1:
            fused = embs[0]
        else:
            cat = torch.cat(embs, dim=1)
            gate_logits = self.gate(cat)
            avail_stack = torch.stack(avail, dim=1)  # (B, n_views), 1 if view present
            gate_logits = gate_logits.masked_fill(avail_stack == 0, -1e9)
            gate_w = torch.softmax(gate_logits, dim=1)
            stacked = torch.stack(embs, dim=1)  # (B, n_views, emb_dim)
            fused = (stacked * gate_w.unsqueeze(-1)).sum(dim=1)
        return self.head(fused).squeeze(-1)


def to_tensor(x, device=DEVICE):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_fusion(ecfp_tr, expr_tr, text_tr, text_mask_tr, y_tr, sample_weights_sched,
                  use_ecfp=True, use_expr=True, use_text=True, n_epochs=60, lr=1e-3, seed=0):
    """sample_weights_sched: either a fixed 1D array (uniform ablation, no-debias) or a
    callable epoch -> 1D weight array (the DebiasedDTA interpolation schedule)."""
    torch.manual_seed(seed)
    model = GatedFusion(use_ecfp, use_expr, use_text).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ecfp_t = to_tensor(ecfp_tr) if use_ecfp else None
    expr_t = to_tensor(expr_tr) if use_expr else None
    text_t = to_tensor(text_tr) if use_text else None
    text_mask_t = to_tensor(text_mask_tr.astype(np.float32)) if use_text else None
    y_t = to_tensor(y_tr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(n_epochs):
        model.train()
        w = sample_weights_sched(epoch) if callable(sample_weights_sched) else sample_weights_sched
        w_t = to_tensor(w)
        opt.zero_grad()
        logits = model(ecfp_t, expr_t, text_t, text_mask_t)
        loss = (bce(logits, y_t) * w_t).mean()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_fusion(model, ecfp=None, expr=None, text=None, text_mask=None, use_ecfp=True, use_expr=True, use_text=True):
    model.eval()
    ecfp_t = to_tensor(ecfp) if use_ecfp else None
    expr_t = to_tensor(expr) if use_expr else None
    text_t = to_tensor(text) if use_text else None
    text_mask_t = to_tensor(text_mask.astype(np.float32)) if use_text else None
    logits = model(ecfp_t, expr_t, text_t, text_mask_t)
    return torch.sigmoid(logits).cpu().numpy()
