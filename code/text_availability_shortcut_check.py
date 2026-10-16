"""Tests a specific reviewer concern: having ANY Hub target annotation at all is itself
correlated with the true label (epilepsy: 22/31 positives vs. 2,264/9,847 negatives have a target
annotation, RESULTS.md section 2), and the gate in fusion_model.py feeds the raw availability bit
in as an input feature (see `avail.append(text_mask.float())`), not just a hard on/off switch. So
the model could gain AUROC from "this compound has a known target at all" rather than from what
the target-function text actually says, a second, undebiased shortcut.

Test: for every TRAIN and TEST fold, keep text_mask exactly as real (true availability is not
touched), but for rows where text IS available, shuffle which real embedding they get, independently
per split and disjointly within train and within test (a train-available row only swaps with
another train-available row, same for test), using a fold-local RNG. This breaks the true
compound-to-content correspondence while preserving the true availability pattern and its
correlation with the label. If performance holds up under this shuffle, the gain is coming from
availability (or some crude marginal property of being annotated), not from the text's content.
If performance collapses toward fusion_struct_expr (no text), the real content is doing the work.

Run only on epilepsy (the class the concern was raised about), text_only and fusion_full, same
5-seed x 5-fold scaffold CV as the headline pipeline, to keep this a bounded, targeted check.

Writes: results/text_availability_shortcut_check.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/code")
from datalib import load, splits, metrics, standardize_view
from fusion_model import train_fusion, predict_fusion

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 60


def shuffle_available_content(text, mask, rng):
    """Permute rows where mask is True among themselves; leave mask-False rows untouched."""
    text_shuf = text.copy()
    avail_idx = np.where(mask)[0]
    if len(avail_idx) > 1:
        perm = rng.permutation(avail_idx)
        text_shuf[avail_idx] = text[perm]
    return text_shuf


def main():
    cmp, ecfp, expr, text, text_mask, y = load("all", "epilepsy")
    print(f"epilepsy n={len(y)} n_pos={y.sum()}, text available for {text_mask.sum()} compounds")

    rows = []
    for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=SEEDS):
        ytr, yte = y[tr], y[te]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        rng = np.random.default_rng(seed * 1000 + fold)

        expr_tr_s, expr_te_s = standardize_view(expr[tr], expr[te])
        text_tr_s, text_te_s = standardize_view(text[tr], text[te], text_mask[tr], text_mask[te])

        # shuffled-content control: same availability pattern, scrambled real content,
        # disjoint within train and within test so no test-row's content leaks from train
        text_tr_shuf = shuffle_available_content(text_tr_s, text_mask[tr], rng)
        text_te_shuf = shuffle_available_content(text_te_s, text_mask[te], rng)

        m_text_real = train_fusion(None, None, text_tr_s, text_mask[tr], ytr, np.ones(len(tr)),
                                    use_ecfp=False, use_expr=False, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        s_text_real = predict_fusion(m_text_real, text=text_te_s, text_mask=text_mask[te],
                                      use_ecfp=False, use_expr=False, use_text=True)

        m_text_shuf = train_fusion(None, None, text_tr_shuf, text_mask[tr], ytr, np.ones(len(tr)),
                                    use_ecfp=False, use_expr=False, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        s_text_shuf = predict_fusion(m_text_shuf, text=text_te_shuf, text_mask=text_mask[te],
                                      use_ecfp=False, use_expr=False, use_text=True)

        m_full_real = train_fusion(ecfp[tr], expr_tr_s, text_tr_s, text_mask[tr], ytr, np.ones(len(tr)),
                                    use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        s_full_real = predict_fusion(m_full_real, ecfp=ecfp[te], expr=expr_te_s, text=text_te_s,
                                      text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        m_full_shuf = train_fusion(ecfp[tr], expr_tr_s, text_tr_shuf, text_mask[tr], ytr, np.ones(len(tr)),
                                    use_ecfp=True, use_expr=True, use_text=True, n_epochs=N_EPOCHS, seed=seed)
        s_full_shuf = predict_fusion(m_full_shuf, ecfp=ecfp[te], expr=expr_te_s, text=text_te_shuf,
                                      text_mask=text_mask[te], use_ecfp=True, use_expr=True, use_text=True)

        m_struct_expr = train_fusion(ecfp[tr], expr_tr_s, None, None, ytr, np.ones(len(tr)),
                                      use_ecfp=True, use_expr=True, use_text=False, n_epochs=N_EPOCHS, seed=seed)
        s_struct_expr = predict_fusion(m_struct_expr, ecfp=ecfp[te], expr=expr_te_s,
                                        use_ecfp=True, use_expr=True, use_text=False)

        for name, s in [("text_only_real", s_text_real), ("text_only_shuffled", s_text_shuf),
                         ("fusion_full_real", s_full_real), ("fusion_full_shuffled", s_full_shuf),
                         ("fusion_struct_expr_notext", s_struct_expr)]:
            rows.append(dict(disease="epilepsy", seed=seed, fold=fold, model=name, **metrics(yte, s)))

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/text_availability_shortcut_check.csv", index=False)
    print(out.groupby("model")["auroc"].agg(["mean", "std"]).round(4))
    print(f"\nWrote {RESULTS}/text_availability_shortcut_check.csv")


if __name__ == "__main__":
    main()
