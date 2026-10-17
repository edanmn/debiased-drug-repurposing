"""Full-protocol Chemprop run, requested after the reduced run (1 seed, 2 folds, 15 epochs,
run_chemprop.py) was flagged in review as a real but thin baseline. Same model/featurizer as
run_chemprop.py, same 15-epoch budget (already validated there, and full 30-epoch training was
found too slow for a complete sweep), extended to all 5 seeds x all 5 scaffold folds x all 9
classes (up to 225 fits). This can run for hours; it is a coverage increase, not a correctness
fix, and is launched as a long background job rather than blocking anything else.

Writes: results/chemprop_folds_full.csv (separate from chemprop_folds.csv, the original reduced
run, which is left untouched so the documented reduced-protocol result stays intact regardless
of how this one turns out).
"""
import sys
import time
import pandas as pd

sys.path.insert(0, "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/code")
from datalib import load, splits, metrics
from run_chemprop import make_dataset, build_model, log
import torch
from lightning import pytorch as pl
from chemprop import data as cp_data

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
SEEDS = (0, 1, 2, 3, 4)
N_EPOCHS = 15


def run_class_full(disease, n_epochs=N_EPOCHS, seeds=SEEDS):
    cmp, ecfp, expr, text, text_mask, y = load("all", disease)
    smiles = cmp.smiles.values
    rows = []
    for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=seeds):
        ytr, yte = y[tr], y[te]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        t0 = time.time()
        train_ds = make_dataset(smiles[tr], ytr)
        test_ds = make_dataset(smiles[te], yte)
        train_loader = cp_data.build_dataloader(train_ds, batch_size=256, shuffle=True)
        test_loader = cp_data.build_dataloader(test_ds, batch_size=512, shuffle=False)
        model = build_model()
        trainer = pl.Trainer(
            max_epochs=n_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1, enable_progress_bar=False, enable_checkpointing=False, logger=False,
        )
        trainer.fit(model, train_loader)
        preds = trainer.predict(model, test_loader)
        scores = torch.cat([p.flatten() for p in preds]).numpy()
        m = metrics(yte, scores)
        rows.append(dict(disease=disease, seed=seed, fold=fold, model="chemprop_dmpnn", **m))
        log(f"[FULL][{disease}] chemprop seed={seed} fold={fold} auroc={m['auroc']:.3f} ({time.time()-t0:.1f}s)")
        # Write incrementally so partial progress survives if this is interrupted.
        pd.DataFrame(rows).to_csv(f"{RESULTS}/chemprop_folds_full.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    diseases = sys.argv[1:] or ["epilepsy", "antibacterial", "antidiabetic", "antineoplastic",
                                 "antipsychotic", "betablocker", "depression", "lipid", "nsaid"]
    all_rows = []
    for d in diseases:
        try:
            df = run_class_full(d)
            all_rows.append(df)
        except Exception as e:
            log(f"[FULL][{d}] FAILED: {type(e).__name__}: {e}")
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(f"{RESULTS}/chemprop_folds_full.csv", index=False)
        log("[FULL] Wrote results/chemprop_folds_full.csv")
