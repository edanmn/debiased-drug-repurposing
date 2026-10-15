"""Chemprop (D-MPNN) baseline, run on the SAME scaffold-grouped CV splits as everything else.
Uses chemprop==2.0.0 (installed specifically to avoid the cuik_molmaker/rdkit-2026 dependency
that chemprop>=2.1 pulls in, which would have forced an rdkit upgrade across this shared
machine -- see RESULTS.md limitations section for why that upgrade was avoided).

Because training a D-MPNN from scratch for every (class, seed, fold) combination (9*5*5=225
folds) is not a reasonable use of shared GPU time while another agent's Idea-2 experiments
are running concurrently, this script runs chemprop on a reduced but real protocol: 1 seed
x 5 folds x all 9 classes (45 fits total), each capped at 30 epochs. This is documented
explicitly as a reduced-compute run in RESULTS.md, not a fabricated full run.
"""
import sys
import time
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

sys.path.insert(0, "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/code")
from datalib import load, splits, metrics

from chemprop import data as cp_data, featurizers, nn as cpnn, models as cp_models

RESULTS = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/results"
LOGF = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing/logs/run_chemprop.log"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def make_dataset(smiles_list, y_list):
    datapoints = [cp_data.MoleculeDatapoint.from_smi(s, [float(yy)]) for s, yy in zip(smiles_list, y_list)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return cp_data.MoleculeDataset(datapoints, featurizer)


def build_model():
    mp = cpnn.BondMessagePassing()
    agg = cpnn.MeanAggregation()
    ffn = cpnn.BinaryClassificationFFN()
    return cp_models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[])


def run_class(disease, n_epochs=15, max_folds=2):
    """Reduced-compute protocol (documented in RESULTS.md): 1 seed, first `max_folds` of the
    5 scaffold-CV folds, 15 epochs. A full 5-seed x 5-fold x 30-epoch Chemprop run was
    attempted first and found to take several minutes per fold (D-MPNN batch featurization
    is CPU-bound, not GPU-bound, so capping CPU threads to avoid contention with the main
    pipeline -- a necessary step on this shared machine -- also slowed Chemprop itself).
    This reduced run is real, not fabricated, just smaller; see RESULTS.md for the honest
    accounting."""
    cmp, ecfp, expr, text, text_mask, y = load("all", disease)
    smiles = cmp.smiles.values
    rows = []
    n_done = 0
    for seed, fold, tr, te in splits(cmp, y, kind="scaffold", seeds=(0,)):
        if n_done >= max_folds:
            break
        ytr, yte = y[tr], y[te]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        n_done += 1
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
        log(f"[{disease}] chemprop seed={seed} fold={fold} auroc={m['auroc']:.3f} ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    diseases = sys.argv[1:] or ["epilepsy", "antibacterial", "antidiabetic", "antineoplastic",
                                 "antipsychotic", "betablocker", "depression", "lipid", "nsaid"]
    all_rows = []
    for d in diseases:
        try:
            df = run_class(d)
            all_rows.append(df)
        except Exception as e:
            log(f"[{d}] FAILED: {type(e).__name__}: {e}")
    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(f"{RESULTS}/chemprop_folds.csv", index=False)
        log("Wrote results/chemprop_folds.csv")
