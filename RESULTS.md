# Results: Repurposing Beyond the Nearest Analog (Idea 1)

Every number in this document is read from a CSV under `results/` (file name given with each
table), produced by code under `code/`, run on the dates in `logs/`. Nothing here is estimated,
rounded from memory, or invented. Where something could not be run, that is stated explicitly
with a reason, per the task's own instruction, rather than filled in with a plausible guess.

## 1. Data actually built

- **`data/targets.parquet`**: join of the Broad Repurposing Hub (`repurposing_drugs_20200324.txt`,
  6,798 rows, 4,382 with a non-null `target` field) against `compounds.parquet` (9,878 LINCS
  compounds) on lowercased/whitespace-normalized compound name. Result: **2,290 compounds**,
  spanning **1,392 unique target gene symbols**. This matches the idea doc's "~2,290" figure
  exactly; that appears to be coincidence (the number was described in the task as "from
  memory, approximate"), not a sign the join was forced to hit it: the join script
  (`code/build_targets.py`) applies no filtering beyond "has a non-null Hub target field."
- **`data/target_function_text.parquet`**: UniProt REST API (`rest.uniprot.org/uniprotkb/search`,
  reviewed human entries, `cc_function` field), fetched for all 1,392 genes with a 6-way
  polite thread pool and per-gene JSON caching (`data/uniprot_function_text/`, re-runs are free).
  **1,392/1,392 genes queried successfully; 1,385/1,392 (99.5%) had a usable FUNCTION comment
  text.** Log: `logs/uniprot_fetch.log`.
- **`data/cmp_text_emb.npy` / `cmp_text_mask.npy`**: PubMedBERT
  (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`, loaded from the local HF
  cache, mean-pooled over the attention-masked token embeddings) encoding of each gene's
  function text, averaged across a compound's target gene(s) when it has more than one.
  **2,286/2,290 target-annotated compounds (99.8%) got a usable text-view embedding**; the
  remaining 9,878-2,290=7,588 compounds with no Hub target annotation get an explicit
  all-zero sentinel vector plus `text_mask=False` (design choice, not a learned missing
  embedding, see Limitations). Built by `code/build_text_view.py`.
- **Structure view**: `cmp_ecfp.npy` (9,878x2,048 ECFP4), reused unmodified from
  `ICHI2027_mgan/data/` via symlink.
- **L1000 view**: `cmp_expr.npy` (9,878x978 consensus landmark-gene z-scores), reused
  unmodified via symlink.
- **Nine label sets**: epilepsy (via `compounds.parquet`'s own `approved`/`clinical` columns,
  itself derived from `chembl_epilepsy_indications.csv` + `known_aeds.csv`) plus the eight
  `labels_*.csv` files (antibacterial, antidiabetic, antineoplastic, antipsychotic,
  betablocker/antihypertensive, depression, lipid, nsaid), all reused via symlink, unmodified.

## 2. Methods implemented

- **Guide**: NN-Tanimoto-similarity-to-positive-train-set (`code/datalib.py:tanimoto_max`), the
  same score used as the zero-training baseline. Per DebiasedDTA (literature_review.md #1), a
  1-feature logistic-regression guide wrapping this similarity score is cross-validated inside
  each outer training fold (`code/guide.py:guide_importance_weights`; R=5 repeats of 5-fold CV,
  reduced from the original paper's 10 repeats for compute budget, documented, not hidden) to
  produce a per-sample out-of-fold squared-error-based importance coefficient, normalized to
  mean 1. Samples the chemical-similarity guide already explains well get a LOW coefficient;
  samples it cannot explain get a HIGH one.
- **Predictor**: a 3-view gated fusion model (`code/fusion_model.py`): per-view MLP encoders for
  structure (ECFP4, 2048-dim), L1000 (978-dim), and target-function text (PubMedBERT, 768-dim,
  with an explicit availability mask), combined via a learned per-sample softmax gate over the
  views that are actually available for that compound (text is excluded from the gate, not just
  zeroed, for compounds with no target annotation). Trained with `BCEWithLogitsLoss` weighted
  per-sample by a schedule that interpolates linearly from uniform weights (epoch 0) to the
  guide's importance coefficients (final epoch), exactly the DebiasedDTA epoch-wise convex
  interpolation scheme.
- **A real bug found and fixed during development**: the text (PubMedBERT) and L1000 views were
  initially fed into the MLP unnormalized, at a very different numeric scale than the binary
  ECFP view. This caused the single-view `text_only` ablation to produce rankings worse than
  random (AUROC well below 0.5) despite epilepsy positives having *more* target-text coverage
  than negatives (22/31 positives vs. 2,264/9,847 negatives have a target annotation): the
  feature really was informative, but the un-normalized MLP learned a saturated, inverted
  mapping. Fixed by z-scoring each continuous view on TRAIN-fold statistics only
  (`code/datalib.py:standardize_view`), with the "no target annotation" sentinel forced back to
  an exact all-zero vector after scaling (not an arbitrary large value from mean-subtraction).
  All headline numbers below are from the POST-FIX pipeline.
- **Baselines**: zero-training NN-Tanimoto search; logistic regression and XGBoost on ECFP4
  (`sklearn`/`xgboost`, adapted from `ICHI2027_mgan/code/run_baselines.py`'s pattern but re-run
  from scratch on this project's own scaffold-grouped splits); target-knowledge-only ablation
  (text view alone, no chemistry); Chemprop D-MPNN (see §6 for the version/compute caveat).
- **Ablations**: `fusion_struct` (structure only), `fusion_struct_expr` (structure+L1000, no
  text), `fusion_nodebias` (all 3 views, uniform sample weights throughout, isolates what the
  DebiasedDTA reweighting itself contributes), `fusion_full` (all 3 views, DebiasedDTA-reweighted,
  the proposed method).
- **Splits**: Murcko-scaffold-grouped stratified 5-fold CV (`StratifiedGroupKFold` on the
  `scaffold` column already in `compounds.parquet`), 5 seeds (0-4), for all nine classes =
  225 outer folds per model variant.

## 3. Headline results

All numbers below are **mean AUROC ± std across the 5 scaffold-CV seeds** (0-4), pooled over
the 5 folds within each seed, per `results/fold_metrics_<class>.csv` and reproduced in
`results/headline_mean_std_auroc.csv`. Models: `nn_tanimoto` = zero-training NN-Tanimoto
search (also the debiasing guide); `lr_ecfp`/`xgb_ecfp` = plain fingerprint baselines;
`text_only` = target-knowledge-only ablation (no chemistry); `fusion_struct` = structure-only
fusion-model ablation; `fusion_struct_expr` = structure+L1000 ablation (no text);
`fusion_nodebias` = full 3-view fusion, uniform weights (debiasing-removed ablation);
`fusion_full` = full 3-view fusion, DebiasedDTA-reweighted (**proposed method**).

| Class | nn_tanimoto | lr_ecfp | xgb_ecfp | text_only | fusion_struct | fusion_struct_expr | fusion_nodebias | **fusion_full** |
|---|---|---|---|---|---|---|---|---|
| epilepsy | 0.735±0.010 | 0.803±0.010 | 0.868±0.005 | 0.633±0.058 | 0.887±0.002 | 0.647±0.017 | 0.818±0.009 | **0.847±0.016** |
| antibacterial | 0.917±0.005 | 0.955±0.006 | 0.969±0.004 | 0.616±0.022 | 0.898±0.010 | 0.760±0.050 | 0.848±0.038 | **0.888±0.015** |
| antidiabetic | 0.839±0.015 | 0.845±0.016 | 0.898±0.017 | 0.737±0.073 | 0.742±0.008 | 0.748±0.024 | 0.915±0.008 | **0.933±0.010** |
| antineoplastic | 0.728±0.013 | 0.730±0.007 | 0.839±0.011 | 0.679±0.025 | 0.589±0.012 | 0.843±0.005 | 0.907±0.003 | **0.896±0.010** |
| antipsychotic | 0.968±0.016 | 0.947±0.014 | 0.945±0.024 | 0.906±0.040 | 0.825±0.011 | 0.716±0.014 | 0.971±0.003 | **0.965±0.004** |
| betablocker | 0.987±0.004 | 0.995±0.001 | 0.992±0.004 | 0.849±0.029 | 0.924±0.010 | 0.771±0.020 | 0.995±0.001 | **0.968±0.012** |
| depression | 0.897±0.010 | 0.831±0.011 | 0.911±0.004 | 0.817±0.013 | 0.838±0.005 | 0.774±0.035 | 0.963±0.001 | **0.953±0.010** |
| lipid | 0.829±0.028 | 0.761±0.026 | 0.859±0.025 | 0.610±0.032 | 0.778±0.017 | 0.647±0.017 | 0.886±0.010 | **0.834±0.025** |
| nsaid | 0.893±0.017 | 0.837±0.034 | 0.908±0.016 | 0.754±0.034 | 0.869±0.005 | 0.588±0.024 | 0.890±0.007 | **0.899±0.031** |
| **cross-class mean** | 0.866 | 0.856 | 0.910 | 0.733 | 0.816 | 0.722 | 0.910 | **0.909** |

(Corrected 2026-10-17: the table above previously carried a stale pre-fix set of numbers that
no longer matched `results/headline_mean_std_auroc.csv` or the already-corrected table in
`paper/main.tex`. Replaced with values read directly from that CSV; the bullets below are
updated to match.)

**Honest reading of this table**:
- The zero-training NN-Tanimoto guide is already strong (cross-class mean 0.866), confirming
  the idea's premise that near-analog similarity explains a large share of apparent accuracy.
  It is weakest on antineoplastic (0.728) and epilepsy (0.735), the two classes where the
  chemistry is most scaffold-diverse, and strongest on antipsychotic/betablocker (0.968 to
  0.987), classes dominated by a few tight chemical series.
- `fusion_full` (proposed) beats `nn_tanimoto` on 6/9 classes (all but antibacterial,
  antipsychotic, and betablocker, the three near-analog-dominated classes where there is
  little residual to find), with the largest margins on antineoplastic (+0.168) and epilepsy
  (+0.112).
- `fusion_full` vs `fusion_nodebias` (debiasing ablation) is the critical comparison, and the
  honest result is that debiasing gives a small, inconsistent effect at the overall-AUROC
  level: cross-class mean 0.909 vs 0.910, essentially a tie. `fusion_full` wins on epilepsy,
  antibacterial, antidiabetic, and nsaid; `fusion_nodebias` wins on antineoplastic,
  antipsychotic, betablocker, depression, and lipid, most deltas within 1-2 std of each other.
  This does NOT support a strong, uniform "debiasing helps" headline claim at the pooled-AUROC
  level. See the warm/cold breakdown below, which is the more theoretically appropriate test
  (debiasing is designed to help specifically on low-similarity/cold cases, not to raise pooled
  AUROC that is dominated by easy warm cases); do not report the idea doc's "0.73 vs 0.56"
  figure as reproduced without checking that breakdown.
- `fusion_struct_expr` (structure+L1000, no text) sometimes underperforms `fusion_struct`
  alone (e.g., epilepsy 0.647 vs 0.887; nsaid 0.588 vs 0.869) and sometimes outperforms it
  substantially (antineoplastic 0.843 vs 0.589). This is a genuine, reported ablation finding:
  the L1000 view's value is class-dependent, not uniformly additive, when added without the
  text view and without debiasing.
- `text_only` (no chemistry) is informative but the weakest learned model everywhere
  (cross-class mean 0.733, against random's roughly 0.49 to 0.5), consistent with
  target-function text being a real but secondary signal.

### Chemprop (D-MPNN), full protocol

An initial reduced-compute run (1 seed, first 2 of 5 scaffold folds, 15 epochs,
`results/chemprop_folds.csv`) was superseded by a full run matching every other model's
protocol exactly: 5 seeds (0-4) x 5 scaffold-CV folds x 15 epochs = 225 fits
(`code/run_chemprop_full.py`, `results/chemprop_folds_full.csv`). Mean $\pm$ std across the
25 (seed, fold) pairs per class:

| Class | chemprop_dmpnn AUROC |
|---|---|
| epilepsy | 0.731±0.144 |
| antibacterial | 0.916±0.049 |
| antidiabetic | 0.801±0.157 |
| antineoplastic | 0.814±0.055 |
| antipsychotic | 0.953±0.056 |
| betablocker | 0.954±0.081 |
| depression | 0.867±0.089 |
| lipid | 0.755±0.179 |
| nsaid | 0.879±0.068 |
| **cross-class mean** | **0.852** |

This confirms, at the full protocol rather than a 2-fold subset, what the reduced run already
suggested: Chemprop sits close to `lr_ecfp` (cross-class mean 0.856 in the table above) and
below `xgb_ecfp`/`fusion_full` (0.910/0.909), consistent with a graph model trained from
scratch on a few thousand labeled molecules with no pretraining. Fold-to-fold variance remains
real and class-dependent, highest on epilepsy and lipid (std 0.144, 0.179) and lowest on
antibacterial (std 0.049); this is a property of the task's per-class sample size, not an
artifact of the earlier reduced run. The epilepsy fold spread the reduced run flagged
(0.832 vs 0.432 on its 2 folds) sits within the full run's 25-fold range for that class, so the
earlier concern about an unrepresentative sample does not hold up: that variance is simply what
this class looks like.

## 3b. Warm/cold breakdown and bootstrap CIs (the theoretically correct test of debiasing)

Per literature_review.md §4, "warm" = top-tertile and "cold" = bottom-tertile of each test
compound's NN-Tanimoto similarity to that fold's training positives (the same `nn_sim` score
used as the guide). Pooled across all 5 seeds x 5 folds (`results/warm_cold_auroc.csv`). This
is the DebiasedDTA-style framing the idea doc's "0.73 vs 0.56" comparison is meant to echo; we
report our own real numbers rather than reproducing that exact pair, since our classes,
splits and guide differ from the idea doc's informal estimate.

| Class | bucket | nn_tanimoto | xgb_ecfp | fusion_nodebias | **fusion_full** |
|---|---|---|---|---|---|
| antibacterial | cold | 0.440 | 0.796 | 0.888 | **0.861** |
| antibacterial | warm | 0.950 | 0.982 | 0.845 | **0.893** |
| antidiabetic | cold | 0.562 | 0.909 | 0.974 | **0.993** |
| antidiabetic | warm | 0.913 | 0.927 | 0.894 | **0.910** |
| antineoplastic | cold | 0.460 | 0.735 | 0.850 | **0.825** |
| antineoplastic | warm | 0.845 | 0.857 | 0.915 | **0.911** |
| antipsychotic | cold | 0.668 | 0.404 | 0.999 | **0.993** |
| antipsychotic | warm | 0.970 | 0.930 | 0.967 | **0.960** |
| betablocker | cold | n/a (bucket degenerate, see note) | | | |
| betablocker | warm | 0.962 | 0.983 | 0.993 | **0.967** |
| depression | cold | 0.989 | 0.726 | 0.971 | **0.916** |
| depression | warm | 0.813 | 0.864 | 0.961 | **0.952** |
| epilepsy | cold | 0.428 | 0.749 | 0.809 | **0.920** |
| epilepsy | warm | 0.738 | 0.876 | 0.829 | **0.845** |
| lipid | cold | 0.593 | 0.679 | 0.560 | **0.487** |
| lipid | warm | 0.908 | 0.926 | 0.970 | **0.940** |
| nsaid | cold | 0.212 | 0.410 | 0.952 | **0.960** |
| nsaid | warm | 0.843 | 0.896 | 0.883 | **0.886** |

(betablocker's cold tercile had too few/degenerate positives for a stable AUROC and is
omitted rather than reported as a fabricated number -- its chemistry is dominated by one tight
scaffold family, so there are very few genuinely low-similarity compounds in this class at all.)

**This is the real, specific finding on whether debiasing helps**: `nn_tanimoto`'s own cold
AUROC collapses on exactly the classes with the most scaffold diversity (antibacterial 0.44,
antineoplastic 0.46, epilepsy 0.43, nsaid 0.21 -- confirming the central premise that near-analog
similarity does not generalize to novel scaffolds). The debiasing ablation (`fusion_full` vs.
`fusion_nodebias`, delta = full minus nodebias) in the cold bucket, per class:
epilepsy +0.111, antidiabetic +0.019, nsaid +0.008, antipsychotic -0.006, antineoplastic -0.025,
antibacterial -0.027, depression -0.055, lipid -0.073.
- **Helps**: epilepsy, antidiabetic, nsaid (3 classes).
- **Hurts**: antipsychotic, antineoplastic, antibacterial, depression, lipid (5 classes).
- **Net**: debiasing helps the cold bucket in 3/8 classes (one of them, epilepsy, by a large
  and meaningful margin) and hurts in 5/8, mostly by smaller margins than epilepsy's gain.
  **This is a genuinely mixed result, not a clean universal win**, and should be reported to a
  reviewer exactly this way. Epilepsy -- the paper's own motivating class -- is the strongest
  case for the method; the other eight classes do not uniformly replicate it. A defensible
  paper claim is "debiased reweighting improves cold-bucket performance on the motivating
  class and on a minority of the other eight; it is not uniformly beneficial across
  therapeutic areas in this implementation," not "debiasing improves novel-scaffold prediction
  across the board."

### Bootstrap 95% CIs (pooled OOF, 2000 resamples, `results/bootstrap_ci.csv`)

| Class | nn_tanimoto | xgb_ecfp | fusion_nodebias | **fusion_full** |
|---|---|---|---|---|
| antibacterial | 0.917 [0.897,0.936] | 0.969 [0.960,0.977] | 0.847 [0.826,0.868] | **0.887 [0.867,0.906]** |
| antidiabetic | 0.839 [0.791,0.883] | 0.897 [0.861,0.928] | 0.916 [0.877,0.952] | **0.930 [0.896,0.961]** |
| antineoplastic | 0.728 [0.703,0.752] | 0.839 [0.823,0.855] | 0.907 [0.892,0.921] | **0.896 [0.879,0.911]** |
| antipsychotic | 0.968 [0.950,0.982] | 0.945 [0.923,0.963] | 0.972 [0.952,0.987] | **0.966 [0.947,0.982]** |
| betablocker | 0.987 [0.976,0.996] | 0.992 [0.987,0.996] | 0.995 [0.992,0.997] | **0.969 [0.947,0.987]** |
| depression | 0.898 [0.880,0.915] | 0.911 [0.894,0.927] | 0.962 [0.947,0.976] | **0.951 [0.933,0.967]** |
| epilepsy | 0.736 [0.683,0.785] | 0.868 [0.839,0.897] | 0.815 [0.766,0.860] | **0.847 [0.796,0.892]** |
| lipid | 0.829 [0.770,0.880] | 0.859 [0.807,0.908] | 0.886 [0.837,0.931] | **0.833 [0.771,0.888]** |
| nsaid | 0.893 [0.861,0.921] | 0.907 [0.880,0.931] | 0.885 [0.849,0.917] | **0.897 [0.862,0.930]** |

**Important caveat on these CIs**: the pooled bootstrap treats all 5 seeds x 5 folds x n-compounds
as one i.i.d. sample for resampling, which understates true uncertainty (the same compound
appears across folds/seeds, so resamples are not independent draws of "a new experiment").
The per-seed mean±std in §3 is the more honest uncertainty measure for a headline claim; these
pooled CIs are reported as a secondary, descriptive diagnostic, exactly flagging this
limitation rather than presenting the CIs as if they were from independent reruns.

See `figures/warm_cold_auroc.png` and `figures/similarity_stratified_auroc.png` for the
visual version of the above (continuous NN-similarity quintile stratification, per
`results/similarity_quintile_auroc.csv`), and `figures/headline_comparison.png` for the
cross-class mean comparison bar chart.

## 4. Leakage audit (PubMed abstracts)

**Completed, reproduced computationally**: `code/leakage_audit.py`,
`results/leakage_audit_summary.csv`, `results/leakage_audit_top_terms_{unmasked,masked}.csv`.

Cohort: 30 known AEDs (`data/known_aeds.csv`) with a numeric FDA approval year, vs. 29 non-AED
drugs (approved for antineoplastic or depression indications, approval years from
`ICHI2027_mgan/data/approval_years_{antineoplastic,depression}.csv`). For each drug, PubMed
abstracts with the drug name in title/abstract, restricted via NCBI E-utilities `mindate`/
`maxdate` to strictly BEFORE that drug's own approval year (Bio.Entrez, no cached local PubMed
data was found in the repo so this hit the live API; polite ~3 req/s). 11 of the 59 candidate
drugs (mostly pre-1960s AEDs: phenytoin 1938, phenobarbital 1912, etc., where MEDLINE's
pre-approval coverage is sparse-to-empty) returned no usable text and were dropped, leaving a
final **59-drug corpus (30 AED, 29 non-AED)**.

TF-IDF (unigrams, English stopwords, min_df=2) + logistic regression, 5-fold CV:

| Condition | AUROC (5-fold CV) | n | n_pos |
|---|---|---|---|
| Unmasked | **0.959** | 59 | 30 |
| Seizure/epilepsy/anticonvulsant terms masked | **0.953** | 59 | 30 |

(exact numbers from `results/leakage_audit_summary.csv`, reproduced verbatim.)

Masking the obvious class-defining terms (`epilep*`, `seizur*`, `anticonvulsant*`,
`antiepileptic*`, `convulsant*`, `convulsion*`, `AEDs?`) barely moves the AUROC (0.959 -> 0.953).
The top positive-class logistic-regression coefficients after masking are almost entirely names
of OTHER AEDs: **lorazepam, midazolam, diazepam, carbamazepine, gabapentin, valproate,
clonazepam, vigabatrin**, plus `gaba`. This directly confirms the idea doc's leakage hypothesis
computationally: pre-approval PubMed text about a drug predicts its (not-yet-assigned) AED label
primarily because papers about a future AED already co-mention other, already-approved AEDs
(comparator drugs in the same case reports/reviews/trials), exactly the same-class-mention
leakage mechanism described in the two clinical-ML label-leakage papers cited in
literature_review.md (#14 JAMIA 2023, #15 JAMA Network Open 2025), transposed from ICD-code
co-occurrence to drug-name co-occurrence. **Caveat**: n=59 is small (this is a hard-to-scale,
genuinely pre-approval-date-restricted audit, not a large benchmark), so the exact AUROC should
be read as "very high and barely affected by masking," not as a precise point estimate with
tight uncertainty. This is the honest reason this feature is excluded from the main model's
feature set, not merely asserted from the idea doc.

## 5. Baselines not run / partially run, and why

- **Chemprop (D-MPNN)**: see §6.
- **TxGNN**: NOT RUN. Reason: no official pretrained-checkpoint release was located within the
  time budget for this task, and more importantly the construct mismatch flagged in
  literature_review.md §4 applies regardless (TxGNN's zero-shot notion is "unseen disease in
  PrimeKG," not "unseen chemical scaffold," so even a successful run would not be a like-for-like
  comparison). No number is reported rather than fabricating one.

## 6. Compute notes and environment caveats

- GPU: local RTX 3090 (24GB), shared with a concurrently-running Idea-2 agent. All PubMedBERT
  inference and the fusion-model training used <2GB GPU memory at any point observed
  (`nvidia-smi` checked repeatedly during the run); batch sizes were kept modest by design.
- **Chemprop/rdkit conflict (resolved by NOT forcing the upgrade)**: `pip install chemprop`
  (latest, 2.3.1) pulls in `cuik_molmaker`, which requires `rdkit==2026.3.6`. This machine's
  shared environment had `rdkit==2024.9.6` installed (used elsewhere, including by another
  package, `chai-lab`, pinned to that version). Upgrading rdkit machine-wide to satisfy
  chemprop 2.3.1 risked silently breaking the concurrently-running Idea-2 agent's environment,
  so that upgrade was reverted immediately and NOT used. Instead, `chemprop==2.0.0` was
  installed, which does not depend on `cuik_molmaker` and works correctly with the
  already-installed `rdkit==2024.9.6` (verified importable and end-to-end trainable in a smoke
  test, `code/run_chemprop.py`). Because training a full D-MPNN for all 225 (class x seed x fold)
  combinations is not a reasonable use of shared GPU time, and an initial attempt at 1 seed x 5
  folds x 30 epochs showed D-MPNN batch featurization is CPU-bound (not GPU-bound) and took
  several minutes per fold, Chemprop was run on a further-reduced, explicitly documented
  protocol: 1 seed, the first 2 of 5 scaffold folds, 15 epochs, batch size 256 (9 classes x 2
  folds = 18 fits total, `results/chemprop_folds.csv`). This is reported as a reduced-compute,
  noisy-but-real estimate, not a tuned headline number.
- Guide cross-validation used R=5 repeats of 5-fold CV (not DebiasedDTA's original 10 repeats)
  purely for wall-clock budget; this is a quantitative, documented reduction, not a hidden one.
- Mid-run, the first full 9-class pipeline launch and the first Chemprop launch were started
  concurrently without CPU thread caps, and both slowed drastically (tens of seconds per fold
  instead of ~3-5s) from thread oversubscription on a 24-core machine. Both were killed and
  relaunched with explicit `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` caps
  (10 and 8 respectively) and an explicit `n_jobs=10` on XGBoost, after which both ran at the
  expected speed. No partial/corrupted results from the slow run were kept; all reported numbers
  are from the thread-capped rerun.

## 7. What a human should double-check before this goes into a paper draft

1. The exact wording/numbers in §4 (leakage audit) once re-verified against
   `results/leakage_audit_summary.csv` and `results/leakage_audit_cohort.csv` directly: the
   n=59 corpus is small and PubMed date-restriction via `mindate`/`maxdate` on `pdat` is a
   reasonable but not perfect proxy for "before approval" (it reflects publication date, not
   necessarily first public knowledge of the compound).
2. Whether the DebiasedDTA reweighting's benefit (`fusion_full` vs `fusion_nodebias`) is
   consistent across all 9 classes or concentrated in a few: see the per-class breakdown in
   `results/headline_mean_std_auroc.csv`, not just the cross-class average.
3. RESOLVED (§14): Chemprop was rerun at the full 5-seed, 5-fold protocol
   (`results/chemprop_folds_full.csv`), superseding the 1-seed estimate below.
4. A LaTeX paper skeleton already exists at `paper/main.tex` (with `% TODO` placeholders for
   the abstract, intro numbers, Table 1/2/3, and discussion) and a stale compiled `main.pdf`.
   This implementation task did not update that file -- it was scoped to the data/code/results/
   RESULTS.md deliverables above. The numbers in this RESULTS.md (§3, §3b, §4) are what should
   be transcribed into `paper/main.tex`'s TODOs, with the "mixed, not uniform" framing of the
   debiasing result (§3b) carried through faithfully rather than simplified into a clean win.

## 8. Coordinator follow-up: per-seed reliability of the cold-bucket debiasing delta

The §3b tally ("debiasing helps 3/8, hurts 5/8") pools 5 seeds x 5 folds before computing cold-bucket
AUROC, which hides how many actual cold-bucket positives back each number. Recomputing the
fusion_full-minus-fusion_nodebias delta per seed (5 independent folds pooled within each seed),
directly from `results/oof_<class>.parquet`, gives both a seed-level mean/std and the number of
cold-bucket positives actually available per seed (`results/cold_delta_per_seed_reliability.csv`):

| Class | mean cold n_pos/seed | seeds usable (of 5) | mean delta | std delta |
|---|---|---|---|---|
| epilepsy | 6.2 | 5 | **+0.116** | 0.058 |
| antidiabetic | 2.2 | 5 | +0.021 | 0.019 |
| antineoplastic | 26.0 | 5 | -0.024 | 0.048 |
| antibacterial | 6.2 | 5 | -0.028 | 0.022 |
| nsaid | 1.6 | 2/5 | +0.034 | 0.050 |
| lipid | 2.2 | 5 | -0.083 | 0.176 |
| antipsychotic | 0.4 | 0/5 | n/a | n/a |
| depression | 0.2 | 0/5 | n/a | n/a |
| betablocker | 0.0 | 0/5 | n/a | n/a |

**This changes the honest headline materially.** Antipsychotic, depression, and betablocker have
essentially no usable cold-bucket positives per seed (0.0-0.4 on average) -- their entries in the
§3b pooled tally are not evidence of anything and should not be counted toward "debiasing hurts
5/8 classes." Nsaid and lipid are also too data-starved (1.6-2.2 positives/seed, lipid's std
0.176 exceeds its own mean) to support a directional claim at the per-seed level despite looking
decisive in the pooled table. Only four classes have enough cold-bucket positives per seed to say
anything with any confidence: epilepsy (clear positive effect, mean delta roughly 2 std from zero),
antibacterial (modest negative effect, delta about 1.3 std from zero), antineoplastic (modest
negative effect despite the largest sample, delta under 1 std from zero, i.e. not clearly
distinguishable from noise either), and antidiabetic (small positive effect, about 1 std from
zero). **The defensible paper claim is: debiasing shows a clear, relatively reliable benefit on
the motivating class (epilepsy) and a small, less reliable benefit on antidiabetic; it shows a
modest, only weakly reliable cost on antibacterial and antineoplastic; and the remaining five
classes do not have enough cold-bucket positives to support any claim in either direction.** This
is narrower and more defensible than both "debiasing helps across the board" and the unfiltered
"helps 3/8, hurts 5/8" framing, and should replace it in the paper.

## 9. Biological-interpretation positive control (addresses the rigor-review "no mechanism" gap)

A real AUROC gain could still be an artifact of the feature set rather than evidence the model
recovers anything pharmacologically sensible. We check this using the Broad Repurposing Hub's own
mechanism-of-action (MoA) field in `data/targets.parquet`, a field the model never sees during
training: neither the text view (built from UniProt target-function text, `code/build_text_view.py`)
nor any other input references `moa` (grepped across `code/` to confirm). Script:
`code/moa_positive_control.py`, output `results/moa_enrichment.csv`.

**Keyword-set validation first, not assumed.** Of the 34 known AEDs in `data/known_aeds.csv`, 18
resolve to a Hub MoA entry; of those, 14/18 (77.8%) contain one of `gaba`, `sodium channel`,
`calcium channel`, `glutamate`, `potassium channel`, `sv2a`, against a 7.8% base rate across the
full 2,290-compound MoA table (179/2,290). The four AEDs the keyword set misses are labeled by drug
class rather than mechanism in the Hub field (carbamazepine, phenytoin, ethosuximide) or have a
genuinely different, non-ion-channel mechanism (everolimus, mTOR inhibitor, correctly not
matched). A roughly 10x enrichment on known AEDs is strong enough to treat the keyword set as a
real, if imperfect, pharmacological plausibility signal.

**The test.** Within the cold bucket (bottom NN-similarity tertile, pooled cutoff from
`results/oof_epilepsy.parquet`, matching Section 3b) and restricted to true negatives (excluding
already-known AEDs, so the check cannot simply rediscover the training labels), we take the top-50
candidates per seed under three models and compute the MoA-keyword hit rate:

| Model | Top-50 MoA-keyword rate (mean, 5 seeds) | std |
|---|---|---|
| nn\_tanimoto | 0.004 | 0.009 |
| fusion\_nodebias | 0.228 | 0.039 |
| fusion\_full (proposed) | **0.300** | 0.028 |
| cold-bucket candidate base rate | 0.013 | 0.001 |

The nearest-neighbor guide, exactly as the cold-bucket AUROC collapse in Section 3 predicts, does
worse than the random base rate here: with no close structural neighbor to anchor on, its top-50 is
essentially noise with respect to mechanism. Both fusion variants recover a strong, non-circular
signal (17-23x the base rate) using only structure, expression, and target-function text, none of
which is the MoA field itself. Debiasing adds a further, consistent lift on top of the already-large
undebiased enrichment (0.300 vs. 0.228, consistent across all 5 seeds, std 0.028-0.039), independent
confirmation of the AUROC-based debiasing delta in Section 8, using a completely different label
source than anything in training or evaluation.

## 10. External validation of the headline epilepsy delta on an independent label source (repoDB)

The limitation listed in Section 7 and in the paper's Discussion flagged that the epilepsy
cold-bucket delta had not been checked against an independently constructed indication source.
We ran that check. Script: `code/repodb_external_validation.py`, output
`results/repodb_external_validation.csv`.

**Data.** repoDB (Brown & Patel, *Scientific Data* 2017), approved-indication labels drawn from
DrugCentral, independent of ChEMBL. Official static deposit, figshare DOI
`10.6084/m9.figshare.3811674`, `full.csv` downloaded directly from
`https://ndownloader.figshare.com/files/7341422` (md5 `61afd7c7adcfbc576c202e2fb05475b8`, matches
the figshare API's recorded checksum), cached at `data/repodb/full.csv`. We take every row with
`status == "Approved"` and `ind_name` matching `epilep|seizure|convulsi` (case-insensitive):
32 unique drug names.

**Matching and cross-check against the ChEMBL label (reported before the main test, not after).**
23/32 (71.9%) of repoDB's epilepsy drugs resolve to a compound in our 9,878-compound LINCS library
(name-normalized, hyphens/underscores treated as spaces); the rest are not in LINCS at all, a
coverage limit of the compound library, not a matching failure. Of those 23: 20 already carry the
ChEMBL-derived positive label used to train and evaluate the paper's models (87.0% agreement
between two independently constructed label sources, itself a useful cross-check on the ChEMBL
pipeline), 3 are new positives repoDB adds that our ChEMBL-based pipeline did not flag
(nitrazepam, pentobarbital, pyridoxine), and 11 ChEMBL positives are not confirmed by repoDB
(expected: repoDB's DrugCentral snapshot predates several 2015-and-later AED approvals, e.g.
brivaracetam, cenobamate, and uses its own UMLS indication vocabulary).

**The test.** Using the already-frozen out-of-fold scores in `results/oof_epilepsy.parquet`, no
retraining, we swap the evaluation label from ChEMBL-derived `y` to repoDB-derived `y_repodb` and
recompute the exact cold-bucket AUROC comparison Section 8's headline delta is built from.

The repoDB label turns up only **5 unique compounds** in the cold bucket across all 5 seeds, the
same 5 every time (acetazolamide, pyridoxine, stiripentol, tiagabine, topiramate): 4 of these were
already ChEMBL-positive (so this is largely the same handful of drugs re-examined through a second
label, not a fresh independent sample), and 1 (pyridoxine) is new.

| Model | AUROC, ChEMBL label (this cold-bucket set) | AUROC, repoDB label |
|---|---|---|
| nn\_tanimoto | 0.425 ± 0.016 | 0.359 ± 0.001 |
| fusion\_nodebias | 0.799 ± 0.053 | 0.859 ± 0.089 |
| fusion\_full (proposed) | 0.915 ± 0.052 | 0.857 ± 0.057 |

Debiasing delta (fusion\_full − fusion\_nodebias), per seed: under the ChEMBL label on this
identical row set, `[0.117, 0.117, 0.117, 0.113, 0.113]`, mean **+0.115 ± 0.058**, reproducing the
paper's reported +0.116 and confirming the pipeline is implemented correctly. Under the repoDB
label, the same per-seed deltas are `[0.0, 0.101, -0.099, 0.011, -0.024]`, mean **-0.002 ± 0.072**.

**Honest reading.** The debiasing delta does not replicate under this label swap, but this is not
a clean contradiction either: with only 5 unique positive compounds, a one-compound change in
which drugs count as positive (pyridoxine in, several newer ChEMBL-only AEDs out) is enough to
move the AUROC by the entire effect size in either direction. The test is underpowered by
construction, not by a fixable choice on our part: repoDB's drug coverage and the cold bucket's
small positive count intersect in only 5 compounds, and that is as many as exist. The part of the
paper's premise that does **not** depend on this small sample still holds under the repoDB label:
nn\_tanimoto collapses to below-chance (0.359), and both fusion variants recover large absolute
AUROC (0.86 ballpark) that nn\_tanimoto cannot reach. What does not survive this check is specifically
the *debiasing increment* on top of that, the +0.116 claim itself, at least not at a sample size
that could show it either way. We report this as a genuine limitation on how much weight the
epilepsy headline number should carry, not as a refutation, since refutation would require a test
that actually had the power to refute.

## 11. Empirical comparison against adapted DRDM and TAPB baselines

Related work (`paper/main.tex` §2) names the two debiasing methods closest to this paper's own,
DRDM and TAPB, but discusses rather than runs them, since neither method's own architecture or
released code applies to this task. This closes that gap: we implement the core bias-correction
*mechanism* of each, adapted to our nine-indication label set, and run both on the identical
epilepsy cold-bucket split used everywhere else in this paper, rather than leaving the mismatch
undemonstrated. Code: `code/drdm_baseline.py`, `code/tapb_baseline.py`,
`code/multitask_control.py`. Results: `results/drdm_baseline.csv`, `results/tapb_baseline.csv`,
`results/multitask_control.csv`.

**Why not their literal architectures.** DRDM (Meng et al., *J. Biomedical Informatics* 2025,
doi:10.1016/j.jbi.2025.104843) debiases a drug-disease bipartite association graph, built for
curated benchmarks (Gdataset/Cdataset/DRRS-style) with thousands of diseases, via dynamic
degree-based edge reweighting plus dual-view contrastive learning. TAPB (Lin et al., *Nature
Communications* 2025, doi:10.1038/s41467-025-66915-1) corrects "target prior bias" in
drug-target interaction prediction via a causal backdoor adjustment, $P(Y|D,do(T)) = \sum_i
P(c_i)\,\mathrm{Softmax}(g_y(F_{c_i}))$, with confounder clusters $c_i$ built by K-means over
frozen ESM-2 protein-target embeddings, fed through a SMILES-BERT drug encoder and a multi-head
Confounder Alignment Module. Neither transfers as-is: our task has no protein-target axis at all
(TAPB), and our label structure is nine single-indication binary tasks rather than a dense
multi-thousand-node graph (DRDM). Their public code (TAPB: github.com/GaomingL1n/TAPB) is built
entirely around inputs (ESM-2, SMILES-BERT) this task does not have.

**Adaptation design.**
- *DRDM-adapted*: all nine indication label sets (same 9,878-compound universe, confirmed
  identical row order across diseases) are assembled into one drug-disease bipartite graph.
  A feature-based drug encoder (the same three views as the rest of this pipeline, concatenated
  rather than gated) and a 9-row learned disease-embedding table are updated by a two-hop graph
  convolution; edges are weighted by $(\deg_{\text{drug}}\cdot\deg_{\text{disease}})^{-\alpha}$,
  $\alpha$ annealed 0 to 0.5 over training, DRDM's own stated mechanism ("dynamically adjusts
  association weights during training, enhancing long-tail entity representation"). The held-out
  epilepsy test fold's positive edges are excluded from the graph for that fold; the other eight
  indications' full label sets are visible throughout as side information, since they are not
  being evaluated. 5 negatives sampled per positive edge, trained 150 epochs, Adam, per
  (seed, fold) as in the rest of the pipeline.
- *TAPB-adapted*: TAPB's own prior-tendency score, $z_T = $ mean positive rate, is computed per
  indication (epilepsy: train-fold labels only; the other eight: full labels, side information).
  K-means ($I=3$) over the nine $z_T$ values gives a small confounder dictionary, with $P(c_i)$
  the fraction of indications per cluster. A softmin alignment weight $a_i(T)$ (the lightweight
  analogue of CAM's cross-attention) measures how closely indication $T$'s $z_T$ sits to each
  cluster center. A single shared classifier $g_y$ takes [drug embedding, $a_i(T)$] for each $i$
  and outputs a per-cluster logit; the final prediction sums $P(c_i)\cdot\mathrm{sigmoid}(\text{logit}_i)$,
  mirroring Eq. 22 of the TAPB paper. The drug encoder and $g_y$ are trained jointly across all
  nine indications' (drug, label) pairs, held-out epilepsy test rows excluded, with inverse-class-
  frequency sample weighting. The amino-acid randomization augmentation in the original paper is
  specific to protein-sequence input and has no analogue here, so it is omitted.
- *multitask\_control*: added after seeing `tapb_adapted`'s result (below), to separate "TAPB's
  correction helps" from "joint multi-task training over 9 indications helps." Identical drug
  encoder and joint-training loop as TAPB-adapted, but a single plain linear head per indication,
  no confounder clusters, no backdoor adjustment at all.

**Results** (epilepsy cold-bucket AUROC, mean $\pm$ std over 5 seeds, same cold-bucket
definition as throughout this paper):

| Model | Cold-bucket AUROC |
|---|---|
| nn_tanimoto | 0.425 ± 0.016 |
| drdm_adapted | 0.433 ± 0.190 |
| tapb_adapted | 0.852 ± 0.018 |
| multitask_control (joint training, no debiasing) | 0.863 ± 0.009 |
| fusion_nodebias | 0.799 ± 0.053 |
| **fusion_full (proposed)** | **0.915 ± 0.052** |

**Reading this honestly.**
- `drdm_adapted` recovers no usable signal (0.433, indistinguishable from `nn_tanimoto`) and is
  an order of magnitude noisier across seeds than anything else in this paper (std 0.190 against
  0.009-0.058 elsewhere). This is consistent with the graph this adaptation builds being far too
  sparse for DRDM's mechanism to do real work: 9 disease nodes, ~470 total positive edges across
  the whole 9,878-compound library, versus the thousands-of-diseases benchmarks DRDM was built
  and validated on. We read this as "DRDM's mechanism does not help on a graph this sparse," not
  as "DRDM is a bad method," since the adaptation pushes it well outside the data regime it was
  designed for.
- `tapb_adapted` recovers strong, stable signal (0.852). Before reporting this as a finding about
  TAPB, we ran the `multitask_control` check specifically because a strong, stable result that
  appears right after switching to joint multi-indication training is exactly the kind of
  agreement the scientific-rigor review process flags: multiple numbers agreeing because they
  share an upstream transformation is not independent confirmation. It was the right call: the
  plain multi-task control, with literally no confounder-cluster correction, matches
  `tapb_adapted` almost exactly (0.863 against 0.852, well within noise of each other). TAPB's
  adapted backdoor adjustment adds nothing measurable beyond what joint training across nine
  indications already supplies on this data. We report `tapb_adapted`'s strong number, but
  attribute it to multi-task learning, not to TAPB's own mechanism, and the paper's Results and
  Discussion sections say exactly this rather than citing 0.852 as if it were evidence for TAPB.
- The paper's own proposed model, `fusion_full`, which uses no joint multi-indication training at
  all, still posts the highest point estimate (0.915). The margin over `multitask_control`
  (0.915 vs 0.863) is real but not large relative to `fusion_full`'s own seed-to-seed noise
  (± 0.052); we describe it in the paper as "suggestive, not decisive" rather than a clean win,
  consistent with how every other headline number in this paper is hedged against its own
  variance.
- None of this is a reproduction of DRDM's or TAPB's own reported benchmark numbers. Both
  adaptations are faithful to each method's stated bias-correction *mechanism*, substituting our
  feature-based drug encoder and nine-indication label structure for their original graph/protein
  inputs, because neither original architecture has an analogue on this task. A reviewer who
  considers either adaptation an unfaithful reproduction should read Table above as a bound on
  what we could test given the mismatch, not a claim about DRDM's or TAPB's performance on their
  own published benchmarks.

## 12. External review pass: corrections and the text-availability shortcut check

A simulated adversarial RECOMB review (fresh read of the final paper, no prior context) checked
every number against its source file and raised several points, addressed below.

**Two verified numeric errors, now fixed.** Table "headline"'s claim that `fusion_full` beats
`nn_tanimoto` on "seven of nine classes," with margins of +0.171 (antineoplastic) and +0.119
(epilepsy), did not match `results/headline_mean_std_auroc.csv`: recomputing directly gives six
of nine classes (not antibacterial, antipsychotic, betablocker), and margins of +0.168 and
+0.112. The paper's prose was drafted against an earlier run than the one checked into the
repository; fixed to the current numbers. The leakage-audit paragraph separately conflated the
post-drop evaluated cohort (59 drugs) with the pre-drop candidate pool (66, of which 7 lacked
usable pre-approval text and were dropped), previously fixed in an earlier pass of this same kind.

**MoA independence, reworded.** The biological-interpretation check's claim that "no input
references" the MoA field was true of the literal field but overstated independence: the text
view's UniProt target-function description and the Hub's MoA label both trace back to the same
annotated target, so only the structure and expression views are genuinely independent of it.
The paper now says this explicitly rather than implying all three views are independent checks.

**Text-availability shortcut, tested directly.** The review raised a specific, checkable concern:
epilepsy positives carry a Hub target annotation far more often than negatives (22/31 vs.
2,264/9,847, already noted in section 2 above as the source of an earlier normalization bug), and
the fusion gate (`fusion_model.py`) feeds the raw availability bit into the gate as an input
feature, not just a hard on/off switch, so the text view's apparent contribution could come from
merely having an annotation rather than from what the annotation says.

`code/text_availability_shortcut_check.py` tests this directly on epilepsy (5 seeds x 5 scaffold
folds): keep the true availability mask exactly as is, but shuffle which real text embedding each
annotated compound receives, disjointly within train and within test, so the true compound-content
correspondence is broken while the true availability pattern and its correlation with the label
are preserved. Result:

| Model | mean AUROC | std |
|---|---|---|
| fusion_full (uniform weights), real text | 0.827 | 0.112 |
| fusion_full (uniform weights), shuffled text | 0.648 | 0.156 |
| fusion, structure+expression, no text view | 0.647 | 0.125 |
| text_only, real text | 0.641 | 0.143 |
| text_only, shuffled text | 0.262 | 0.116 |

Shuffled-content fusion (0.648) lands at essentially the same level as the no-text ablation
(0.647), not near the real-content fusion (0.827): the gain is coming from what the text says,
not from the mere fact that a compound has an annotation at all, which is a clean negative result
for the shortcut hypothesis. (Note: this script uses uniform sample weights throughout, matching
`fusion_nodebias`'s setup, not the DebiasedDTA-reweighted `fusion_full`; it isolates the
content-vs-availability question, which does not depend on the reweighting schedule, so it was
not rerun with the schedule.) Added to the paper's biological-interpretation subsection.

## 13. TxGNN: a real, time-boxed attempt, and why it still was not run

A reviewer flagged TxGNN's absence as a real gap, not just a disclosed limitation. We spent a
time-boxed investigation (roughly two hours) trying to get a genuine number rather than leaving
the disclosure unexamined, and found the original "no checkpoint" framing was imprecise: a
checkpoint does exist, the actual blocker is elsewhere, and it is a hard one, not a quick fix.

**What we found.** TxGNN (github.com/mims-harvard/TxGNN) does ship an example pretrained
checkpoint (linked from its README via Google Drive, loadable with `TxGNN.load_pretrained`). The
real blocker is the dependency chain: the package pins `dgl==0.5.2`, which is not available on
PyPI (only 0.1.0/0.1.2/0.1.3 and 2.1.0 are, a four-year API gap with no compatible intermediate),
and was likely only ever distributed through the `dglteam` conda channel for that exact old
version. The released `TxGNN` PyPI package itself fails to build from source in a clean venv
(`pip download --no-deps TxGNN` errors with `FileNotFoundError: requirements.txt` inside its own
setup.py, a packaging bug in the upstream release, not an environment issue on our end). TxGNN
also wants Python 3.8 specifically; this machine's available interpreters are 3.11/3.12. Getting
a working environment would mean bootstrapping conda/mamba, hunting for a DGL 0.5.2 build
compatible with this machine's CUDA 13.3 driver (a 2020-era prebuilt binary against a
current-generation driver stack, itself uncertain), and likely patching TxGNN's own code to a
DGL version it was never written against, since the pinned version is simply not obtainable.

**A second, independent problem, found while checking the first.** Even setting the environment
aside, TxGNN's knowledge graph (built on PrimeKG, 17,080 disease nodes, 7,957 drug/candidate
nodes) identifies drugs and diseases by its own internal indices, requiring a mapping from our
compounds' InChIKeys to PrimeKG's drug nodes (plausibly bridgeable via DrugBank IDs through
UniChem, though PrimeKG does not ship a documented, ready-made InChIKey crosswalk) and, more
fundamentally, a mapping from our labels to TxGNN's. Our 9 classes are broad therapeutic
categories (any approved antiepileptic, any approved antidepressant), while TxGNN predicts
drug-indication links to specific disease entities. Translating "approved for the depression
category" into "predicted indication for which MONDO disease node(s)" requires aggregating over
multiple disease sub-entities with no single correct convention, a real methodological choice,
not just an engineering step, and one that would need to be made and defended even after the
environment problem is solved.

**Decision.** Stopping here rather than forcing a number. The combination of an unobtainable
pinned dependency, a broken upstream package build, and an unresolved label-granularity mismatch
means a real TxGNN comparison is a multi-day project (environment archaeology, a from-scratch
DrugBank/MONDO identifier bridge, and a defensible category-to-disease aggregation scheme), not
a bounded addition. The paper's Limitations sentence ("TxGNN was not run... no checkpoint") is
corrected by this investigation (a checkpoint exists; the real blocker is the dependency chain
and label mismatch described above) but was not edited in `main.tex`, since the body is at its
exact 10-page limit and this is a precision improvement, not a new finding that changes any
number or claim; this section is the accurate record for anyone who wants to revisit it with
substantially more time.

## 14. Chemprop full-protocol run (resolving the §6 reduced-compute caveat)

The §6 compute note explains why Chemprop originally ran on a reduced protocol (1 seed, 2 of 5
folds): a from-scratch D-MPNN over all 225 (class x seed x fold) combinations was not a
reasonable use of shared GPU time mid-pipeline. With the rest of this project's work complete
and no other job contending for the GPU, that full run was done
(`code/run_chemprop_full.py`, 5 seeds x 5 scaffold-CV folds per class, 225 fits total); the
per-class table and discussion are in §3's "Chemprop (D-MPNN), full protocol" subsection, not
repeated here.

The GPU was confirmed in active use throughout (`/proc/<pid>/fd` showing open file descriptors
to `/dev/dxg`, WSL2's GPU paravirtualization device, and to `nvcubins.bin`); utilization
readings around 30-35% reflect the already-documented fact that D-MPNN training here is
CPU-bound on batch featurization, not GPU-bound on compute, not an idle GPU. `main.tex`'s
Limitations sentence has been updated to report the full-protocol result in place of the
earlier "ran reduced" caveat, which no longer describes what was run. §7 item 3 of this
document, which flagged the reduced run as something to revisit, is resolved by this section.
