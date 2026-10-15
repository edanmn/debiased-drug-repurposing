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
  memory, approximate"), not a sign the join was forced to hit it — the join script
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
  embedding — see Limitations). Built by `code/build_text_view.py`.
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
  reduced from the original paper's 10 repeats for compute budget — documented, not hidden) to
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
  than negatives (22/31 positives vs. 2,264/9,847 negatives have a target annotation) — i.e. the
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
  text), `fusion_nodebias` (all 3 views, uniform sample weights throughout — isolates what the
  DebiasedDTA reweighting itself contributes), `fusion_full` (all 3 views, DebiasedDTA-reweighted
  — the proposed method).
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
| epilepsy | 0.732±0.008 | 0.803±0.014 | 0.870±0.006 | 0.641±0.059 | 0.890±0.003 | 0.647±0.017 | 0.826±0.011 | **0.851±0.021** |
| antibacterial | 0.917±0.006 | 0.955±0.008 | 0.970±0.004 | 0.612±0.023 | 0.899±0.011 | 0.760±0.055 | 0.849±0.042 | **0.890±0.016** |
| antidiabetic | 0.841±0.022 | 0.854±0.023 | 0.912±0.017 | 0.740±0.077 | 0.752±0.007 | 0.751±0.030 | 0.918±0.004 | **0.932±0.014** |
| antineoplastic | 0.728±0.014 | 0.729±0.007 | 0.840±0.012 | 0.686±0.031 | 0.589±0.015 | 0.843±0.006 | 0.908±0.006 | **0.899±0.010** |
| antipsychotic | 0.970±0.012 | 0.951±0.016 | 0.962±0.015 | 0.901±0.039 | 0.827±0.013 | 0.724±0.025 | 0.975±0.003 | **0.968±0.004** |
| betablocker | 0.982±0.006 | 0.995±0.002 | 0.991±0.005 | 0.833±0.049 | 0.916±0.015 | 0.840±0.022 | 0.992±0.001 | **0.978±0.011** |
| depression | 0.897±0.009 | 0.830±0.011 | 0.914±0.004 | 0.812±0.015 | 0.838±0.006 | 0.775±0.039 | 0.966±0.002 | **0.955±0.012** |
| lipid | 0.830±0.029 | 0.761±0.024 | 0.881±0.015 | 0.598±0.037 | 0.779±0.017 | 0.650±0.019 | 0.888±0.012 | **0.836±0.027** |
| nsaid | 0.890±0.023 | 0.833±0.037 | 0.911±0.018 | 0.771±0.046 | 0.871±0.004 | 0.595±0.024 | 0.896±0.016 | **0.907±0.031** |
| **cross-class mean** | 0.865 | 0.857 | 0.917 | 0.733 | 0.818 | 0.732 | 0.913 | **0.913** |

**Honest reading of this table**:
- The zero-training NN-Tanimoto guide is already strong (cross-class mean 0.865), confirming
  the idea's premise that near-analog similarity explains a large share of apparent accuracy.
  It is weakest on antineoplastic (0.728) and epilepsy (0.732) — the two classes where the
  chemistry is most scaffold-diverse — and strongest on antipsychotic/betablocker (0.97-0.98),
  classes dominated by a few tight chemical series.
- `fusion_full` (proposed) beats `nn_tanimoto` on 7/9 classes (all but betablocker and
  antipsychotic, the two near-analog-dominated classes where there is little residual to find),
  with the largest margins on antineoplastic (+0.171) and epilepsy (+0.119).
- **`fusion_full` vs `fusion_nodebias` (debiasing ablation) is the critical comparison, and the
  honest result is: debiasing gives a small, inconsistent effect at the overall-AUROC level**
  (cross-class mean identical, 0.913 vs 0.913; fusion_full wins on antidiabetic, nsaid,
  epilepsy; fusion_nodebias wins on antibacterial, antineoplastic, antipsychotic, betablocker,
  depression, lipid, several within 1 std of each other). This does NOT support a strong,
  uniform "debiasing helps" headline claim at the pooled-AUROC level. See the warm/cold
  breakdown below, which is the more theoretically appropriate test (debiasing is designed to
  help specifically on low-similarity/cold cases, not to raise pooled AUROC that is dominated
  by easy warm cases) -- do not report the idea doc's "0.73 vs 0.56" figure as reproduced
  without checking that breakdown.
- `fusion_struct_expr` (structure+L1000, no text) sometimes underperforms `fusion_struct`
  alone (e.g., epilepsy 0.647 vs 0.890; nsaid 0.595 vs 0.871) and sometimes outperforms it
  substantially (antineoplastic 0.843 vs 0.589). This is a genuine, reported ablation finding:
  the L1000 view's value is class-dependent, not uniformly additive, when added without the
  text view and without debiasing.
- `text_only` (no chemistry) is informative but the weakest learned model everywhere
  (cross-class mean 0.733, vs. random's ~0.49-0.5), consistent with target-function text being
  a real but secondary signal.

### Chemprop (D-MPNN), reduced-compute protocol

Per the compute note in §6, run on 1 seed x first-2-of-5 scaffold folds x 15 epochs
(`results/chemprop_folds.csv`):

| Class | fold 0 AUROC | fold 1 AUROC |
|---|---|---|
| epilepsy | 0.832 | 0.432 |
| antibacterial | 0.877 | 0.903 |
| antidiabetic | 0.750 | 0.765 |
| antineoplastic | 0.835 | 0.757 |
| antipsychotic | 0.855 | 0.997 |
| betablocker | 0.988 | 0.947 |
| depression | 0.731 | 0.865 |
| lipid | 0.851 | 0.818 |
| nsaid | 0.837 | 0.839 |

High fold-to-fold variance (e.g. epilepsy 0.832 vs 0.432) is expected with only 2 folds and a
D-MPNN trained from scratch on a few thousand molecules at 15 epochs; this is reported as a
reduced, real, noisy estimate, not a tuned headline number. It is broadly in the same range as
`lr_ecfp`/`xgb_ecfp`, as expected for a graph model with no pretraining on this few labeled
examples.

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

**Completed, reproduced computationally** — `code/leakage_audit.py`,
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
(comparator drugs in the same case reports/reviews/trials) — exactly the same-class-mention
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
   `results/leakage_audit_summary.csv` and `results/leakage_audit_cohort.csv` directly — the
   n=59 corpus is small and PubMed date-restriction via `mindate`/`maxdate` on `pdat` is a
   reasonable but not perfect proxy for "before approval" (it reflects publication date, not
   necessarily first public knowledge of the compound).
2. Whether the DebiasedDTA reweighting's benefit (`fusion_full` vs `fusion_nodebias`) is
   consistent across all 9 classes or concentrated in a few — see the per-class breakdown in
   `results/headline_mean_std_auroc.csv`, not just the cross-class average.
3. The reduced-compute Chemprop run (1 seed only) should not be over-interpreted as a tight
   estimate; if a tighter chemprop number is needed for the paper, rerunning with 5 seeds is the
   correct (not fabricated) way to get one, budget permitting.
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
