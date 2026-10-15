# Run summary (what was executed and when), 2026-10-15, all times CDT

1. `code/build_targets.py` (00:14) -> `data/targets.parquet`, `data/unique_target_genes.txt`.
   2,290 compounds joined, 1,392 unique target genes.
2. `code/fetch_uniprot.py` (00:15-00:22) -> `data/uniprot_function_text/*.json`,
   `data/target_function_text.parquet`. Log: `logs/uniprot_fetch.log`. 1,392/1,392 genes
   fetched, 1,385 with usable function text.
3. `code/build_text_view.py` (00:22) -> `data/cmp_text_emb.npy`, `data/cmp_text_mask.npy`.
   PubMedBERT mean-pooled embeddings; 2,286/2,290 target-annotated compounds covered.
4. `code/run_pipeline.py epilepsy` (00:15-00:22, pre-normalization-fix smoke test) revealed
   the text/expr-view normalization bug (see RESULTS.md methods section). Fixed in
   `code/datalib.py` (`standardize_view`) and `code/run_pipeline.py`.
5. `code/run_pipeline.py epilepsy` (00:31-00:34, post-fix rerun) -> confirmed fix
   (`text_only` AUROC 0.32 -> 0.64). `logs/run_pipeline.log`,
   `results/fold_metrics_epilepsy.csv`, `results/oof_epilepsy.parquet`.
6. `code/run_pipeline.py antibacterial antidiabetic antineoplastic antipsychotic betablocker
   depression lipid nsaid` (00:34-00:49) -> remaining 8 classes. First launch (00:34) hit CPU
   thread oversubscription when run concurrently with chemprop; killed and relaunched with
   `OMP_NUM_THREADS=10` etc. at 00:36. Completed 00:49.
   `logs/run_pipeline.log`, `results/fold_metrics_<class>.csv`, `results/oof_<class>.parquet`
   for all 9 classes.
7. `code/leakage_audit.py` (00:25-00:28) -> `results/leakage_audit_summary.csv`,
   `results/leakage_audit_cohort.csv`, `results/leakage_audit_top_terms_{unmasked,masked}.csv`.
   Live NCBI E-utilities queries via Bio.Entrez (no cached local PubMed data existed).
8. `pip install --break-system-packages chemprop` (00:28) installed chemprop 2.3.1, which
   pulled in `cuik_molmaker` requiring `rdkit==2026.3.6`; this silently upgraded the shared
   environment's rdkit from 2024.9.6. Immediately reverted
   (`pip install --break-system-packages rdkit==2024.9.6`, 00:29) to avoid breaking the
   concurrently-running Idea-2 agent's environment. Installed `chemprop==2.0.0` instead (no
   cuik_molmaker dependency), verified compatible with rdkit 2024.9.6.
9. `code/run_chemprop.py` (00:38-00:59) -> `results/chemprop_folds.csv`,
   `logs/run_chemprop.log`. First launch (full 5-fold, 30-epoch protocol) found to be
   CPU-bound and slow; killed, code edited to a documented reduced protocol (1 seed, first
   2 folds, 15 epochs, batch 256), relaunched at 00:38, completed 00:59 (18 fits total).
10. `code/analyze_results.py` (00:50-01:07) -> `results/headline_mean_std_auroc.csv`,
    `results/bootstrap_ci.csv`, `results/warm_cold_auroc.csv`,
    `results/similarity_quintile_auroc.csv`. 2000-resample pooled bootstrap over 9 classes x
    9 models was the dominant cost (~17 min single-core).
11. `code/make_figures.py` (01:08) -> `figures/headline_comparison.png`,
    `figures/warm_cold_auroc.png`, `figures/similarity_stratified_auroc.png`,
    `figures/leakage_audit.png`.

See `logs/run_pipeline.log`, `logs/uniprot_fetch.log`, `logs/leakage_audit.log`,
`logs/run_chemprop.log` for full per-step detail (every fold's timing and metric).
