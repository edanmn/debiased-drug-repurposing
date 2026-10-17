# Repurposing beyond the nearest analog

Code and results for a RECOMB 2027 submission on debiasing drug-indication prediction against
chemical nearest-neighbor-analog bias, using a DebiasedDTA-style (Ozcelik et al. 2023) guide/predictor
reweighting scheme and a three-view gated fusion model (chemical structure, L1000 expression, and
UniProt target-function text encoded with PubMedBERT).

**Status: not yet published.** This README and the local commit history exist so the repository is
ready to push in one step once the authors confirm it should be made public.

## Layout

- `code/` -- all pipeline and model code (data builds, guide/predictor training, baselines, ablations,
  leakage audit).
- `data/` -- derived data. Several files are symlinks into a sibling project
  (`ICHI2027_mgan/data/`) that built the base compound/fingerprint/expression/label tables from public
  sources (LINCS GEO series GSE92742/GSE70138, the ChEMBL REST API, the Broad Repurposing Hub, UniProt);
  see that project's own `README.md` for the base-layer rebuild instructions. This project's own
  additions (`cmp_text_emb.npy`, `cmp_text_mask.npy`, `target_function_text.parquet`, `targets.parquet`)
  are built by `code/build_targets.py` and `code/build_text_view.py` and are reproducible from the
  public UniProt REST API with no further dependency.
- `results/` -- every CSV/parquet referenced in `RESULTS.md`, produced by the code above.
- `figures/` -- plots referenced in the paper.
- `paper/` -- the LaTeX source (`main.tex`), bibliography (`refs.bib`), and compiled `main.pdf`.
- `RESULTS.md` -- the full, source-of-truth experimental write-up, including what was and was not
  run and why, and an honest statement of which findings are reliable vs. underpowered.
- `literature_review.md` -- the related-work research this submission's citations are drawn from.
- `DATA_PLAN.md` -- the dataset-provenance plan this project followed, confirming every data source
  used is an established public resource.

## Before this repository is made public

- Still open: 14 files under `data/` are tracked as symlinks into a sibling project
  (`ICHI2027_mgan/data/`) on the original author's machine; a clone of this repository alone
  gets broken links for those paths. Replace them with the real files, a fetch script, or a
  hosted-data release before relying on this repo for reproducibility.
- Done: `paper/main.tex`'s code-availability URL now points to the live, pushed repository
  (`https://github.com/edanmn/debiased-drug-repurposing`), confirmed reachable.
- Confirm no large raw LINCS `.gctx` files or other multi-gigabyte intermediates are swept in by an
  unqualified `git add` (none found as of this check).

## Reproducing

See `RESULTS.md` section 1-2 for exactly which script produced which file, and the compute notes in
section 6 for GPU/CPU requirements and known environment caveats (e.g. the `chemprop`/`rdkit` version
pin documented there).
