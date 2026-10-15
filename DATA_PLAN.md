# Data plan — Idea 1: Repurposing Beyond the Nearest Analog

Confirmed by literature review (`literature_review.md`) and a full local data inventory. Everything below is either an established public dataset with a citable resource paper, or an existing in-repo file built by prior work (`ICHI2027_mgan`). Nothing here is an invented dataset.

## Reuse directly (already built, in `ICHI2027_mgan/data/`)
- `compounds.parquet` — master compound table, 9,878 LINCS compounds, keyed by `ik14`.
- `cmp_ecfp.npy` — ECFP4 fingerprints, shape (9878, 2048), built by `build_compounds.py` via RDKit Morgan.
- `cmp_expr.npy` — consensus L1000 landmark-gene (978-gene) signatures, shape (9878, 978), built by `build_expression.py` from raw LINCS Phase 1 (`lincs_phase1/L5_phase1.gctx`, GSE92742) and Phase 2 (`lincs_phase2/GSE70138_...gctx`) GCTX files.
- `labels_antibacterial.csv`, `labels_antidiabetic.csv`, `labels_antineoplastic.csv`, `labels_antipsychotic.csv`, `labels_betablocker.csv` (antihypertensive proxy), `labels_depression.csv`, `labels_lipid.csv`, `labels_nsaid.csv`, plus epilepsy via `chembl_epilepsy_indications.csv` + `data/known_aeds.csv` — nine ChEMBL-derived indication label sets, each 9,878 rows, cols `ik14,name,max_phase,ind_name,atc,positive,clinical`.
- `chembl_max_phase_all.csv` — max ChEMBL phase per compound (for stratifying by development stage if useful).

## Reuse with a derivation step
- **Repurposing Hub target annotations**: full Broad Hub export at `PACT/PACT/data/repurposing_drugs_20200324.txt` (6,798 compounds, cols `pert_iname, clinical_phase, moa, target, disease_area, indication`). The "2,290 compounds" figure in the idea doc is the intersection of this file with the 9,878 LINCS compounds that have a non-null `target` field — this subset is **not pre-computed** and must be derived by joining on compound name/InChIKey against `compounds.parquet`. Build script: `code/build_targets.py` → `data/targets.parquet` (ik14, target gene symbols, moa, disease_area).

## Must newly fetch (no legitimacy concern — standard public resource, just not cached locally)
- **UniProt function-description text**: not present anywhere in the tree. Fetch via UniProt REST API for every unique target gene symbol found in the derived `targets.parquet` (expect on the order of a few hundred to ~1,000 unique human gene targets). Cache raw JSON/text under `data/uniprot_function_text/` keyed by UniProt accession, and a flattened `data/target_function_text.parquet` (gene_symbol, uniprot_acc, function_text).
- PubMedBERT itself is already cached locally at `~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` — no download needed, just load from cache.

## Explicitly excluded (by design, not an oversight)
- PubMed abstracts about the drug itself — excluded as a feature source due to confirmed label leakage (idea doc's own leakage test, to be reproduced and reported, not just asserted).

## Baselines — status
- **Zero-training nearest-neighbor (NN) search on ECFP4 Tanimoto similarity**: trivial to build directly from `cmp_ecfp.npy`; this is also the "guide" model for the DebiasedDTA-style reweighting.
- **Plain fingerprint model**: logistic regression / XGBoost on ECFP4 — `ICHI2027_mgan/code/candidates.py` and `run_baselines.py` already implement a version of this; reuse/extend rather than rewrite.
- **Chemprop (D-MPNN)**: legitimate standard baseline (Yang et al. 2019; Heid et al. 2024). Install via pip (`chemprop`) and run on SMILES from `compounds.parquet`.
- **TxGNN**: legitimate but evaluates a *different* zero-shot construct (unseen disease in PrimeKG, not unseen chemical scaffold). Attempt only if a usable pretrained checkpoint/package can be obtained within reasonable effort; if infeasible, report as "not run, construct mismatch noted" rather than fabricating a number — see literature_review.md §4 for why a direct comparison needs careful framing regardless.
- **Target-knowledge-only model**: ablation using only the target/text view, no chemistry.

## Known caveats to carry into the paper (from literature_review.md)
- Cite DebiasedDTA as *J. Comput. Biol.* 30(11):1226-1239, 2023 — do not assert RECOMB 2023 acceptance (unconfirmed by direct program listing).
- Must cite and explicitly distinguish from DRDM (Meng et al., *J. Biomed. Inform.* 2025 — confirmed venue) and TAPB (*Nat. Commun.* 2025) in related work.
- Leakage audit (PubMed-abstract AUROC 0.97 via drug-class mentions) must be reproduced computationally in this implementation, not just cited from the idea doc.
