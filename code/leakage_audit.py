"""Leakage audit (task step 8): does the mere presence of a drug in PubMed abstracts,
restricted to papers published BEFORE that drug's approval date, already predict whether
it is an antiepileptic drug (AED) -- and does that signal survive masking the obvious
seizure/epilepsy/anticonvulsant terms? Follows the methodological framing of the two
clinical-ML label-leakage papers cited in literature_review.md (#14 JAMIA 2023, #15 JAMA
Network Open 2025: same-admission ICD codes leaking the outcome label into EHR models --
here the analogous "same-admission code" is a co-mentioned drug name).

Positive class: known AEDs (data/known_aeds.csv) with a numeric approval_year.
Negative class: a matched-size sample of non-AED drugs with known approval years from two
other therapeutic classes (antineoplastic, depression) already in the repo
(ICHI2027_mgan/data/approval_years_*.csv).

For each drug: PubMed esearch for "<name>"[Title/Abstract], restricted to
mindate=1900/maxdate=<approval_year - 1> (strictly pre-approval), up to 60 abstracts via
efetch, concatenated into one document. TF-IDF + logistic regression, 5-fold CV AUROC,
reported both UNMASKED and with explicit class-defining terms masked
(epilep*, seizur*, anticonvulsant*, antiepileptic*, convulsant*, convulsion*).
"""
import re
import time
import numpy as np
import pandas as pd
from Bio import Entrez
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

Entrez.email = "repurposing.recomb2027.audit@example.com"  # placeholder contact per NCBI E-utilities policy

BASE = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing"
RESULTS = f"{BASE}/results"
LOGF = f"{BASE}/logs/leakage_audit.log"
MASK_TERMS = re.compile(
    r"\b(epilep\w*|seizur\w*|anticonvulsant\w*|antiepileptic\w*|convulsant\w*|convulsion\w*|AEDs?)\b",
    re.IGNORECASE,
)


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def fetch_abstracts(name, max_year, max_n=60, sleep=0.34):
    """Pre-approval-date PubMed abstracts mentioning `name` in title/abstract."""
    term = f'"{name}"[Title/Abstract]'
    try:
        h = Entrez.esearch(db="pubmed", term=term, retmax=max_n,
                            mindate="1900", maxdate=str(int(max_year) - 1), datetype="pdat")
        r = Entrez.read(h)
        time.sleep(sleep)
        ids = r.get("IdList", [])
        if not ids:
            return []
        h2 = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        text = h2.read()
        time.sleep(sleep)
        return [text]
    except Exception as e:
        log(f"  fetch failed for {name}: {type(e).__name__}: {e}")
        return []


def build_corpus():
    aeds = pd.read_csv(f"{BASE}/data/known_aeds.csv")
    aeds = aeds[pd.to_numeric(aeds.approval_year, errors="coerce").notna()].copy()
    aeds["approval_year"] = aeds.approval_year.astype(float)
    pos = aeds[["drug_name", "approval_year"]].rename(columns={"drug_name": "name"})
    pos["label"] = 1

    negs = []
    for f in ["approval_years_antineoplastic.csv", "approval_years_depression.csv"]:
        df = pd.read_csv(f"/mnt/c/Users/rishik/epilepsy_repurpose/ICHI2027_mgan/data/{f}")
        df = df.dropna(subset=["first_approval"])
        negs.append(df[["name", "first_approval"]].rename(columns={"first_approval": "approval_year"}))
    neg = pd.concat(negs, ignore_index=True).drop_duplicates("name")
    neg = neg[~neg.name.str.lower().isin(pos.name.str.lower())]
    rng = np.random.default_rng(0)
    neg = neg.sample(n=min(len(pos), len(neg)), random_state=0).reset_index(drop=True)
    neg["label"] = 0

    cohort = pd.concat([pos, neg], ignore_index=True)
    log(f"Cohort: {len(pos)} AED positives, {len(neg)} non-AED negatives")

    docs, labels, names = [], [], []
    for _, row in cohort.iterrows():
        texts = fetch_abstracts(row["name"], row["approval_year"])
        doc = " ".join(texts)
        n_chars = len(doc)
        log(f"  {row['name']} (label={row['label']}, approved {row['approval_year']:.0f}): "
            f"{n_chars} chars fetched")
        if n_chars < 50:
            continue
        docs.append(doc)
        labels.append(row["label"])
        names.append(row["name"])
    return docs, np.array(labels), names


def run_audit(docs, labels, names, masked):
    if masked:
        docs = [MASK_TERMS.sub(" maskedterm ", d) for d in docs]
    vec = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english",
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
    X = vec.fit_transform(docs)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    n_folds = min(5, n_pos, n_neg)
    if n_folds < 2:
        log("Not enough samples per class for CV; skipping.")
        return None, None, None
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    oof = cross_val_predict(clf, X, labels, cv=skf, method="predict_proba")[:, 1]
    auroc = roc_auc_score(labels, oof)

    clf_full = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, labels)
    feat_names = np.array(vec.get_feature_names_out())
    coefs = clf_full.coef_[0]
    top_idx = np.argsort(-coefs)[:25]
    top_terms = list(zip(feat_names[top_idx], coefs[top_idx]))
    return auroc, oof, top_terms


def main():
    docs, labels, names = build_corpus()
    log(f"Final corpus size: {len(docs)} drugs with usable text "
        f"({labels.sum()} AED positives, {(labels==0).sum()} negatives)")
    pd.DataFrame({"name": names, "label": labels, "doc_len": [len(d) for d in docs]}).to_csv(
        f"{RESULTS}/leakage_audit_cohort.csv", index=False)

    rows = []
    for masked in [False, True]:
        auroc, oof, top_terms = run_audit(docs, labels, names, masked)
        tag = "masked" if masked else "unmasked"
        log(f"=== {tag.upper()} AUROC: {auroc} ===")
        if top_terms:
            log(f"Top positive-class terms ({tag}): {top_terms[:15]}")
        rows.append(dict(condition=tag, auroc=auroc, n=len(labels), n_pos=int(labels.sum())))
        if top_terms:
            pd.DataFrame(top_terms, columns=["term", "coef"]).to_csv(
                f"{RESULTS}/leakage_audit_top_terms_{tag}.csv", index=False)

    pd.DataFrame(rows).to_csv(f"{RESULTS}/leakage_audit_summary.csv", index=False)
    log("Wrote leakage_audit_summary.csv and top_terms files")


if __name__ == "__main__":
    main()
