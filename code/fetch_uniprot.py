"""Fetch UniProt function-description text for every unique target gene symbol in targets.parquet.

Caches raw JSON per gene under data/uniprot_function_text/<GENE>.json (so reruns are free),
then flattens to data/target_function_text.parquet (gene_symbol, uniprot_acc, function_text).

Polite: modest thread pool (6 workers), small per-request delay, retries with backoff.
"""
import json
import os
import time
import concurrent.futures as cf
import requests
import pandas as pd

BASE = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing"
CACHE_DIR = f"{BASE}/data/uniprot_function_text"
GENES_FILE = f"{BASE}/data/unique_target_genes.txt"
OUT_PARQUET = f"{BASE}/data/target_function_text.parquet"
LOG = f"{BASE}/logs/uniprot_fetch.log"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(f"{BASE}/logs", exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "RECOMB2027-idea1-research/1.0 (academic, non-commercial)"})

URL_TMPL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene:{gene}+AND+organism_id:9606+AND+reviewed:true"
    "&fields=accession,gene_names,cc_function"
    "&format=json&size=1"
)


def safe_name(gene):
    return gene.replace("/", "_").replace(" ", "_")


def fetch_one(gene):
    cache_path = os.path.join(CACHE_DIR, safe_name(gene) + ".json")
    if os.path.exists(cache_path):
        return gene, "cached"
    url = URL_TMPL.format(gene=gene)
    for attempt in range(4):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                time.sleep(0.15)
                return gene, "ok"
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1 + attempt)
    # write empty marker so we don't retry forever on genuinely bad genes
    with open(cache_path, "w") as f:
        json.dump({"results": [], "_error": "failed_after_retries"}, f)
    return gene, "failed"


def main():
    with open(GENES_FILE) as f:
        genes = [g.strip() for g in f if g.strip()]
    print(f"Total unique genes: {len(genes)}")

    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_one, g): g for g in genes}
        done = 0
        with open(LOG, "a") as logf:
            logf.write(f"\n=== UniProt fetch run start {time.ctime()} ===\n")
            for fut in cf.as_completed(futs):
                gene, status = fut.result()
                results.append((gene, status))
                done += 1
                if done % 50 == 0 or done == len(genes):
                    msg = f"{done}/{len(genes)} done, elapsed {time.time()-t0:.0f}s"
                    print(msg, flush=True)
                    logf.write(msg + "\n")

    status_counts = pd.Series([s for _, s in results]).value_counts()
    print(status_counts)

    # flatten
    rows = []
    for gene in genes:
        cache_path = os.path.join(CACHE_DIR, safe_name(gene) + ".json")
        acc, func_text = None, None
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                res = data.get("results", [])
                if res:
                    entry = res[0]
                    acc = entry.get("primaryAccession")
                    comments = entry.get("comments", [])
                    texts = []
                    for c in comments:
                        if c.get("commentType") == "FUNCTION":
                            for t in c.get("texts", []):
                                texts.append(t.get("value", ""))
                    func_text = " ".join(texts) if texts else None
            except Exception:
                pass
        rows.append({"gene_symbol": gene, "uniprot_acc": acc, "function_text": func_text})

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    n_with_text = df["function_text"].notna().sum()
    print(f"Genes with usable function text: {n_with_text}/{len(df)}")
    with open(LOG, "a") as logf:
        logf.write(f"Genes with usable function text: {n_with_text}/{len(df)}\n")


if __name__ == "__main__":
    main()
