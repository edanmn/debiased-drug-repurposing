"""Build the text view: PubMedBERT embeddings of each target gene's UniProt function text,
averaged per-compound across its known target(s). Compounds without target/text data get
an all-zero vector plus mask=False (handled explicitly downstream, not a learned missing
embedding -- see RESULTS.md for the documented design choice).

Outputs (aligned to compounds.parquet row order, 9878 rows):
  data/cmp_text_emb.npy   (9878, 768) float32
  data/cmp_text_mask.npy  (9878,) bool  -- True if >=1 target gene had usable function text
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

BASE = "/mnt/c/Users/rishik/epilepsy_repurpose/RECOMB2027/idea1_debiased_repurposing"
CMP = "/mnt/c/Users/rishik/epilepsy_repurpose/ICHI2027_mgan/data/compounds.parquet"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

device = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


def main():
    cmp = pd.read_parquet(CMP)
    targets = pd.read_parquet(f"{BASE}/data/targets.parquet")
    func = pd.read_parquet(f"{BASE}/data/target_function_text.parquet")
    func = func.dropna(subset=["function_text"])
    gene2text = dict(zip(func.gene_symbol, func.function_text))
    print(f"Genes with usable function text: {len(gene2text)} / targets file unique genes")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    unique_texts = sorted(set(gene2text.values()))
    print(f"Unique function texts to encode: {len(unique_texts)}")
    text2emb = {}
    BATCH = 16
    with torch.no_grad():
        for i in range(0, len(unique_texts), BATCH):
            batch = unique_texts[i:i + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()
            for t, e in zip(batch, pooled):
                text2emb[t] = e
            if (i // BATCH) % 10 == 0:
                print(f"encoded {i+len(batch)}/{len(unique_texts)}", flush=True)

    gene2emb = {g: text2emb[t] for g, t in gene2text.items() if t in text2emb}

    ik2genes = dict(zip(targets.ik14, targets.target_genes))
    n = len(cmp)
    dim = 768
    emb = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    n_hit = 0
    for i, ik in enumerate(cmp.ik14.values):
        genes_str = ik2genes.get(ik)
        if genes_str is None:
            continue
        genes = genes_str.split("|")
        vecs = [gene2emb[g] for g in genes if g in gene2emb]
        if vecs:
            emb[i] = np.mean(vecs, axis=0)
            mask[i] = True
            n_hit += 1

    print(f"Compounds with a usable text-view embedding: {n_hit} / {n} "
          f"(of {len(ik2genes)} compounds with any target annotation)")

    np.save(f"{BASE}/data/cmp_text_emb.npy", emb)
    np.save(f"{BASE}/data/cmp_text_mask.npy", mask)
    print("Saved cmp_text_emb.npy and cmp_text_mask.npy")


if __name__ == "__main__":
    main()
