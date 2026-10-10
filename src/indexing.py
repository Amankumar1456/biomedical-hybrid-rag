"""
Builds and persists both indexes over the passage corpus:
  - BM25 (lexical) via rank_bm25
  - FAISS flat inner-product index over sentence-transformer embeddings (dense)

Persisting these means the app and eval harness never rebuild indexes
per-run — they load once at startup.
"""
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parent.parent / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "dense.faiss"
IDMAP_PATH = INDEX_DIR / "id_map.pkl"

# Biomedical-domain sentence embedding model. Falls back to a general
# model if you'd rather trade domain fit for speed during dev.
EMBED_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _token_re.findall(text.lower())


def build_bm25(corpus_df, passage_col="passage"):
    tokenized = [tokenize(t) for t in corpus_df[passage_col].tolist()]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    return bm25


def build_dense(corpus_df, passage_col="passage", model_name=EMBED_MODEL_NAME, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus_df[passage_col].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim via inner product
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    return index, model


def build_all(corpus_df, id_col="id", passage_col="passage"):
    id_map = corpus_df[id_col].tolist()  # positional index -> passage_id
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

    bm25 = build_bm25(corpus_df, passage_col)
    dense_index, model = build_dense(corpus_df, passage_col)
    return bm25, dense_index, model, id_map


def load_bm25():
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def load_dense():
    return faiss.read_index(str(FAISS_PATH))


def load_id_map():
    with open(IDMAP_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from data_loader import load_and_cache

    corpus_df, _ = load_and_cache()
    print(f"Building indexes over {len(corpus_df)} passages...")
    build_all(corpus_df)
    print(f"Done. Indexes written to {INDEX_DIR}")
