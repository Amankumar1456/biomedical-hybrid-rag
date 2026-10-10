"""
Runs every required configuration (Step 3) over the same fixed 100
queries, computing retrieval metrics + latency + a rough cost proxy
per query (Step 4's traditional-metrics layer). Results are written
per-config so run_llm_judge.py and the manual-review step can load
them without re-running retrieval.

Configs run: lexical, dense, hybrid, hybrid_expansion, and one
"strongest variant" slot you fill in once you see which config wins
(e.g. hybrid_expansion with a tuned per_mode_k, or a reranker added on
top — pick this AFTER looking at the first four results, that's the
point of the comparison).
"""
import json
import pickle
import time
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import load_and_cache
from indexing import load_bm25, load_dense, load_id_map, EMBED_MODEL_NAME
from retrieval import Retriever
from metrics import compute_all
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
QUERYSET_PATH = Path(__file__).resolve().parent / "queryset.parquet"

CONFIGS = ["lexical", "dense", "hybrid", "hybrid_expansion"]
# Add your chosen strongest variant here once Steps 1-4's first pass is done, e.g.:
# CONFIGS.append("hybrid_expansion_tuned")


def build_retriever() -> Retriever:
    corpus_df, _ = load_and_cache()
    bm25 = load_bm25()
    dense_index = load_dense()
    id_map = load_id_map()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    corpus_texts = corpus_df["passage"].tolist()
    return Retriever(bm25, dense_index, embed_model, id_map, corpus_texts)


def run_config(retriever: Retriever, queryset: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    for _, row in queryset.iterrows():
        relevant = set(row["relevant_passage_ids"])  # adjust field name to actual schema
        start = time.perf_counter()
        hits = retriever.search(row["question"], mode=mode, top_k=10)
        latency = time.perf_counter() - start

        retrieved_ids = [h.passage_id for h in hits]
        m = compute_all(retrieved_ids, relevant)
        rows.append({
            "question_id": row.get("id", row.name),
            "question": row["question"],
            "mode": mode,
            "latency_s": latency,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [h.score for h in hits],
            **m,
        })
    return pd.DataFrame(rows)


def main():
    queryset = pd.read_parquet(QUERYSET_PATH)
    retriever = build_retriever()

    all_results = []
    for mode in CONFIGS:
        print(f"Running config: {mode}")
        df = run_config(retriever, queryset, mode)
        df.to_parquet(RESULTS_DIR / f"{mode}.parquet")
        all_results.append(df)
        print(df[["recall@5", "recall@10", "mrr@10", "ndcg@10", "latency_s"]].mean())

    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby("mode")[["recall@5", "recall@10", "mrr@10", "ndcg@10", "latency_s"]].mean()
    summary.to_csv(RESULTS_DIR / "comparison_table.csv")
    print("\n=== Comparison table ===")
    print(summary)


if __name__ == "__main__":
    main()
