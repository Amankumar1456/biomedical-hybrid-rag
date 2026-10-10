"""
LLM-judge evaluation layer (Step 4). Runs on top of a retrieval
config's saved results: generates the final answer for each query via
answer.generate_answer, then scores with RAGAS.

Keep the judge model/prompt/settings identical across every config you
compare, or the numbers aren't comparable — RAGAS_JUDGE_MODEL below is
the single place that setting lives.
"""
import pickle
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, context_relevancy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import load_and_cache
from answer import generate_answer
from retrieval import Hit

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAGAS_JUDGE_MODEL = "claude-sonnet-4-6"  # fixed across every config evaluated


def score_config(mode: str, queryset: pd.DataFrame, corpus_lookup: dict):
    retrieval_results = pd.read_parquet(RESULTS_DIR / f"{mode}.parquet")

    rows = []
    for _, r in retrieval_results.iterrows():
        gt_row = queryset[queryset["question"] == r["question"]].iloc[0]
        hits = [
            Hit(pid, score, corpus_lookup.get(pid, ""))
            for pid, score in zip(r["retrieved_ids"], r["retrieved_scores"])
        ]
        result = generate_answer(r["question"], hits, model=RAGAS_JUDGE_MODEL)

        rows.append({
            "question": r["question"],
            "answer": result["answer"],
            "contexts": [h.text for h in hits],  # real passage text — required for context_relevancy
            "ground_truth": gt_row.get("answer", ""),
            "invalid_citations": result.get("invalid_citations", []),
            "insufficient_evidence": result["insufficient_evidence"],
        })

    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])
    judged = evaluate(ds, metrics=[answer_correctness, faithfulness, context_relevancy])

    df.to_parquet(RESULTS_DIR / f"{mode}_judged.parquet")
    print(f"[{mode}] RAGAS scores: {judged}")
    print(f"[{mode}] invalid citations flagged: {df['invalid_citations'].apply(len).sum()}")
    return judged


if __name__ == "__main__":
    queryset = pd.read_parquet(Path(__file__).resolve().parent / "queryset.parquet")
    corpus_df, _ = load_and_cache()
    corpus_lookup = dict(zip(corpus_df["id"], corpus_df["passage"]))
    for mode in ["lexical", "dense", "hybrid", "hybrid_expansion"]:
        score_config(mode, queryset, corpus_lookup)
