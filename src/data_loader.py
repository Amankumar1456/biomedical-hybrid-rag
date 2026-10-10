"""
Loads rag-datasets/rag-mini-bioasq from Hugging Face.

The dataset ships two configs:
  - "question-answer-passages": question, answer, relevant_passage_ids
  - "text-corpus": id, passage  (the full passage corpus to index)

IMPORTANT SCHEMA NOTE: relevant_passage_ids arrives as a STRING that
looks like a Python list literal (e.g. "[20598273, 6650562, 15829955]"),
not a native list/sequence column — confirmed against the HF dataset
viewer. It's parsed into an actual list[int] here, once, so every
downstream script (run_eval.py, build_queryset.py, run_llm_judge.py)
can just do set(row["relevant_passage_ids"]) without re-parsing.

Run this once; it caches parquet locally under data/ so every other
script reads from disk instead of re-hitting the hub.
"""
import ast
from pathlib import Path
import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.parquet"
QA_PATH = DATA_DIR / "qa.parquet"


def _parse_relevant_ids(val):
    """Handles both the raw stringified-list form and an already-parsed
    list/array, so this is safe to call whether or not the source
    schema changes underneath us."""
    if isinstance(val, str):
        parsed = ast.literal_eval(val)
        return [int(x) for x in parsed]
    return [int(x) for x in val]


def load_and_cache():
    if CORPUS_PATH.exists() and QA_PATH.exists():
        return pd.read_parquet(CORPUS_PATH), pd.read_parquet(QA_PATH)

    qa = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
    corpus = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")

    # Only a "test" split exists for this dataset (confirmed: 4,715 QA
    # rows / 40,181 corpus rows) — the fallback below just guards against
    # a schema change on the hub.
    qa_df = qa["test"].to_pandas() if "test" in qa else next(iter(qa.values())).to_pandas()
    corpus_df = corpus["test"].to_pandas() if "test" in corpus else next(iter(corpus.values())).to_pandas()

    qa_df["relevant_passage_ids"] = qa_df["relevant_passage_ids"].apply(_parse_relevant_ids)

    qa_df.to_parquet(QA_PATH)
    corpus_df.to_parquet(CORPUS_PATH)
    return corpus_df, qa_df


if __name__ == "__main__":
    corpus_df, qa_df = load_and_cache()
    print(f"corpus: {len(corpus_df)} passages, columns={list(corpus_df.columns)}")
    print(f"qa: {len(qa_df)} questions, columns={list(qa_df.columns)}")
    sample = qa_df.iloc[0].to_dict()
    print(sample)
    assert hasattr(sample["relevant_passage_ids"], "__len__"), "relevant_passage_ids is not a sequence"
    assert len(sample["relevant_passage_ids"]) > 0, "relevant_passage_ids is empty"
    print("relevant_passage_ids parsed correctly as list[int]:", sample["relevant_passage_ids"][:3])

