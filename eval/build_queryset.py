"""
Builds the fixed 100-query evaluation set (Step 3), saved once so every
retrieval config is evaluated against the exact same queries.

BioASQ's own question typing (yesno/factoid/list/summary) isn't present
in this mini dataset's schema (question, answer, id, relevant_passage_ids
only — confirmed against the dataset card, no type column). So typing is
reconstructed here, in two tiers of confidence:

  1. yes_no is grounded directly in the ANSWER field (answer.strip().lower()
     in {"yes","no"}), not guessed from question phrasing — this is exact,
     not a heuristic, since BioASQ yes/no answers are literally "yes"/"no".
  2. definition / list / treatment / mechanism / effect are keyword
     heuristics over the QUESTION text, applied in a fixed priority order
     so a question matching several patterns doesn't silently fall to
     whichever dict key happens to iterate first.

This is "good enough for stratified sampling," per the assignment's own
"where possible" wording — not a claim of ground-truth question typing.
Diagnostics are printed (and written to eval/queryset_diagnostics.csv) so
you can eyeball misclassifications before treating the split as final.
"""
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import load_and_cache

OUT_PATH = Path(__file__).resolve().parent / "queryset.parquet"
DIAG_PATH = Path(__file__).resolve().parent / "queryset_diagnostics.csv"

# Order matters: first match wins. More specific/rare patterns go first
# so they aren't swallowed by broader ones (e.g. a "what genes cause X"
# question should land in "list", not get grabbed by "definition"'s
# leading "what" or by "effect"'s "cause").
# "definition" is deliberately LAST: "What is the treatment for X?" and
# "What is the mechanism of Y?" both start with "what is", so checking
# definition before the more specific topical categories would swallow
# them. Definition should only catch what's left over — genuinely bare
# "what is <entity>" questions with no treatment/mechanism/effect/list
# content, e.g. "What is BRCA1?".
PATTERN_PRIORITY = [
    ("list", re.compile(
        r"^(list|name)\b"
        r"|\b(which|what)\b.*\b(genes?|drugs?|symptoms?|types?|proteins?|"
        r"mutations?|biomarkers?|enzymes?|receptors?|syndromes?)s?\b",
        re.I,
    )),
    ("treatment", re.compile(r"\b(treat(ment|s)?|therap(y|ies)|drug|medication|dose|dosage|regimen)\b", re.I)),
    ("mechanism", re.compile(r"\b(mechanism|pathway|how does|how is|bind(s|ing)?|interact(s|ion)?|regulat(e|es|ion))\b", re.I)),
    ("effect", re.compile(r"\b(effect|cause[sd]?|associat(e|ed|ion)|risk|impact|role of|contribut(e|es|ion))\b", re.I)),
    ("definition", re.compile(r"^what\s+(is|are|was|were)\b|^define\b", re.I)),
]

REQUIRED_BUCKETS = ["yes_no", "definition", "list", "treatment", "mechanism", "effect"]


def classify(row) -> str:
    answer = str(row["answer"]).strip().lower()
    if answer in {"yes", "no"}:
        return "yes_no"

    question = row["question"]
    for label, pat in PATTERN_PRIORITY:
        if pat.search(question):
            return label
    return "other"


def build(n=100, seed=42):
    _, qa_df = load_and_cache()
    qa_df = qa_df.copy()
    qa_df["q_type"] = qa_df.apply(classify, axis=1)

    full_counts = qa_df["q_type"].value_counts()
    print("=== Full dataset type distribution ===")
    print(full_counts)
    thin = [b for b in REQUIRED_BUCKETS if full_counts.get(b, 0) < 5]
    if thin:
        print(f"WARNING: thin buckets (<5 examples in full dataset): {thin}")

    # Stratified sample: draw evenly across the six required buckets
    # first, then top up from whatever's left (including "other") to
    # reach exactly n, so every run has the same fixed size regardless
    # of how buckets are sized.
    per_type = max(1, n // len(REQUIRED_BUCKETS))
    parts = []
    for label in REQUIRED_BUCKETS:
        bucket = qa_df[qa_df["q_type"] == label]
        parts.append(bucket.sample(min(per_type, len(bucket)), random_state=seed))

    sampled = pd.concat(parts)
    if len(sampled) < n:
        remaining = qa_df.drop(sampled.index)
        topup = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, topup])
    elif len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed)

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle order
    sampled.to_parquet(OUT_PATH)

    print("\n=== Sampled 100-query set distribution ===")
    print(sampled["q_type"].value_counts())
    print(f"Saved {len(sampled)} queries to {OUT_PATH}")

    # Diagnostics for manual spot-checking of the heuristic before you
    # treat the split as final — required by "where possible" wording,
    # not a guarantee, so check this before relying on it.
    qa_df[["id", "question", "answer", "q_type"]].to_csv(DIAG_PATH, index=False)
    print(f"Full-dataset classification written to {DIAG_PATH} for review")

    other_sample = qa_df[qa_df["q_type"] == "other"]["question"].head(5).tolist()
    if other_sample:
        print("\nSample of unclassified ('other') questions — check these:")
        for q in other_sample:
            print(f"  - {q}")

    return sampled


if __name__ == "__main__":
    build()
