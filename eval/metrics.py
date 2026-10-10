"""
Traditional retrieval metrics computed against relevant_passage_ids.
Kept dependency-free (plain Python/NumPy) since each metric is a
handful of lines — not worth pulling in a package for.
"""
import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(retrieved_ids[:k], start=1)
        if pid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all(retrieved_ids: list[str], relevant_ids: set[str]) -> dict:
    return {
        "recall@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr@10": mrr_at_k(retrieved_ids, relevant_ids, 10),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }
