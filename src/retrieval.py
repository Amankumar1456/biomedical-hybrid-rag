"""
Three retrieval modes over the same indexed passages, plus RRF fusion
for hybrid search. Every function returns a list of
{"passage_id": ..., "score": ..., "text": ...} so the caller always has
IDs and scores available for logging (Step 1 requirement) and for the
UI's evidence panel (Step 2 requirement).
"""
from dataclasses import dataclass

import numpy as np

from indexing import tokenize
from query_expansion import expand_query


@dataclass
class Hit:
    passage_id: str
    score: float
    text: str


class Retriever:
    def __init__(self, bm25, dense_index, embed_model, id_map, corpus_texts):
        self.bm25 = bm25
        self.dense_index = dense_index
        self.embed_model = embed_model
        self.id_map = id_map  # positional idx -> passage_id
        self.corpus_texts = corpus_texts  # positional idx -> passage text

    # ---- single-query, single-mode retrieval ----

    def lexical(self, query: str, top_k=10) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(self.id_map[i], float(scores[i]), self.corpus_texts[i])
            for i in top_idx
            if scores[i] > 0
        ]

    def dense(self, query: str, top_k=10) -> list[Hit]:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.dense_index.search(q_emb, top_k)
        return [
            Hit(self.id_map[i], float(s), self.corpus_texts[i])
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]

    # ---- fusion across modes and across expanded query variants ----

    @staticmethod
    def rrf_fuse(ranked_lists: list[list[Hit]], k=60, top_k=10) -> list[Hit]:
        """
        Reciprocal Rank Fusion: score(doc) = sum over lists of 1/(k + rank).
        Rank-based, so it needs no score normalization across BM25 and
        cosine similarity, which live on incomparable scales.
        """
        rrf_scores: dict[str, float] = {}
        doc_text: dict[str, str] = {}
        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked):
                rrf_scores[hit.passage_id] = rrf_scores.get(hit.passage_id, 0.0) + 1.0 / (k + rank + 1)
                doc_text[hit.passage_id] = hit.text

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Hit(pid, score, doc_text[pid]) for pid, score in fused]

    def hybrid(self, query: str, top_k=10, per_mode_k=20) -> list[Hit]:
        return self.rrf_fuse(
            [self.lexical(query, per_mode_k), self.dense(query, per_mode_k)], top_k=top_k
        )

    def hybrid_with_expansion(self, query: str, top_k=10, per_mode_k=20) -> list[Hit]:
        variants = expand_query(query)  # [original, expansion_1, ...]
        ranked_lists = []
        for variant in variants:
            ranked_lists.append(self.lexical(variant, per_mode_k))
            ranked_lists.append(self.dense(variant, per_mode_k))
        return self.rrf_fuse(ranked_lists, top_k=top_k)

    # ---- unified entry point used by both the app and the eval harness ----

    def search(self, query: str, mode: str, top_k=10) -> list[Hit]:
        if mode == "lexical":
            return self.lexical(query, top_k)
        if mode == "dense":
            return self.dense(query, top_k)
        if mode == "hybrid":
            return self.hybrid(query, top_k)
        if mode == "hybrid_expansion":
            return self.hybrid_with_expansion(query, top_k)
        raise ValueError(f"unknown retrieval mode: {mode}")
