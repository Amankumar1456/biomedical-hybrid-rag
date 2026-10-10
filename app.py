"""
FastAPI app: the "product" from Step 2. Search endpoint returns the
original query, its expansions, ranked evidence (id/score/snippet),
and a final grounded answer with citations or an insufficient-evidence
flag. Run with: uvicorn app:app --reload
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_loader import load_and_cache
from indexing import load_bm25, load_dense, load_id_map, EMBED_MODEL_NAME
from retrieval import Retriever
from query_expansion import expand_query
from answer import generate_answer
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Biomedical Hybrid Search")

# Loaded once at startup — indexes are read from disk, not rebuilt per request.
_corpus_df, _ = load_and_cache()
_retriever = Retriever(
    bm25=load_bm25(),
    dense_index=load_dense(),
    embed_model=SentenceTransformer(EMBED_MODEL_NAME),
    id_map=load_id_map(),
    corpus_texts=_corpus_df["passage"].tolist(),
)

INSUFFICIENT_THRESHOLD = 0.01  # tune against your eval results — RRF top score floor


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid_expansion"  # lexical | dense | hybrid | hybrid_expansion
    top_k: int = 10


@app.post("/search")
def search(req: SearchRequest):
    expansions = expand_query(req.query) if "expansion" in req.mode else [req.query]
    hits = _retriever.search(req.query, mode=req.mode, top_k=req.top_k)

    if not hits or hits[0].score < INSUFFICIENT_THRESHOLD:
        return {
            "query": req.query,
            "expansions": expansions,
            "evidence": [],
            "answer": "INSUFFICIENT_EVIDENCE: no sufficiently relevant passages were found.",
            "citations": [],
            "insufficient_evidence": True,
        }

    result = generate_answer(req.query, hits)
    return {
        "query": req.query,
        "expansions": expansions,
        "evidence": [
            {"passage_id": h.passage_id, "score": h.score, "text": h.text[:400]} for h in hits
        ],
        "answer": result["answer"],
        "citations": result["citations"],
        "invalid_citations": result.get("invalid_citations", []),
        "insufficient_evidence": result["insufficient_evidence"],
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
