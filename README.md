# Biomedical Hybrid Search — rag-mini-bioasq

## Setup
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python src/data_loader.py      # caches the dataset to data/
python src/indexing.py         # builds BM25 + FAISS indexes to indexes/
python eval/build_queryset.py  # samples the fixed 100-query eval set
python eval/run_eval.py        # runs all configs, writes results/comparison_table.csv
python eval/run_llm_judge.py   # RAGAS scoring on top of each config's answers
uvicorn app:app --reload       # serves the UI at http://localhost:8000
```

## Design choices

- **Storage / vector DB**: FAISS `IndexFlatIP` (in-memory, flat, exact) — the
  corpus is small enough (~40k passages) that approximate indexing (IVF/HNSW)
  isn't needed; flat search keeps recall exact so it doesn't confound the
  retrieval-mode comparison itself.
- **Embedding model**: `pritamdeka/S-PubMedBert-MS-MARCO`, chosen for
  biomedical-domain pretraining over a general sentence-transformer.
- **Fusion method**: Reciprocal Rank Fusion (RRF, k=60) across BM25 and dense
  rankings, and across all query-expansion variants for the
  `hybrid_expansion` config. RRF was chosen over score-based fusion because
  BM25 scores and cosine similarities live on different, hard-to-normalize
  scales — RRF only needs rank position.
- **Final-answer model**: Claude, called with a system prompt that restricts
  it to the retrieved passages only and requires inline `[pid:...]`
  citations; the model must emit `INSUFFICIENT_EVIDENCE` rather than
  answering outside the provided context.
- **Citation validation**: enforced in code in `src/answer.py`
  (`generate_answer`) — every `[pid:...]` the model emits is checked against
  the set of passage IDs actually retrieved for that query; any citation to
  an ID outside that set is flagged as `invalid_citations` in the response
  and in the eval logs, rather than trusted at face value.

## Results

_Fill in after running `eval/run_eval.py` and `eval/run_llm_judge.py`:_

| Config | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Latency (s) |
|---|---|---|---|---|---|
| Lexical | | | | | |
| Dense | | | | | |
| Hybrid | | | | | |
| Hybrid + Expansion | | | | | |
| Strongest variant | | | | | |

**Winner:** _TBD — state which config won, why (which failure mode of the
other configs it fixes), and the latency/cost trade-off it costs you._

## Known limitations
- Query-type classification for the 100-query set is a keyword heuristic,
  not ground-truth labeling.
- `INSUFFICIENT_THRESHOLD` in `app.py` is a placeholder — tune it against
  observed RRF score distributions once you have real eval numbers.
