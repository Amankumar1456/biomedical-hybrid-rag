"""
Final-answer generation: takes the user's question + the retrieved
passages (already ranked/fused) and produces a grounded answer with
inline [passage_id] citations, or an explicit insufficient-evidence
response if the passages don't actually cover the question.
"""
import os
import re

from openai import OpenAI

from retrieval import Hit

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={"HTTP-Referer": "http://localhost:8000", "X-Title": "bioasq-rag"},
)

ANSWER_SYSTEM_PROMPT = """You are a biomedical question-answering assistant.

Answer the user's question using ONLY the numbered passages provided below. \
Every factual claim must carry an inline citation to the passage ID it came \
from, formatted like [pid:123]. Do not use outside knowledge.

If the passages do not contain enough information to answer the question, \
respond with exactly: INSUFFICIENT_EVIDENCE followed by a one-sentence \
explanation of what's missing. Do not guess or fill gaps."""

CITATION_RE = re.compile(r"\[pid:([^\]]+)\]")


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[pid:{h.passage_id}] {h.text}" for h in hits)


def generate_answer(query: str, hits: list[Hit], model="claude-sonnet-4-6") -> dict:
    if not hits:
        return {
            "answer": "INSUFFICIENT_EVIDENCE: no passages were retrieved for this query.",
            "citations": [],
            "insufficient_evidence": True,
        }

    context = build_context(hits)
    resp = _client.chat.completions.create(
        model="anthropic/claude-3.5-haiku",
        max_tokens=600,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {query}"},
        ],
    )
    text = resp.choices[0].message.content.strip()

    insufficient = text.startswith("INSUFFICIENT_EVIDENCE")
    cited_ids = set(CITATION_RE.findall(text))
    retrieved_ids = {h.passage_id for h in hits}

    # Citation validity check (Step 4's "check citation validity in code"):
    # flag any cited ID that wasn't actually in the retrieved context —
    # that would mean the model cited a passage it wasn't given.
    invalid_citations = cited_ids - retrieved_ids

    return {
        "answer": text,
        "citations": sorted(cited_ids),
        "invalid_citations": sorted(invalid_citations),
        "insufficient_evidence": insufficient,
    }
