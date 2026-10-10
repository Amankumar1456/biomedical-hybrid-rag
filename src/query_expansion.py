"""
Query expansion: given a user question, generate a small set of
alternate phrasings (synonyms, rephrasing, related biomedical terms)
that get retrieved alongside the original query.

The original query is always retained in the returned list — expansion
adds to retrieval coverage, it never replaces the user's actual question.
"""
import json
import os

import anthropic

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXPANSION_PROMPT = """You expand biomedical search queries for a retrieval system.

Given the user's question, produce 2-3 alternate phrasings that would help \
retrieve relevant passages a literal keyword match might miss: use synonyms, \
expand abbreviations, or rephrase as a statement instead of a question. \
Stay strictly on the same topic — do not introduce new questions.

Respond with ONLY a JSON array of strings, no preamble, no markdown fences. \
Example: ["expansion one", "expansion two"]"""


def expand_query(query: str, model="claude-sonnet-4-6", max_expansions=3) -> list[str]:
    """Returns [original_query, expansion_1, expansion_2, ...]."""
    try:
        resp = _client.messages.create(
            model=model,
            max_tokens=300,
            system=EXPANSION_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text = resp.content[0].text.strip()
        expansions = json.loads(text)
        expansions = [e for e in expansions if isinstance(e, str)][:max_expansions]
    except Exception as e:
        # Query expansion is an enhancement, not a hard dependency —
        # retrieval still works on the original query alone if this fails.
        print(f"[query_expansion] fell back to original only: {e}")
        expansions = []

    return [query] + expansions


if __name__ == "__main__":
    q = "What is the role of BRCA1 in breast cancer?"
    print(expand_query(q))
