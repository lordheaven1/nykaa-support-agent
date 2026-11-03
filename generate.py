"""
generate.py
Grounded answer generation engine under MOCK_LLM.
Combines a calibrated similarity threshold (0.30 for kb_sentence) with a
non-support domain guardrail to refuse both unrelated and near-miss brand queries.
"""

from typing import Any, Dict, List
from rag_core import retrieve

SIMILARITY_THRESHOLD = 0.30

OUT_OF_SCOPE_KEYWORDS = {
    "share price",
    "stock exchange",
    "stock price",
    "market cap",
    "ipo",
    "ceo",
    "quarterly revenue",
    "nse",
    "bse",
}


def is_near_miss_out_of_scope(query: str) -> bool:
    """Detects corporate/investor queries that fall outside customer support scope."""
    q_lower = query.lower()
    return any(term in q_lower for term in OUT_OF_SCOPE_KEYWORDS)


def answer(query: str, collection_name: str = "kb_sentence", threshold: float = SIMILARITY_THRESHOLD) -> Dict[str, Any]:
    """
    Produces grounded answer for customer query based strictly on retrieved KB chunks.
    Falls back deterministically if the query hits the out-of-scope domain filter
    or if top-1 similarity is below the calibrated threshold.
    """
    retrieved = retrieve(query, collection_name=collection_name, k=3)
    top_sim = retrieved[0]["similarity"] if retrieved else 0.0

    if not retrieved or top_sim < threshold or is_near_miss_out_of_scope(query):
        return {
            "query": query,
            "answer": "I don't know based on the Nykaa knowledge base.",
            "sources": [],
            "top_similarity": round(top_sim, 4),
            "top_chunk_text": "",
            "grounded": False,
        }

    top_chunk = retrieved[0]
    sources = []
    for r in retrieved:
        if r["similarity"] >= threshold and r["doc_id"] not in sources:
            sources.append(r["doc_id"])

    clean_text = top_chunk["text"].strip()
    extractive_answer = f"According to Nykaa policy: {clean_text}"

    return {
        "query": query,
        "answer": extractive_answer,
        "sources": sources,
        "top_similarity": round(top_chunk["similarity"], 4),
        "top_chunk_text": clean_text,
        "grounded": True,
    }


if __name__ == "__main__":
    sample = answer("How long do I have to return a lipstick?", "kb_sentence")
    print(sample)
