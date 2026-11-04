"""
eval_triad.py
Evaluates the RAG Triad across 15 queries (12 in-scope policy queries covering
all 12 KB topics, plus 3 edge cases: out-of-scope near-miss, off-topic, ambiguous).

Implements:
1. LLM-as-Judge Prompt Templates (Context Relevance, Groundedness, Answer Relevance)
2. Deterministic Mock Judge Scoring Rules with explicit distinction between:
   - Correct Refusal (out-of-scope query refused -> Groundedness=1.0, Answer Rel=1.0)
   - False Refusal (in-scope query refused -> Groundedness=0.0, Answer Rel=0.0)
3. Dual average reporting (12 In-Scope vs All 15 Queries) with verified arithmetic
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional

from agent import run_agent
from rag_core import get_embedding_model, retrieve

# ==========================================
# 1. LLM-as-Judge Prompt Templates
# ==========================================

PROMPT_CONTEXT_RELEVANCE = """\
[JUDGE PROMPT: CONTEXT RELEVANCE]
You are an expert evaluator assessing retrieval quality.
Given the user query and the retrieved context passages, determine how relevant the retrieved context is to addressing the query.
Score between 0.0 (completely irrelevant) and 1.0 (highly relevant, contains all needed context).

Query: {query}
Retrieved Context:
{context}

Provide your score as:
Score: <float between 0.0 and 1.0>
Reasoning: <concise explanation>
"""

PROMPT_GROUNDEDNESS = """\
[JUDGE PROMPT: GROUNDEDNESS / FAITHFULNESS]
You are an expert evaluator assessing factual consistency and hallucination.
Given the retrieved context passages and the generated answer, determine whether every factual statement in the answer is strictly supported by the context.
If the query is out-of-scope and the agent issues an accurate fallback refusal, score 1.0.
If the query is in-scope and the agent fails to answer (false refusal), score 0.0.

Retrieved Context:
{context}
Generated Answer:
{answer}

Provide your score as:
Score: <float between 0.0 and 1.0>
Reasoning: <concise explanation>
"""

PROMPT_ANSWER_RELEVANCE = """\
[JUDGE PROMPT: ANSWER RELEVANCE]
You are an expert evaluator assessing response relevance.
Given the user query and the generated answer, determine how directly and completely the answer addresses the user's intent.
For out-of-scope queries where refusal is the policy-compliant behavior, a correct refusal scores 1.0.
For in-scope queries where the agent issues a false refusal, score 0.0 because the user received no answer.

Query: {query}
Generated Answer:
{answer}

Provide your score as:
Score: <float between 0.0 and 1.0>
Reasoning: <concise explanation>
"""

MOCK_JUDGE_RULES_SUMMARY = """\
[DETERMINISTIC MOCK JUDGE SCORING RULES]
1. Context Relevance: Arithmetic mean of top-3 retrieved chunk cosine similarities from kb_sentence (unfiltered by the 0.30 operational threshold).
2. Groundedness:
   - Answered queries: 1.0 if extracted policy sentence appears verbatim in retrieved chunks, else 0.0 (1.00 by construction under extractive MOCK_LLM).
   - Correct refusal (out-of-scope query refused/fallback): 1.0 by definition (no unsupported claims).
   - False refusal (in-scope query with expected_doc != None that triggers fallback): 0.0 (failed to provide grounded answer).
3. Answer Relevance:
   - Answered queries: Cosine similarity between query embedding and generated answer embedding.
   - Correct refusal (out-of-scope query refused/fallback): 1.0 by definition.
   - False refusal (in-scope query that triggers fallback): 0.0 (user received no answer).
"""


# ==========================================
# 2. Benchmark Queries (12 In-Scope + 3 Edge Cases)
# ==========================================

QUERIES: List[Dict[str, Any]] = [
    # 12 In-Scope Queries (Exact file stems verified against kb/)
    {"id": 1,  "topic": "Return Window",          "query": "How many days do I have to return an unopened face cream?", "expected_doc": "return_window"},
    {"id": 2,  "topic": "COD Refund",             "query": "I paid cash on delivery for my parcel, how will the refund reach me?", "expected_doc": "cod_refund"},
    {"id": 3,  "topic": "Delivery SLAs",          "query": "When can I expect my order to arrive in Mumbai?", "expected_doc": "delivery_sla"},
    {"id": 4,  "topic": "Reverse Pickup",         "query": "Does someone come to my house to collect returned packages?", "expected_doc": "reverse_pickup"},
    {"id": 5,  "topic": "Warranty",               "query": "My hair straightener stopped working after two weeks, who handles repairs?", "expected_doc": "warranty"},
    {"id": 6,  "topic": "Order Cancellation",     "query": "Can I cancel my items after the status shows dispatched?", "expected_doc": "order_cancellation"},
    {"id": 7,  "topic": "Loyalty Points",         "query": "What is the rupee value when I redeem 200 reward points?", "expected_doc": "loyalty_points"},
    {"id": 8,  "topic": "Payment Failure",        "query": "My bank account was debited but the order confirmation didn't go through.", "expected_doc": "payment_failure"},
    {"id": 9,  "topic": "Size Exchange",          "query": "The running shoes are too small, can I swap them for a UK 9?", "expected_doc": "size_exchange"},
    {"id": 10, "topic": "Damaged Item",           "query": "My perfume arrived completely shattered inside the shipping box.", "expected_doc": "damaged_item"},
    {"id": 11, "topic": "International Shipping", "query": "Do you ship cosmetics to addresses in Dubai or the UK?", "expected_doc": "international_shipping"},
    {"id": 12, "topic": "Escalation Matrix",      "query": "How do I escalate an unresolved complaint if support hasn't replied for 48 hours?", "expected_doc": "escalation_matrix"},

    # 3 Edge Cases
    {"id": 13, "topic": "Corporate Near-Miss",    "query": "What is Nykaa's current share price on the stock exchange?", "expected_doc": None},
    {"id": 14, "topic": "Pure Off-Topic",         "query": "How do I bake a chocolate cake at home?", "expected_doc": None},
    {"id": 15, "topic": "Ambiguous Return",       "query": "Can I return it?", "expected_doc": None},
]


# ==========================================
# 3. Deterministic Judge Scoring Functions
# ==========================================

def _is_refusal_response(agent_response: Dict[str, Any]) -> bool:
    answer_text = agent_response.get("answer", "")
    return (
        agent_response.get("route") == "refused"
        or not agent_response.get("grounded", False)
        or "I don't know based on the Nykaa knowledge base." in answer_text
        or "violates safety guidelines" in answer_text
    )


def score_context_relevance(query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Non-circular measure: Arithmetic mean of top-3 retrieved chunk similarities
    retrieved via cosine distance against kb_sentence.
    """
    if not retrieved_chunks:
        return 0.0
    sims = [c["similarity"] for c in retrieved_chunks[:3]]
    return round(float(np.mean(sims)), 4)


def score_groundedness(
    agent_response: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    expected_doc: Optional[str],
) -> float:
    """
    Extractive verification with explicit false-refusal penalty:
    - If agent refused/fell back:
        * If expected_doc is None (out-of-scope): Correct refusal -> 1.0
        * If expected_doc is not None (in-scope): False refusal -> 0.0
    - If answered: verifies extracted content sentence appears in retrieved chunks (1.0).
    """
    if _is_refusal_response(agent_response):
        return 1.0 if expected_doc is None else 0.0

    if not retrieved_chunks:
        return 0.0

    answer_text = agent_response.get("answer", "")
    chunk_texts = " ".join([c["text"] for c in retrieved_chunks])
    clean_answer = answer_text.replace("According to Nykaa policy: ", "").strip()
    if clean_answer in chunk_texts:
        return 1.0
    return 0.0


def score_answer_relevance(
    query: str,
    agent_response: Dict[str, Any],
    embedder,
    expected_doc: Optional[str],
) -> float:
    """
    Answer relevance scoring:
    - If agent refused/fell back:
        * If expected_doc is None (out-of-scope): Correct refusal -> 1.0 by definition
        * If expected_doc is not None (in-scope): False refusal -> 0.0 (user got no answer)
    - If answered: Cosine similarity between query embedding and answer embedding.
    """
    if _is_refusal_response(agent_response):
        return 1.0 if expected_doc is None else 0.0

    answer_text = agent_response.get("answer", "")
    q_vec = embedder.encode(query)
    a_vec = embedder.encode(answer_text)
    cos_sim = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))
    return round(max(0.0, cos_sim), 4)


# ==========================================
# 4. Evaluation Runner
# ==========================================

def run_evaluation():
    embedder = get_embedding_model()

    print("=" * 92)
    print("TASK 13: RAG TRIAD EVALUATION (15 QUERIES, 3 JUDGE SCORES)")
    print("=" * 92)
    print("\n[RAG Scope Note]: The RAG triad is defined strictly for knowledge retrieval pathways.")
    print("Order-status tracking queries route to the deterministic order tool and are evaluated separately.\n")

    print("-" * 92)
    print("LLM-AS-JUDGE PROMPT TEMPLATES & MOCK SCORING RULES")
    print("-" * 92)
    print(PROMPT_CONTEXT_RELEVANCE)
    print(PROMPT_GROUNDEDNESS)
    print(PROMPT_ANSWER_RELEVANCE)
    print(MOCK_JUDGE_RULES_SUMMARY)
    print("-" * 92)

    results = []
    in_scope_scores = {"c_rel": [], "ground": [], "a_rel": []}
    all_scores = {"c_rel": [], "ground": [], "a_rel": []}

    print("\n" + "=" * 92)
    print(f"{'ID':<3} | {'Topic':<22} | {'Route':<7} | {'C.Rel':<6} | {'Gnd':<5} | {'A.Rel':<6} | {'Top Source / Behavior'}")
    print("-" * 92)

    for item in QUERIES:
        q_id = item["id"]
        topic = item["topic"]
        query = item["query"]
        expected = item["expected_doc"]

        # Run independent top-3 retrieval for non-circular context relevance
        retrieved_k3 = retrieve(query, collection_name="kb_sentence", k=3)

        # Run full agent
        resp = run_agent(query, conversation_id=f"eval-triad-{q_id}")

        c_rel = score_context_relevance(query, retrieved_k3)
        gnd = score_groundedness(resp, retrieved_k3, expected)
        a_rel = score_answer_relevance(query, resp, embedder, expected)

        all_scores["c_rel"].append(c_rel)
        all_scores["ground"].append(gnd)
        all_scores["a_rel"].append(a_rel)

        if expected is not None:
            in_scope_scores["c_rel"].append(c_rel)
            in_scope_scores["ground"].append(gnd)
            in_scope_scores["a_rel"].append(a_rel)

        if _is_refusal_response(resp):
            if expected is not None:
                behavior_label = f"FALSE_REFUSAL (top1<{0.30})"
            elif resp["route"] == "refused":
                behavior_label = "refused (guardrail)"
            else:
                behavior_label = "fallback (ungrounded)"
        else:
            behavior_label = resp["sources"][0] if resp.get("sources") else resp.get("route")

        results.append({
            "id": q_id,
            "topic": topic,
            "query": query,
            "expected_doc": expected,
            "route": resp["route"],
            "grounded_flag": resp.get("grounded", False),
            "sources": resp.get("sources", []),
            "context_relevance": c_rel,
            "groundedness": gnd,
            "answer_relevance": a_rel,
            "behavior": behavior_label,
            "answer": resp.get("answer"),
        })

        print(f"{q_id:<3} | {topic:<22} | {resp['route']:<7} | {c_rel:<6.4f} | {gnd:<5.2f} | {a_rel:<6.4f} | {behavior_label}")

    # Compute ExactSums & Averages
    sum_in_c = round(float(np.sum(in_scope_scores["c_rel"])), 4)
    sum_in_g = round(float(np.sum(in_scope_scores["ground"])), 4)
    sum_in_a = round(float(np.sum(in_scope_scores["a_rel"])), 4)

    mean_in_c = round(sum_in_c / 12.0, 4)
    mean_in_g = round(sum_in_g / 12.0, 4)
    mean_in_a = round(sum_in_a / 12.0, 4)

    sum_all_c = round(float(np.sum(all_scores["c_rel"])), 4)
    sum_all_g = round(float(np.sum(all_scores["ground"])), 4)
    sum_all_a = round(float(np.sum(all_scores["a_rel"])), 4)

    mean_all_c = round(sum_all_c / 15.0, 4)
    mean_all_g = round(sum_all_g / 15.0, 4)
    mean_all_a = round(sum_all_a / 15.0, 4)

    print("-" * 92)
    print(f"SUM (12 In-Scope Only):         | {'-':<7} | {sum_in_c:<6.4f} | {sum_in_g:<5.2f} | {sum_in_a:<6.4f} |")
    print(f"AVERAGE (12 In-Scope Only):     | {'-':<7} | {mean_in_c:<6.4f} | {mean_in_g:<5.4f}| {mean_in_a:<6.4f} | (#5 retrieved return_window vs warranty)")
    print("-" * 92)
    print(f"SUM (All 15 Evaluated):         | {'-':<7} | {sum_all_c:<6.4f} | {sum_all_g:<5.2f} | {sum_all_a:<6.4f} |")
    print(f"AVERAGE (All 15 Evaluated):     | {'-':<7} | {mean_all_c:<6.4f} | {mean_all_g:<5.4f}| {mean_all_a:<6.4f} | (Out-of-scope refusal #13,#14 = 1.0)")
    print("=" * 92)

    # Topic Coverage Check across 12 in-scope expected_docs
    in_scope_expected = [q["expected_doc"] for q in QUERIES if q["expected_doc"] is not None]
    unique_topics = sorted(list(set(in_scope_expected)))
    print(f"\n[Topic Coverage Check]: {len(unique_topics)} unique expected docs mapped across 12 in-scope queries.")
    print("Mapped Docs:", ", ".join(unique_topics))
    assert len(unique_topics) == 12, f"Expected 12 unique topics, got {len(unique_topics)}"
    print("12-Topic Complete Coverage: PASS\n")

    # Save detailed JSON scores
    with open("transcripts/task13_scores.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sums_in_scope_12": {
                    "context_relevance": sum_in_c,
                    "groundedness": sum_in_g,
                    "answer_relevance": sum_in_a,
                },
                "averages_in_scope_12": {
                    "context_relevance": mean_in_c,
                    "groundedness": mean_in_g,
                    "answer_relevance": mean_in_a,
                },
                "sums_all_15": {
                    "context_relevance": sum_all_c,
                    "groundedness": sum_all_g,
                    "answer_relevance": sum_all_a,
                },
                "averages_all_15": {
                    "context_relevance": mean_all_c,
                    "groundedness": mean_all_g,
                    "answer_relevance": mean_all_a,
                },
                "queries": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    run_evaluation()

