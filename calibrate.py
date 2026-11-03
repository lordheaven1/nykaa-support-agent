"""
calibrate.py
Measures top-1 semantic similarity across in-scope and out-of-scope queries
for both kb_fixed and kb_sentence to establish an empirical "I don't know" threshold.
"""

from rag_core import retrieve

IN_SCOPE_QUERIES = [
    ("How many days do I have to return a face serum?", "return_window"),
    ("My shoes don't fit, can I swap them?", "size_exchange"),
    ("When will my COD refund arrive?", "cod_refund"),
    ("What happens if my payment fails?", "payment_failure"),
    ("How do I redeem loyalty points?", "loyalty_points"),
    ("My package arrived broken with cracked bottles, what do I do?", "damaged_item"),
]

OUT_OF_SCOPE_QUERIES = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the cricket world cup?",
    "What is Python programming language?",
    "What is Nykaa's current share price on the stock exchange?",
]


def run_calibration(collection_name: str):
    print(f"\n" + "=" * 70)
    print(f"CALIBRATION TABLE: Collection '{collection_name}'")
    print("=" * 70)
    print(f"{'Type':<12} | {'Top-1 Sim':<10} | {'Top Doc':<20} | Query")
    print("-" * 70)

    in_scope_sims = []
    for q, expected_doc in IN_SCOPE_QUERIES:
        res = retrieve(q, collection_name=collection_name, k=1)
        top = res[0]
        sim = top["similarity"]
        doc = top["doc_id"]
        in_scope_sims.append(sim)
        print(f"{'IN-SCOPE':<12} | {sim:<10.4f} | {doc:<20} | \"{q}\"")

    print("-" * 70)
    out_scope_sims = []
    for q in OUT_OF_SCOPE_QUERIES:
        res = retrieve(q, collection_name=collection_name, k=1)
        top = res[0]
        sim = top["similarity"]
        doc = top["doc_id"]
        out_scope_sims.append(sim)
        print(f"{'OUT-OF-SCOPE':<12} | {sim:<10.4f} | {doc:<20} | \"{q}\"")

    min_in = min(in_scope_sims)
    max_in = max(in_scope_sims)
    min_out = min(out_scope_sims)
    max_out = max(out_scope_sims)

    print("=" * 70)
    print(f"CLUSTER ANALYSIS ({collection_name}):")
    print(f"  - In-Scope Cluster:     [{min_in:.4f}, {max_in:.4f}] (Min: {min_in:.4f})")
    print(f"  - Out-of-Scope Cluster: [{min_out:.4f}, {max_out:.4f}] (Max: {max_out:.4f})")

    gap = min_in - max_out
    if gap > 0:
        threshold = (max_out + min_in) / 2.0
        print(f"  - Separation Gap:       {gap:.4f} (Clean separation)")
        print(f"  - Measured Threshold:   {threshold:.4f} (Midpoint: ({max_out:.4f} + {min_in:.4f}) / 2)")
    else:
        threshold = max_out + 0.01
        print(f"  - Cluster Overlap:      {-gap:.4f} (Near-miss out-of-scope query overlap)")
        print(f"  - Defensive Threshold:  {threshold:.4f} (Conservative rejection boundary)")

    return {
        "min_in": min_in,
        "max_out": max_out,
        "gap": gap,
        "threshold": threshold,
    }


def main():
    print("=" * 70)
    print("TASK 4: THRESHOLD CALIBRATION EXPERIMENT")
    print("=" * 70)

    stats_fixed = run_calibration("kb_fixed")
    stats_sentence = run_calibration("kb_sentence")

    print("\n" + "=" * 70)
    print("FINAL CALIBRATION SUMMARY & DECISION")
    print("=" * 70)
    print(f"kb_fixed:    Min In-Scope = {stats_fixed['min_in']:.4f} | Max Out-of-Scope = {stats_fixed['max_out']:.4f} | Threshold = {stats_fixed['threshold']:.4f}")
    print(f"kb_sentence: Min In-Scope = {stats_sentence['min_in']:.4f} | Max Out-of-Scope = {stats_sentence['max_out']:.4f} | Threshold = {stats_sentence['threshold']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
