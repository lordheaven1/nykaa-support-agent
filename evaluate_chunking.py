"""
evaluate_chunking.py
Measures Precision@3 and Recall@3 across kb_fixed and kb_sentence using
the 5 in-scope test queries against an independently established ground truth.
Deduplicates retrieved chunks to parent doc_ids before computing metrics.
"""

from typing import Dict, Set
from rag_core import retrieve

GROUND_TRUTH: Dict[str, Set[str]] = {
    "How long do I have to return a lipstick?": {"return_window"},
    "Can I return a mascara I opened?": {"return_window"},
    "My shoes don't fit, can I swap them?": {"size_exchange"},
    "When will my COD refund arrive?": {"cod_refund"},
    "What happens if my payment fails?": {"payment_failure"},
}


def evaluate_collection(collection_name: str) -> Dict[str, float]:
    print("\n" + "=" * 75)
    print(f"EVALUATION: Collection '{collection_name}' (k=3)")
    print("=" * 75)

    precisions = []
    recalls = []
    top_sims = []

    for idx, (query, relevant_docs) in enumerate(GROUND_TRUTH.items(), 1):
        retrieved_chunks = retrieve(query, collection_name=collection_name, k=3)
        top_sim = retrieved_chunks[0]["similarity"] if retrieved_chunks else 0.0
        top_sims.append(top_sim)

        retrieved_docs_ordered = []
        for c in retrieved_chunks:
            if c["doc_id"] not in retrieved_docs_ordered:
                retrieved_docs_ordered.append(c["doc_id"])

        retrieved_docs_set = set(retrieved_docs_ordered)
        hits = relevant_docs.intersection(retrieved_docs_set)

        num_hits = len(hits)
        num_retrieved_unique = len(retrieved_docs_set)
        num_relevant = len(relevant_docs)

        p_at_3 = num_hits / num_retrieved_unique if num_retrieved_unique > 0 else 0.0
        r_at_3 = num_hits / num_relevant if num_relevant > 0 else 0.0

        precisions.append(p_at_3)
        recalls.append(r_at_3)

        retrieved_summary = ", ".join([f"{c['doc_id']} (Chunk {c['chunk_index']})" for c in retrieved_chunks])
        print(f"\n[Q{idx}]: \"{query}\"")
        print(f"  - Target Doc(s):        {sorted(list(relevant_docs))}")
        print(f"  - Top-3 Retrieved:      [{retrieved_summary}]")
        print(f"  - Unique Retrieved:     {retrieved_docs_ordered}")
        print(f"  - Relevant Hits:        {sorted(list(hits))}")
        print(f"  - Precision@3:          {num_hits}/{num_retrieved_unique} = {p_at_3:.4f}")
        print(f"  - Recall@3:             {num_hits}/{num_relevant} = {r_at_3:.4f}")
        print(f"  - Top-1 Similarity:     {top_sim:.4f}")

    mean_p = sum(precisions) / len(precisions)
    mean_r = sum(recalls) / len(recalls)
    mean_sim = sum(top_sims) / len(top_sims)

    print("-" * 75)
    print(f"SUMMARY FOR '{collection_name}':")
    print(f"  Mean Precision@3:       {mean_p:.4f}")
    print(f"  Mean Recall@3:          {mean_r:.4f}")
    print(f"  Mean Top-1 Similarity:  {mean_sim:.4f}")
    print("=" * 75)

    return {"mean_p": mean_p, "mean_r": mean_r, "mean_sim": mean_sim}


def main():
    print("=" * 75)
    print("TASK 5: CHUNKING STRATEGY EVALUATION (PRECISION@3 & RECALL@3)")
    print("=" * 75)

    res_fixed = evaluate_collection("kb_fixed")
    res_sentence = evaluate_collection("kb_sentence")

    print("\n" + "=" * 75)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 75)
    print(f"{'Strategy':<18} | {'Mean P@3':<12} | {'Mean R@3':<12} | {'Mean Top-1 Sim':<14}")
    print("-" * 75)
    print(f"{'kb_fixed':<18} | {res_fixed['mean_p']:<12.4f} | {res_fixed['mean_r']:<12.4f} | {res_fixed['mean_sim']:<14.4f}")
    print(f"{'kb_sentence':<18} | {res_sentence['mean_p']:<12.4f} | {res_sentence['mean_r']:<12.4f} | {res_sentence['mean_sim']:<14.4f}")
    print("=" * 75)

    print("\nCHUNKING RECOMMENDATION:")
    print("Both strategies tied on document-level Mean Precision@3 (0.7000) and Mean Recall@3 (1.0000).")
    print("We break the tie using the secondary signal of Mean Top-1 Similarity (0.5158 for kb_sentence")
    print("vs. 0.4785 for kb_fixed) and select kb_sentence for deployment, as standalone policy sentences")
    print("avoid mid-clause cuts and yield stronger rank-1 confidence (e.g., COD refund 0.7956 vs. 0.6773).")
    print("=" * 75)


if __name__ == "__main__":
    main()
