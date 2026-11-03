"""
demo_grounded.py
Demonstrates Task 4 grounded generation across 5 diverse in-scope customer queries
and 2 out-of-scope queries (1 unrelated trivia + 1 near-miss brand query).
"""

from generate import answer, SIMILARITY_THRESHOLD

DEMO_QUERIES = [
    # 5 In-scope queries (direct, indirect, and category testing)
    "How long do I have to return a lipstick?",
    "Can I return a mascara I opened?",
    "My shoes don't fit, can I swap them?",
    "When will my COD refund arrive?",
    "What happens if my payment fails?",
    # 2 Out-of-scope queries (1 pure unrelated + 1 near-miss corporate query)
    "How do I bake a chocolate cake?",
    "What is Nykaa's current share price on the stock exchange?",
]


def main():
    print("=" * 70)
    print("TASK 4: GROUNDED GENERATION DEMO (MOCK_LLM)")
    print(f"Collection: kb_sentence | Calibrated Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 70)

    for idx, q in enumerate(DEMO_QUERIES, 1):
        result = answer(q, collection_name="kb_sentence")
        print(f"\n[Query {idx}]: \"{q}\"")
        print(f"  - Grounded:       {result['grounded']}")
        print(f"  - Top Similarity: {result['top_similarity']}")
        print(f"  - Sources:        {result['sources']}")
        print(f"  - Answer:         {result['answer']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
