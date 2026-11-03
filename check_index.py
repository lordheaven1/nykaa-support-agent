"""
check_index.py
Runs a sample query check against both ChromaDB collections (kb_fixed and kb_sentence)
to verify semantic retrieval and inspect top-3 similarity scores.
"""

import logging
import os
import warnings

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QUERY = "How long do I have to return a lipstick?"


def query_collection(collection, query_text: str, model: SentenceTransformer, top_k: int = 3):
    query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def display_results(results, title: str):
    print(f"\n--- {title} (Top 3) ---")
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(docs)):
        dist = distances[i]
        sim = 1.0 - dist
        doc_id = metas[i]["doc_id"]
        chunk_idx = metas[i]["chunk_index"]
        text_preview = docs[i].replace("\n", " ")
        print(f"Rank {i+1} [Sim: {sim:.4f} | Dist: {dist:.4f}] Doc: {doc_id} (Chunk {chunk_idx})")
        print(f"  Text: \"{text_preview}\"")


def main():
    print("=" * 65)
    print("SAMPLE RETRIEVAL VERIFICATION")
    print(f"Query: \"{QUERY}\"")
    print("=" * 65)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    col_fixed = client.get_collection("kb_fixed")
    res_fixed = query_collection(col_fixed, QUERY, model, top_k=3)
    display_results(res_fixed, "Collection: kb_fixed (Fixed-Size Chunks)")

    col_sentence = client.get_collection("kb_sentence")
    res_sentence = query_collection(col_sentence, QUERY, model, top_k=3)
    display_results(res_sentence, "Collection: kb_sentence (Sentence-Based Chunks)")
    print("=" * 65)


if __name__ == "__main__":
    main()
