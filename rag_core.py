"""
rag_core.py
Central retrieval helper and mock environment configuration for Day 3, 4, and 5.
Loads SentenceTransformer and ChromaDB persistently as singletons.
"""

import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
for logger_name in ["transformers", "sentence_transformers", "huggingface_hub", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1") == "1"

_MODEL: Optional[SentenceTransformer] = None
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None


def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def get_chroma_client() -> chromadb.PersistentClient:
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
    return _CHROMA_CLIENT


def retrieve(query: str, collection_name: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves top-k chunks from specified ChromaDB collection.
    Converts cosine distance to cosine similarity via (1.0 - distance).
    Returns list of dicts: {text, doc_id, chunk_index, similarity}.
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(collection_name)

    query_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if not results or not results["documents"] or not results["documents"][0]:
        return retrieved

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        sim = 1.0 - float(dist)
        retrieved.append(
            {
                "text": doc_text,
                "doc_id": meta.get("doc_id", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "similarity": sim,
            }
        )
    return retrieved


if __name__ == "__main__":
    print(f"rag_core initialized. MOCK_LLM={MOCK_LLM}")
    sample = retrieve("How long do I have to return a lipstick?", "kb_sentence", k=1)
    if sample:
        top = sample[0]
        print(f"Test query top match: Doc '{top['doc_id']}' (Sim: {top['similarity']:.4f})")
