"""
index_kb.py
Loads KB documents, applies dual chunking strategies, embeds with all-MiniLM-L6-v2,
and indexes into two isolated ChromaDB collections (kb_fixed and kb_sentence).
"""

import glob
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

from typing import Any, Dict, List
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import fixed_size_chunks, sentence_chunks

KB_DIR = "kb"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


def load_kb_documents(kb_dir: str = KB_DIR) -> Dict[str, str]:
    """Loads all .txt policy files from kb/ mapped by doc_id stem."""
    pattern = os.path.join(kb_dir, "*.txt")
    files = sorted(glob.glob(pattern))
    assert len(files) == 12, f"Expected 12 KB files, found {len(files)}"

    docs: Dict[str, str] = {}
    for filepath in files:
        filename = os.path.basename(filepath)
        doc_id = os.path.splitext(filename)[0]
        with open(filepath, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read().strip()
    return docs


def populate_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    chunks_data: List[Dict[str, Any]],
    model: SentenceTransformer,
):
    """Safely recreates collection and ingests chunk records with dense embeddings."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks_data]
    ids = [c["id"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def main():
    print("=" * 65)
    print("KNOWLEDGE BASE INDEXING (TASK 3)")
    print("=" * 65)

    docs = load_kb_documents()
    print(f"Loaded {len(docs)} documents from {KB_DIR}/:")
    for doc_id, text in docs.items():
        print(f"  - {doc_id}.txt ({len(text)} characters)")

    fixed_chunks_data: List[Dict[str, Any]] = []
    sentence_chunks_data: List[Dict[str, Any]] = []

    for doc_id, text in docs.items():
        f_chunks = fixed_size_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for idx, chunk_text in enumerate(f_chunks):
            fixed_chunks_data.append(
                {
                    "id": f"{doc_id}::fixed::{idx}",
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "strategy": "fixed_size",
                    },
                }
            )

        s_chunks = sentence_chunks(text)
        for idx, chunk_text in enumerate(s_chunks):
            sentence_chunks_data.append(
                {
                    "id": f"{doc_id}::sentence::{idx}",
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "strategy": "sentence",
                    },
                }
            )

    print(f"\nLoading SentenceTransformer('{EMBEDDING_MODEL_NAME}')...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print("Indexing collection 'kb_fixed'...")
    col_fixed = populate_collection(client, "kb_fixed", fixed_chunks_data, model)

    print("Indexing collection 'kb_sentence'...")
    col_sentence = populate_collection(client, "kb_sentence", sentence_chunks_data, model)

    fixed_lens = [len(c["text"]) for c in fixed_chunks_data]
    sent_lens = [len(c["text"]) for c in sentence_chunks_data]

    avg_fixed = sum(fixed_lens) / len(fixed_lens) if fixed_lens else 0
    avg_sent = sum(sent_lens) / len(sent_lens) if sent_lens else 0

    print("\n" + "=" * 65)
    print("COLLECTION INDEXING SUMMARY")
    print("=" * 65)
    print(f"Collection 'kb_fixed':")
    print(f"  - Total Chunks: {col_fixed.count()}")
    print(f"  - Chunk Size / Overlap: {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    print(f"  - Average Chunk Length: {avg_fixed:.1f} characters")

    print(f"\nCollection 'kb_sentence':")
    print(f"  - Total Chunks: {col_sentence.count()}")
    print(f"  - Strategy: Standalone sentence split")
    print(f"  - Average Chunk Length: {avg_sent:.1f} characters")
    print("=" * 65)


if __name__ == "__main__":
    main()
