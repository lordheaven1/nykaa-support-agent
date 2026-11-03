"""
chunking.py
Implements fixed-size overlapping and sentence-based chunking strategies
for the Nykaa customer support knowledge base.
"""

import re
from typing import List


def fixed_size_chunks(
    text: str, chunk_size: int = 200, overlap: int = 40
) -> List[str]:
    """
    Slices text into fixed-size character windows with a defined character overlap.
    Safeguards against infinite loops by asserting overlap < chunk_size.
    """
    assert (
        overlap < chunk_size
    ), f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks


def sentence_chunks(text: str) -> List[str]:
    """
    Splits text on sentence boundaries using regex lookbehind for punctuation [.!?].
    Produces semantically complete, standalone sentence chunks.
    """
    text = text.strip()
    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = [s.strip() for s in raw_sentences if s.strip()]
    return chunks


if __name__ == "__main__":
    sample_path = "kb/return_window.txt"
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_doc = f.read()

    print("=" * 60)
    print("CHUNKING STRATEGY COMPARISON DEMO")
    print(f"Sample Document: {sample_path} ({len(sample_doc)} chars)")
    print("=" * 60)

    fixed = fixed_size_chunks(sample_doc, chunk_size=200, overlap=40)
    print(f"\n--- Strategy 1: Fixed-Size (size=200, overlap=40) -> {len(fixed)} chunks ---")
    for idx, c in enumerate(fixed):
        print(f"[{idx}] ({len(c)} chars): {c}")

    sent = sentence_chunks(sample_doc)
    print(f"\n--- Strategy 2: Sentence-Based -> {len(sent)} chunks ---")
    for idx, c in enumerate(sent):
        print(f"[{idx}] ({len(c)} chars): {c}")
    print("=" * 60)
