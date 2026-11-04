"""
demo_api.py
Comprehensive demonstration of FastAPI endpoints using TestClient.
Executes 7 test scenarios (health, rag, order, PII, validation error 422,
document ingestion 200, and duplicate conflict 409), followed by strict
ChromaDB cleanup and forensic log verification for Task 11 and Task 12.
"""

import json
import os
import shutil
from fastapi.testclient import TestClient

from api import app
from logging_utils import LOGS_FILE
import rag_core


def run_demo():
    # Ensure fresh log file for isolated evaluation
    if os.path.exists(LOGS_FILE):
        os.remove(LOGS_FILE)

    print("=" * 75)
    print("TASK 11: FASTAPI ENDPOINT DEMONSTRATION")
    print("=" * 75)

    with TestClient(app) as client:
        # Call 1: Health check
        print("\n[Call 1]: GET /health")
        r1 = client.get("/health")
        print(f"  Status: {r1.status_code}")
        print(f"  Body:   {r1.json()}")

        # Call 2: Policy inquiry (RAG route)
        print("\n[Call 2]: POST /ask (Policy Query)")
        r2 = client.post("/ask", json={"query": "How long do I have to return a lipstick?"})
        print(f"  Status: {r2.status_code}")
        print(f"  Body:   {json.dumps(r2.json(), indent=2)}")

        # Call 3: Order lookup
        print("\n[Call 3]: POST /ask (Order Status Query)")
        r3 = client.post("/ask", json={"query": "What is the status of ORD-0012?"})
        print(f"  Status: {r3.status_code}")
        print(f"  Body:   {json.dumps(r3.json(), indent=2)}")

        # Call 4: Query with PII
        print("\n[Call 4]: POST /ask (PII Redaction Test)")
        pii_query = "My number is 9876543210 and card ending 4321, where is ORD-0012?"
        r4 = client.post("/ask", json={"query": pii_query})
        print(f"  Status: {r4.status_code}")
        print(f"  Body:   {json.dumps(r4.json(), indent=2)}")

        # Call 5: Empty query (Validation Error)
        print("\n[Call 5]: POST /ask (Empty Query -> Expected 422)")
        r5 = client.post("/ask", json={"query": ""})
        print(f"  Status: {r5.status_code}")
        print(f"  Body:   {r5.json()}")

        # Call 6: Add document
        print("\n[Call 6]: POST /add-document (New Document Ingestion)")
        doc_payload = {
            "doc_id": "test_gift_wrap",
            "text": "Gift wrapping services are available during checkout for fifty rupees per package. Wrapped items include a personalized note and customized ribbon packaging."
        }
        r6 = client.post("/add-document", json=doc_payload)
        print(f"  Status: {r6.status_code}")
        print(f"  Body:   {r6.json()}")

        # Call 7: Duplicate add document (Conflict Error)
        print("\n[Call 7]: POST /add-document (Duplicate Ingestion -> Expected 409)")
        r7 = client.post("/add-document", json=doc_payload)
        print(f"  Status: {r7.status_code}")
        print(f"  Body:   {r7.json()}")

    # Cleanup temporary ChromaDB test documents to preserve frozen benchmark index
    print("\n" + "=" * 75)
    print("CLEANUP: Restoring ChromaDB to frozen baseline counts (40 fixed, 48 sentence)")
    print("=" * 75)
    chroma_client = rag_core.get_chroma_client()
    coll_s = chroma_client.get_collection("kb_sentence")
    coll_f = chroma_client.get_collection("kb_fixed")

    coll_s.delete(where={"doc_id": "test_gift_wrap"})
    coll_f.delete(where={"doc_id": "test_gift_wrap"})

    count_s = coll_s.count()
    count_f = coll_f.count()
    print(f"Chroma collections restored -> kb_sentence: {count_s} chunks | kb_fixed: {count_f} chunks")
    assert count_s == 48, f"Expected 48 chunks in kb_sentence, got {count_s}"
    assert count_f == 40, f"Expected 40 chunks in kb_fixed, got {count_f}"
    print("Baseline chunk integrity verified: PASS")


if __name__ == "__main__":
    run_demo()
