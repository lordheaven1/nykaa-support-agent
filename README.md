# Nykaa Domain Support Agent (LangGraph)

Completed the **Nykaa (E-commerce & Retail)** track.
Everything runs under `MOCK_LLM` with zero API keys.

## How to Reproduce
Run the project scripts in sequence from the project root:

    # Day 1: Dataset Generation
    python dataset.py

    # Day 2: KB Chunking, Vector Indexing & Verification
    python chunking.py
    python index_kb.py
    python check_index.py

    # Day 3: Calibration, Grounded Generation, Evaluation & Tool Execution
    python calibrate.py
    python demo_grounded.py
    python evaluate_chunking.py
    python demo_tool.py

Outputs match files in `transcripts/` deterministically.

---

## Dataset Design Choices (Task 1)
- **Seed**: `42` (Landed within the target band on the first run; no re-seeding or weight adjustments were needed).
- **Total Records**: `60`
- **Field Draw Sequence**: Each record draws `category`, `status`, `order_value_inr`, `days_since_created`, then the `delayed_shipment` roll in strict sequential order from a single isolated `random.Random(42)` instance.
- **Category Weights**: Apparel `0.25`, Electronics `0.15`, Home `0.15`, Footwear `0.15`, Beauty `0.30` (Matches `CATEGORIES` declaration order).
- **Status Weights**: Placed `0.15`, Shipped `0.25`, Delivered `0.40`, Returned `0.10`, Refunded `0.10` (Matches `STATUSES` declaration order).
- **Price Range Reasoning**: Ranges reflect authentic Nykaa basket sizes, where cosmetic items and home decor are lower-to-mid ticket while personal styling appliances and tech gadgets command higher ticket values.
- **Price Ranges (INR)**:
  - Apparel: ₹700 – ₹6,000
  - Electronics: ₹1,500 – ₹12,000
  - Home: ₹500 – ₹4,000
  - Footwear: ₹900 – ₹5,000
  - Beauty: ₹200 – ₹3,500
- **delayed_shipment Rule & Probabilities**:
  - *Definition*: Missed promised dispatch or delivery SLA date. Flag is always `False` for `Returned` and `Refunded` orders (a deliberate simplification: the flag only describes orders not yet returned or refunded).
  - *Probabilities*:
    - `Returned` / `Refunded`: `0.0`
    - `Placed` / `Shipped`: `0.45` if `days_since_created >= 5`, else `0.05`
    - `Delivered`: `0.20` if `days_since_created >= 8`, else `0.0`
- **Validation Results**:
  - Categories: Beauty (20), Apparel (15), Footwear (9), Home (9), Electronics (7) [All >= 3: PASS]
  - Statuses: Delivered (20), Shipped (17), Placed (10), Refunded (7), Returned (6) [All >= 1: PASS]
  - Delayed Shipment Rate: **20.00%** (12/60) [Within 10.0%–30.0%: PASS]
  - Age Range: [0, 30] verified for all records
  - ID Uniqueness: 60 unique non-numeric IDs (`ORD-0001` to `ORD-0060`)

---

## Knowledge Base & Dual-Collection Indexing (Tasks 2 & 3)
- **Knowledge Base (`kb/`)**: 12 plain-prose policy documents (each exactly 4 sentences, 408–559 characters) covering return windows, COD refunds, delivery SLAs, reverse pickup, brand warranty, order cancellation, loyalty points, payment failures, size exchange, damaged items, international shipping, and escalation workflows.
- **KB Refinement & Freeze Note**: After the first retrieval check, `return_window.txt` was reworded to name `makeup, cosmetics, skincare` explicitly because the Beauty sentence originally only said `Beauty`. A query-specific version containing `'lipstick'` was tried and rejected as overfitting. The KB is permanently frozen after this edit.
- **Chunking Strategies (`chunking.py`)**:
  - **Fixed-Size (`kb_fixed`)**: Sliding window with `chunk_size=200` and `overlap=40` (guarded by `assert overlap < chunk_size`). Produces **40 chunks** across the 12 files (average length: **176.8 characters**).
  - **Sentence-Based (`kb_sentence`)**: Regex boundary splitter `(?<=[.!?])\s+`. Produces **48 chunks** across the 12 files—4 standalone sentences per document (average length: **123.4 characters**).
- **Vector Store (`index_kb.py`)**:
  - Embedded locally using `sentence-transformers/all-MiniLM-L6-v2`.
  - Indexed into persistent ChromaDB (`chroma_db/`) across two collections (`kb_fixed` and `kb_sentence`) with `metadata={"hnsw:space": "cosine"}`.
  - Chunk IDs follow `<doc_id>::<strategy>::<idx>` with `{"doc_id", "chunk_index", "strategy"}` metadata for automated evaluation.
- **Sample Retrieval Check (`check_index.py`)**:
  - Query: `"How long do I have to return a lipstick?"`
  - `kb_fixed` Rank 1: `return_window` (Sim `0.5188`), Rank 2: `return_window` (Sim `0.4551`), Rank 3: `damaged_item` (Sim `0.4096`).
  - `kb_sentence` Rank 1: `return_window` (Sim `0.5182`), Rank 2: `return_window` (Sim `0.4635`), Rank 3: `return_window` (Sim `0.4472`).

---

## Retrieval Calibration & Grounded Generation (Task 4)
- **Retrieval Engine (`rag_core.py`)**: Singleton loader for `SentenceTransformer` and `ChromaDB` client. Converts cosine distance to similarity via `1.0 - distance`. Configured with `MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"`.
- **Threshold Calibration (`calibrate.py`)**:
  - Evaluated top-1 similarity across 6 diverse in-scope customer queries and 5 out-of-scope queries on both collections.
  - For the deployed `kb_sentence` collection, purely unrelated queries form a tight cluster in `[0.0779, 0.1236]` (and `[0.0848, 0.1510]` on `kb_fixed`), while natural in-scope queries score in `[0.3453, 0.7956]`.
  - A near-miss corporate query (*"What is Nykaa's current share price on the stock exchange?"*) scored `0.4771` on `kb_sentence` (`0.4812` on `kb_fixed`), overlapping the in-scope range because financial terms match `loyalty_points`. Raising the similarity threshold to `0.49` would reject valid indirect customer queries like *"My shoes don't fit"* (`0.3453`).
  - **Operational Decision**: We calibrate the similarity threshold for the deployed `kb_sentence` collection at **`0.30`** (between max unrelated `0.1236` and min in-scope `0.3453`) and pair it with a deterministic out-of-scope domain guardrail (`is_near_miss_out_of_scope` in `generate.py`) that catches corporate/investor queries (`share price`, `stock exchange`, `ipo`, `market cap`).
- **Grounded Generation (`generate.py` & `demo_grounded.py`)**:
  - Queries below `0.30` similarity or matching the non-support domain filter return `"I don't know based on the Nykaa knowledge base."` with empty sources (`grounded=False`).
  - Tested in `demo_grounded.py` across 5 in-scope queries and 2 out-of-scope queries (both the unrelated cake query at `0.0779` and the near-miss share price query at `0.4771` cleanly trigger the fallback).

---

## Chunking Evaluation & Recommendation (Task 5)
Evaluated using 5 distinct in-scope queries against pre-established ground truth parent documents (`evaluate_chunking.py`). Retrieved chunks are deduplicated to parent `doc_id`s prior to computing metrics (so the precision denominator is the count of unique retrieved documents, not 3):

| Strategy | Mean Precision@3 | Mean Recall@3 | Mean Top-1 Similarity |
|---|---|---|---|
| **`kb_fixed`** | **0.7000** | **1.0000** | **0.4785** |
| **`kb_sentence`** | **0.7000** | **1.0000** | **0.5158** |

- **Tie-Breaking Recommendation**:
  Both strategies **tied** on document-level Mean Precision@3 (`0.7000`) and Mean Recall@3 (`1.0000`). We break the tie using the measured secondary signal of **Mean Top-1 Similarity (`0.5158` for `kb_sentence` vs. `0.4785` for `kb_fixed`)** and select `kb_sentence` for deployment. Preserving natural sentence boundaries prevents mid-sentence cuts and increases rank-1 confidence on focused policy queries (e.g., COD refund scores `0.7956` vs. `0.6773`).

---

## Order Status & Escalation Scoring Tool (Task 6)
- **Tool Implementation (`tools.py` & `demo_tool.py`)**:
  - Provides `check_order_status(record_id: str) -> dict` with case/whitespace normalization (`"  ord-0008  "` resolves to `ORD-0008`) and structured error dicts for unknown IDs.
- **Escalation Formula & Resolved-Order Rule**:
  $$\text{recency} = \min\left(\frac{\text{days\_since\_created}}{30}, 1.0\right)$$
  $$\text{delay\_term} = 1.0 \text{ if delayed\_shipment else } 0.0$$
  - **Open orders (`Placed`, `Shipped` — 27 of 60 records)**:
    $$\text{score} = 0.6 \times \text{delay\_term} + 0.4 \times \text{recency}$$
  - **Resolved orders (`Delivered`, `Returned`, `Refunded` — 33 of 60 records)**:
    $$\text{score} = 0.0$$
  - *Justification*: Even if a `Delivered` order experienced a historical shipment delay (`delayed_shipment=True`), the package has already reached the customer, so it requires no active delivery SLA escalation. This aligns with `kb/escalation_matrix.txt`, which restricts automated Tier-2/Tier-3 escalation to *unresolved* dispatch and transit breaches beyond 48 hours.
- **Zero-Inflated Distribution & Percentile Threshold**:
  - Because 33 of 60 orders are resolved (`score = 0.0`), the full 60-order score distribution is zero-inflated:
    - `P50`: **0.0000** | `P75`: **0.3000** | `P80`: **0.3733** | `P90`: **0.6960**
  - **Decision Rule**: Threshold is set at **`0.3733`**, corresponding to the **80th percentile** of all 60 orders (which effectively isolates the top urgency tier among the 27 open orders—specifically all delayed open orders plus non-delayed open orders aged $\ge 28$ days). Exactly **13 of 60 orders** exceed this threshold.
