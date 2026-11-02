# Nykaa Domain Support Agent (LangGraph)

An automated support agent built for the **Nykaa (E-commerce & Retail)** track using LangGraph. The system operates entirely under `MOCK_LLM` with **zero API key dependencies**.

---

## Nykaa Dataset Design & Specifications (Task 1)

* **Random Seed:** `42`
* **Total Records:** `60` unique non-numeric IDs (`ORD-0001` through `ORD-0060`)
* **Category Weights & Distribution:**
  * **Beauty:** `30%` (20 records)
  * **Apparel:** `25%` (15 records)
  * **Electronics:** `15%` (7 records)
  * **Home:** `15%` (9 records)
  * **Footwear:** `15%` (9 records)
  * *Status Check:* All categories meet the minimum threshold (>= 3).

* **Order Status Weights & Distribution:**
  * **Delivered:** `40%` (20 records)
  * **Shipped:** `25%` (17 records)
  * **Placed:** `15%` (10 records)
  * **Refunded:** `10%` (7 records)
  * **Returned:** `10%` (6 records)
  * *Status Check:* All statuses meet the minimum threshold (>= 1).

* **Price Ranges (INR) & Design Rationale:**
  * **Beauty:** ?200 � ?3,500
  * **Apparel:** ?700 � ?6,000
  * **Footwear:** ?900 � ?5,000
  * **Electronics:** ?1,500 � ?12,000
  * **Home:** ?500 � ?4,000
  * *Rationale:* Ranges accurately mirror authentic Nykaa basket sizes, where cosmetic items and home decor represent lower-to-mid ticket sizes, whereas personal styling appliances and electronics scale higher.

* **Delayed Shipment Rule & Probabilities:**
  * *Definition:* Defined as missing the promised dispatch or delivery SLA date. Applies exclusively to active or delivered orders (excluding terminal states like returned or refunded).
  * *Probability Rules:*
    * `Returned` / `Refunded`: `0.0`
    * `Placed` / `Shipped`: `0.45` if `days_since_created >= 5`, else `0.05`
    * `Delivered`: `0.20` if `days_since_created >= 8`, else `0.0`
  * *Validation Result:* Overall Delayed Shipment Rate is **20.00%** (12/60), perfectly falling within the target range of `10.0%�30.0%` (PASS).

* **Additional Constraints:**
  * Age Range: Verified within `[0, 30]` days for all records.
