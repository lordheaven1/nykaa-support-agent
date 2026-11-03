"""
tools.py
Order status lookup and deterministic escalation scoring engine for Task 6.
Provides check_order_status with percentile-calibrated escalation logic.
"""

from typing import Any, Dict, List
import numpy as np
from dataset import generate_orders

# Load orders dataset once at module level
_RAW_ORDERS: List[Dict[str, Any]] = generate_orders()
ORDERS_BY_ID: Dict[str, Dict[str, Any]] = {order["record_id"].upper(): order for order in _RAW_ORDERS}

RESOLVED_STATUSES = {"DELIVERED", "RETURNED", "REFUNDED"}


def calculate_escalation_score(order: Dict[str, Any]) -> float:
    """
    Computes a defendable escalation score in [0.0, 1.0].
    Resolved orders (Delivered, Returned, Refunded) have score 0.0.
    Open orders (Placed, Shipped) weight delayed_shipment (0.6) and recency (0.4).
    """
    status = str(order.get("status", "")).strip().upper()
    if status in RESOLVED_STATUSES:
        return 0.0

    delayed = 1.0 if order.get("delayed_shipment", False) else 0.0
    days = float(order.get("days_since_created", 0))
    recency = min(days / 30.0, 1.0)

    score = (0.6 * delayed) + (0.4 * recency)
    return round(float(score), 4)


def compute_dataset_escalation_stats() -> Dict[str, Any]:
    """Computes distribution and percentiles of escalation scores across all 60 orders."""
    scores = [calculate_escalation_score(o) for o in _RAW_ORDERS]
    p50 = float(np.percentile(scores, 50))
    p75 = float(np.percentile(scores, 75))
    p80 = float(np.percentile(scores, 80))
    p90 = float(np.percentile(scores, 90))

    # Pick 80th percentile as the empirical escalation threshold
    threshold = round(p80, 4)
    exceed_count = sum(1 for s in scores if s >= threshold and s > 0.0)

    return {
        "scores": scores,
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "p80": round(p80, 4),
        "p90": round(p90, 4),
        "threshold": threshold,
        "exceed_count": exceed_count,
        "total_orders": len(scores),
    }


STATS = compute_dataset_escalation_stats()
ESCALATION_THRESHOLD: float = STATS["threshold"]


def check_order_status(record_id: str) -> Dict[str, Any]:
    """
    Looks up order by ID with case/whitespace insensitivity.
    Returns status, order value, escalation score, escalation decision, and human reason.
    """
    if not record_id or not isinstance(record_id, str):
        return {
            "error": "Order not found",
            "message": "Invalid or missing order ID provided.",
            "record_id": record_id,
        }

    normalized_id = record_id.strip().upper()
    order = ORDERS_BY_ID.get(normalized_id)

    if not order:
        return {
            "error": "Order not found",
            "message": f"Order ID '{record_id.strip()}' does not exist in Nykaa records.",
            "record_id": record_id.strip(),
        }

    status = order["status"]
    delayed = order["delayed_shipment"]
    days = order["days_since_created"]
    score = calculate_escalation_score(order)
    should_escalate = score >= ESCALATION_THRESHOLD and score > 0.0

    if status.upper() in RESOLVED_STATUSES:
        reason = f"Normal priority: Order is resolved ({status}). Escalation score is 0.0000."
    elif should_escalate:
        reason = (
            f"High priority: Open order '{status}' with delay={delayed} at {days} days old "
            f"yielded escalation score {score:.4f} (>= {ESCALATION_THRESHOLD:.4f} threshold)."
        )
    else:
        reason = (
            f"Normal priority: Open order '{status}' with delay={delayed} at {days} days old "
            f"yielded score {score:.4f} below threshold {ESCALATION_THRESHOLD:.4f}."
        )

    return {
        "record_id": order["record_id"],
        "status": status,
        "order_value_inr": order["order_value_inr"],
        "delayed_shipment": delayed,
        "days_since_created": days,
        "escalation_score": score,
        "escalate": should_escalate,
        "reason": reason,
    }


if __name__ == "__main__":
    print(f"Total orders: {STATS['total_orders']}")
    print(f"Percentiles: P50={STATS['p50']}, P75={STATS['p75']}, P80={STATS['p80']}, P90={STATS['p90']}")
    print(f"Chosen threshold: {ESCALATION_THRESHOLD} ({STATS['exceed_count']} orders flagged for escalation)")
