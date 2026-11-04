"""
logging_utils.py
JSON-Lines audit logger with mandatory internal PII masking.
Appends forensic entries to logs/requests.jsonl with millisecond timing.
"""

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional

from guardrails import mask_pii

LOGS_DIR = "logs"
LOGS_FILE = os.path.join(LOGS_DIR, "requests.jsonl")


def _ensure_log_dir() -> None:
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)


def log_request(
    endpoint: str,
    status_code: int,
    duration_ms: float,
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    raw_request_text: Optional[str] = None,
    route: Optional[str] = None,
    guardrails_triggered: Optional[List[str]] = None,
    grounded: Optional[bool] = None,
    answer_length: Optional[int] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Appends one structured JSONL record to logs/requests.jsonl.
    Mandatory guarantee: raw_request_text is unconditionally scrubbed via mask_pii
    inside this function before serialization to disk.
    """
    _ensure_log_dir()

    # Unconditional internal masking guarantee
    masked_text = None
    pii_tags: List[str] = []
    if raw_request_text is not None:
        masked_text, findings = mask_pii(str(raw_request_text))
        for f in findings:
            pii_tags.append(f"PII_{f['type']}")

    combined_guardrails = list(guardrails_triggered or [])
    for tag in pii_tags:
        if tag not in combined_guardrails:
            combined_guardrails.append(tag)

    entry: Dict[str, Any] = {
        "trace_id": trace_id or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "conversation_id": conversation_id,
        "request_text": masked_text,
        "route": route,
        "guardrails_triggered": sorted(combined_guardrails),
        "grounded": grounded,
        "status_code": status_code,
        "duration_ms": round(float(duration_ms), 2),
        "answer_length": answer_length,
    }

    if error_message:
        entry["error_message"] = error_message

    line = json.dumps(entry, ensure_ascii=False)
    with open(LOGS_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return entry


if __name__ == "__main__":
    t0 = time.perf_counter()
    time.sleep(0.01)
    dur = (time.perf_counter() - t0) * 1000.0
    sample = log_request(
        endpoint="/test",
        status_code=200,
        duration_ms=dur,
        trace_id="tr-test-log",
        raw_request_text="Call 9876543210 regarding card ending 4321",
    )
    print("Logged sample:")
    print(json.dumps(sample, indent=2))
