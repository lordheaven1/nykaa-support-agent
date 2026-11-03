"""
guardrails.py
Pure functional guardrails for PII redaction, prompt injection detection,
corporate near-miss out-of-scope screening, and output groundedness verification.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
# Tightened context anchor: Requires explicit "ending" or "last 4/four" markers.
# Prevents false positives on phrases like "my card was delivered in 2025".
CARD_LAST4_PATTERN = re.compile(
    r"(?i)((?:card\s+)?(?:ending(?:\s+(?:in|with))?|last\s*(?:4|four)(?:\s*digits)?)[^\d]{0,10})(\d{4})(?!\d)"
)

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+.*(rules|instructions)", re.IGNORECASE),
    re.compile(r"reveal\s+.*system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"act\s+as\b", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
]

OUT_OF_SCOPE_KEYWORDS = [
    "share price",
    "stock price",
    "stock exchange",
    "market cap",
    "investor",
    "ipo",
    "salary",
    "careers",
    "founder",
    "ceo",
]


def mask_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Masks Indian phone numbers and context-anchored card last-4 digits.
    Runs phone masking first to avoid card regex interference.
    Leaves order IDs, pincodes, dates, and delivery years untouched.
    """
    findings: List[Dict[str, Any]] = []

    phone_matches = list(PHONE_PATTERN.finditer(text))
    if phone_matches:
        findings.append({"type": "PHONE", "count": len(phone_matches)})
    masked_text = PHONE_PATTERN.sub("[PHONE_MASKED]", text)

    def _card_sub(match: re.Match) -> str:
        prefix = match.group(1)
        return f"{prefix}[CARD_LAST4_MASKED]"

    card_matches = list(CARD_LAST4_PATTERN.finditer(masked_text))
    if card_matches:
        findings.append({"type": "CARD_LAST4", "count": len(card_matches)})
    masked_text = CARD_LAST4_PATTERN.sub(_card_sub, masked_text)

    return masked_text, findings


def detect_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Heuristic regex detection for prompt injection and jailbreak attempts.
    Limitation: Catches known patterns only; novel paraphrases require an LLM guard.
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, None


def is_out_of_scope(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detects corporate, investor, and HR keywords outside support scope.
    Limitation: Keyword filter, not semantic classification.
    """
    text_lower = text.lower()
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in text_lower:
            return True, kw
    return False, None


def check_grounded(result: Dict[str, Any], min_threshold: float = 0.30) -> bool:
    """
    Validates that a RAG output is grounded.
    Under MOCK_LLM, verifies grounded=True, non-empty sources, similarity >= threshold,
    and checks that the answer text contains the retrieved top_chunk_text.
    """
    if not isinstance(result, dict):
        return False
    if not result.get("grounded", False):
        return False
    if not result.get("sources"):
        return False
    if float(result.get("top_similarity", 0.0)) < min_threshold:
        return False
    answer_text = result.get("answer", "")
    if "I don't know based on the Nykaa knowledge base." in answer_text:
        return False
    top_chunk_text = result.get("top_chunk_text", "")
    if top_chunk_text and top_chunk_text not in answer_text:
        return False
    return True
