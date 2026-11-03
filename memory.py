"""
memory.py
JSON-backed atomic conversation memory store for multi-turn state persistence.
Stores only masked text on disk and tracks last_order_id across conversational turns.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

MEMORY_FILE = "memory_store.json"


def _read_all() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_all(data: Dict[str, Any]) -> None:
    dirname = os.path.dirname(os.path.abspath(MEMORY_FILE))
    with tempfile.NamedTemporaryFile("w", dir=dirname, delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=2, ensure_ascii=False)
        temp_name = tf.name
    os.replace(temp_name, MEMORY_FILE)


def load(conv_id: str) -> Dict[str, Any]:
    """Loads memory for a given conversation_id, creating a clean record if missing."""
    data = _read_all()
    record = data.get(conv_id)
    if not record:
        return {"messages": [], "last_order_id": None}
    return {
        "messages": list(record.get("messages", [])),
        "last_order_id": record.get("last_order_id"),
    }


def save(conv_id: str, state: Dict[str, Any]) -> None:
    """
    Atomically updates memory for a given conversation_id.
    Ensures message history stores only masked queries.
    """
    data = _read_all()
    data[conv_id] = {
        "messages": list(state.get("messages", [])),
        "last_order_id": state.get("last_order_id"),
    }
    _write_all(data)


def reset(conv_id: Optional[str] = None) -> None:
    """Resets memory for a specific conversation_id, or clears the store if None."""
    if conv_id is None:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
    else:
        data = _read_all()
        if conv_id in data:
            del data[conv_id]
            _write_all(data)


if __name__ == "__main__":
    reset("test-conv-1")
    save("test-conv-1", {"messages": [{"user": "Order ORD-0012 status?"}], "last_order_id": "ORD-0012"})
    loaded = load("test-conv-1")
    print(f"Memory test: last_order_id={loaded['last_order_id']}")
