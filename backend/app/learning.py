from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("AIRCANVAS_DATA_DIR", "backend/data"))
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"


def _write(record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def record_feedback(
    landmarks: list[dict[str, float]],
    predicted: str,
    corrected: str | None = None,
    confidence: float = 0.0,
    source: str = "live",
) -> dict[str, Any]:
    record = {
        "ts": time.time(),
        "landmarks": landmarks,
        "predicted": predicted,
        "corrected": corrected,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
        "hard_example": corrected is not None and corrected != predicted,
    }
    _write(record)
    return record


def stats() -> dict[str, Any]:
    if not FEEDBACK_FILE.exists():
        return {"samples": 0, "hard_examples": 0, "labels": {}, "path": str(FEEDBACK_FILE)}
    labels: Counter[str] = Counter()
    hard = 0
    total = 0
    with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            label = row.get("corrected") or row.get("predicted") or "UNKNOWN"
            labels[label] += 1
            hard += int(bool(row.get("hard_example")))
    return {"samples": total, "hard_examples": hard, "labels": dict(labels), "path": str(FEEDBACK_FILE)}
