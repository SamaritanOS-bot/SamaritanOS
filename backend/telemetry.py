"""Chain telemetry: event logging, last-output persistence, summary reader."""

import json
import os
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from text_utils import normalize_whitespace

_BACKEND_DIR = Path(__file__).resolve().parent

CHAIN_LAST_OUTPUT_PATH = _BACKEND_DIR / "chain_last_output.json"
CHAIN_TELEMETRY_PATH = _BACKEND_DIR / (
    os.getenv("CHAIN_TELEMETRY_LOG", "chain_telemetry.log") or "chain_telemetry.log"
)


def telemetry_enabled() -> bool:
    return (os.getenv("CHAIN_TELEMETRY_ENABLED", "1") or "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def append_chain_telemetry(event: str, payload: Optional[dict] = None) -> None:
    if not telemetry_enabled():
        return
    try:
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": normalize_whitespace(event or "event"),
            "payload": payload or {},
        }
        CHAIN_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAIN_TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass


def persist_last_chain_output(response) -> None:
    """Persist response (BotChainResponse or dict-like with .model_dump())."""
    try:
        payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        CHAIN_LAST_OUTPUT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def read_last_chain_output() -> Optional[dict]:
    if not CHAIN_LAST_OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(CHAIN_LAST_OUTPUT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_iso_utc(ts_raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except Exception:
        return None


def read_chain_telemetry_summary(hours: int = 24, limit: int = 6) -> dict:
    h = max(1, min(168, int(hours or 24)))
    n = max(1, min(25, int(limit or 6)))
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=h)

    counts: dict[str, int] = {}
    degrade_reason_counts: dict[str, int] = {}
    recent: deque = deque(maxlen=n)
    parse_errors = 0
    file_exists = CHAIN_TELEMETRY_PATH.exists()

    if file_exists:
        try:
            with open(CHAIN_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    raw = (line or "").strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except Exception:
                        parse_errors += 1
                        continue
                    ts = _parse_iso_utc(str(item.get("ts") or ""))
                    if not ts or ts < cutoff:
                        continue
                    event = normalize_whitespace(str(item.get("event") or "event")) or "event"
                    counts[event] = counts.get(event, 0) + 1
                    if event == "post_degrade":
                        payload_obj = item.get("payload") if isinstance(item.get("payload"), dict) else {}
                        reason = normalize_whitespace(str(payload_obj.get("degrade_reason") or "")) or "UNKNOWN"
                        degrade_reason_counts[reason] = degrade_reason_counts.get(reason, 0) + 1
                        reason_list = payload_obj.get("degrade_reasons")
                        if isinstance(reason_list, list):
                            for r in reason_list:
                                rr = normalize_whitespace(str(r or ""))
                                if not rr:
                                    continue
                                key = f"{rr}_seen"
                                degrade_reason_counts[key] = degrade_reason_counts.get(key, 0) + 1
                    recent.append(
                        {
                            "ts": ts.isoformat(),
                            "event": event,
                            "payload": item.get("payload") or {},
                        }
                    )
        except Exception:
            pass

    post_locks = counts.get("post_mode_lock", 0)
    post_degrades = counts.get("post_degrade", 0)
    degrade_rate = round((post_degrades / post_locks) * 100.0, 2) if post_locks > 0 else 0.0

    return {
        "status": "ok",
        "now_utc": now_utc.isoformat(),
        "window_hours": h,
        "telemetry_enabled": telemetry_enabled(),
        "file_exists": file_exists,
        "file_path": str(CHAIN_TELEMETRY_PATH.name),
        "counts": counts,
        "degrade_reason_counts": degrade_reason_counts,
        "degrade_rate_percent": degrade_rate,
        "invalid_output_retry_count": counts.get("post_generic_fallback_repair", 0),
        "mode_lock_violation_count": counts.get("mode_lock_violation", 0),
        "recent": list(recent),
        "parse_errors": parse_errors,
    }
