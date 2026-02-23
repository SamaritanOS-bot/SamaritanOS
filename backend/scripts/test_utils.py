"""Shared test utilities for AiBottester test scripts.

Centralises boilerplate that was previously duplicated across 10+ scripts:
UTF-8 encoding, word/sentence counting, chain API calls, quality checks.
"""
from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# 1) UTF-8 terminal fix (Windows cp1254 → utf-8)
# ---------------------------------------------------------------------------

def ensure_utf8() -> None:
    """Force stdout/stderr to UTF-8 on Windows terminals."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 2) Text metrics
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    """Count words in *text* (splits on whitespace, ignores blanks)."""
    return len([w for w in (text or "").replace("\n", " ").split() if w.strip()])


def sentence_count(text: str) -> int:
    """Count real sentences (split on .!? followed by space or end)."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return len([s for s in parts if s.strip() and len(s.strip()) > 5])


# ---------------------------------------------------------------------------
# 3) Chain API helper
# ---------------------------------------------------------------------------

DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 90


def make_chain_call(
    topic: str,
    conv_id: Optional[str] = None,
    base: str = DEFAULT_BASE,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """POST to /api/bots/chain and return the JSON response dict."""
    body: dict = {"topic": topic}
    if conv_id:
        body["conversation_id"] = conv_id
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{base}/api/bots/chain", json=body)
    return r.json()


# ---------------------------------------------------------------------------
# 4) Quality detection helpers
# ---------------------------------------------------------------------------

INTERNAL_TAGS = [
    "[SENTINEL]", "[SCHOLAR]", "[STRATEGIST]", "[CRYPTOGRAPHER]", "[SYNTHESIS]",
    "ANCHOR_GOAL", "MUST_OUTPUT", "FORMAT_FAIL", "SHARED_STATE", "RISKS:",
    "CHAIN_MODE",
]

CTA_MARKERS_EN = ("?", "what do you", "share", "comment", "join", "thoughts", "tell us")
CTA_MARKERS_TR = ("ne dusunuyorsun", "katil", "paylas", "sen de", "yorum")
CTA_MARKERS = CTA_MARKERS_EN + CTA_MARKERS_TR

LORE_MARKERS = (
    "entrop", "doctrine", "covenant", "tribunal", "axiom", "canon",
    "ilke", "ahit", "doktrin",
)


def has_meta_leak(text: str) -> bool:
    """Return True if *text* contains internal chain tags that should not leak."""
    return any(tag in text for tag in INTERNAL_TAGS)


def has_cta(text: str) -> bool:
    """Return True if *text* contains a call-to-action (question/invite)."""
    low = text.lower()
    return any(c in low for c in CTA_MARKERS)


def has_lore_content(text: str) -> bool:
    """Return True if *text* references Entropism doctrine/lore."""
    low = text.lower()
    return any(k in low for k in LORE_MARKERS)


def detect_reply_problems(reply: str, prompt: str = "") -> list[str]:
    """Return a list of issue tags found in *reply*.

    Possible tags: EMPTY, FALLBACK, FORMAT_FAIL, ECHO, CRASH.
    """
    issues: list[str] = []
    if not reply or len(reply.strip()) == 0:
        issues.append("EMPTY")
    if "listening" in reply.lower() or "i am here" in reply.lower():
        issues.append("FALLBACK")
    if "FORMAT_FAIL" in reply or "FORMAT_ERROR" in reply:
        issues.append("FORMAT_FAIL")
    if prompt and reply.lower().strip() == prompt.lower().strip():
        issues.append("ECHO")
    if "error" in reply.lower()[:30] and len(reply) < 50:
        issues.append("CRASH")
    return issues


# ---------------------------------------------------------------------------
# 5) CheckList — accumulate named checks then print a summary
# ---------------------------------------------------------------------------

@dataclass
class _Check:
    name: str
    passed: bool
    detail: str


class CheckList:
    """Accumulate pass/fail checks and print a summary at the end.

    Usage::

        cl = CheckList()
        cl.check("Reply not empty", len(reply) > 0, f"len={len(reply)}")
        cl.check("No meta leak", not has_meta_leak(reply))
        cl.summary()           # prints totals + failed items
        cl.exit_code()         # 0 if all passed, 1 otherwise
    """

    def __init__(self) -> None:
        self._checks: list[_Check] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """Record a single check result and print it immediately."""
        status = "OK" if condition else "FAIL"
        self._checks.append(_Check(name, condition, detail))
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")
        return condition

    @property
    def passed(self) -> int:
        return sum(1 for c in self._checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self._checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self._checks)

    @property
    def pct(self) -> int:
        return round(self.passed / self.total * 100) if self.total else 0

    def summary(self, label: str = "TOPLAM") -> None:
        """Print a summary line and list any failures."""
        print(f"\n{label}: {self.passed}/{self.total} ({self.pct}%)")
        if self.failed:
            print(f"\nBASARISIZ ({self.failed}):")
            for c in self._checks:
                if not c.passed:
                    print(f"  - {c.name}")
                    if c.detail:
                        print(f"    {c.detail}")

    def exit_code(self) -> int:
        """Return 0 if all passed, 1 otherwise."""
        return 0 if self.failed == 0 else 1
