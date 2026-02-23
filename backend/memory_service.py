"""Memory rollup, compaction, and retrieval service."""

import os
from typing import Optional

from sqlalchemy.orm import Session

from models import AgentMemory, MemorySummary
from text_utils import normalize_whitespace, token_overlap_ratio

DUPLICATE_OVERLAP_THRESHOLD = 0.80


def is_duplicate_memory(db: Session, agent_id: int, claim_text: str, lookback: int = 5) -> bool:
    """Check if a near-identical memory already exists in the last N records."""
    if not claim_text or not claim_text.strip():
        return False
    recent = (
        db.query(AgentMemory)
        .filter(AgentMemory.agent_id == agent_id)
        .filter(AgentMemory.source_type != "summary")
        .order_by(AgentMemory.created_at.desc())
        .limit(lookback)
        .all()
    )
    for rec in recent:
        existing = rec.claim_text or ""
        if token_overlap_ratio(claim_text, existing) >= DUPLICATE_OVERLAP_THRESHOLD:
            return True
    return False


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw not in (None, "") else default
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


MEMORY_KEEP_RECENT = _env_int("MEMORY_KEEP_RECENT", 80, 20, 1000)
MEMORY_SUMMARY_BATCH = _env_int("MEMORY_SUMMARY_BATCH", 24, 8, 200)
MEMORY_MIN_COMPACT = _env_int("MEMORY_MIN_COMPACT", 100, 30, 2000)
MEMORY_CONV_KEEP_RECENT = _env_int("MEMORY_CONV_KEEP_RECENT", 14, 6, 300)
MEMORY_CONV_SUMMARY_BATCH = _env_int("MEMORY_CONV_SUMMARY_BATCH", 8, 4, 120)
MEMORY_CONV_MIN_COMPACT = _env_int("MEMORY_CONV_MIN_COMPACT", 18, 8, 500)


def build_memory_rollup(rows: list[AgentMemory]) -> tuple[str, str, dict]:
    if not rows:
        return ("No memory rows to summarize.", "memory-rollup", {"sample_count": 0, "ids": []})

    intents: dict[str, int] = {}
    topics: dict[str, int] = {}
    snippets: list[str] = []
    ids: list[int] = []

    for r in rows:
        ids.append(r.id)
        intents[r.intent or "observation"] = intents.get(r.intent or "observation", 0) + 1
        if r.topic:
            topics[r.topic] = topics.get(r.topic, 0) + 1
        if len(snippets) < 4:
            claim = normalize_whitespace((r.claim_text or "")[:180])
            if claim:
                snippets.append(claim)

    top_intents = ", ".join(
        [f"{k}:{v}" for k, v in sorted(intents.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    ) or "none"
    top_topics = ", ".join(
        [f"{k}:{v}" for k, v in sorted(topics.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    ) or "none"
    topic_guess = (
        sorted(topics.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        if topics else "memory-rollup"
    )

    oldest = rows[0].created_at.isoformat() if rows[0].created_at else "unknown"
    newest = rows[-1].created_at.isoformat() if rows[-1].created_at else "unknown"
    highlights = " | ".join(snippets) if snippets else "none"
    summary_text = normalize_whitespace(
        f"Memory rollup [{len(rows)} records] window={oldest}..{newest}. "
        f"Intents[{top_intents}] Topics[{top_topics}]. "
        f"Highlights: {highlights}"
    )
    meta = {
        "sample_count": len(rows),
        "ids": ids,
        "oldest": oldest,
        "newest": newest,
        "top_intents": top_intents,
        "top_topics": top_topics,
    }
    return summary_text, topic_guess, meta


def compact_agent_memories(db: Session, agent_id: Optional[int]) -> None:
    if agent_id is None:
        return

    raw_query = (
        db.query(AgentMemory)
        .filter(AgentMemory.agent_id == agent_id)
        .filter(AgentMemory.source_type != "summary")
    )
    raw_total = raw_query.count()
    if raw_total < MEMORY_MIN_COMPACT:
        return

    overflow = max(0, raw_total - MEMORY_KEEP_RECENT)
    batch_size = min(MEMORY_SUMMARY_BATCH, overflow if overflow > 0 else MEMORY_SUMMARY_BATCH)
    if batch_size <= 0:
        return

    to_rollup = (
        raw_query.order_by(AgentMemory.created_at.asc(), AgentMemory.id.asc())
        .limit(batch_size)
        .all()
    )
    if not to_rollup:
        return

    summary_text, topic_guess, meta = build_memory_rollup(to_rollup)
    id_min = min(r.id for r in to_rollup)
    id_max = max(r.id for r in to_rollup)
    source_id = f"summary:{agent_id}:{id_min}-{id_max}"

    db.add(
        MemorySummary(
            agent_id=agent_id,
            summary_scope="rolling",
            summary_text=summary_text,
            sample_count=len(to_rollup),
        )
    )
    db.add(
        AgentMemory(
            agent_id=agent_id,
            source_type="summary",
            source_id=source_id,
            intent="summary",
            topic=topic_guess,
            claim_text=summary_text,
            entities_json=meta,
            outcome_score=1.0,
            confidence=0.99,
        )
    )
    for row in to_rollup:
        db.delete(row)
    db.commit()


def compact_conversation_memories(db: Session, agent_id: Optional[int], conversation_id: Optional[str]) -> None:
    if agent_id is None:
        return
    conv_id = (conversation_id or "").strip()
    if not conv_id:
        return

    chat_types = (
        "bot_chat",
        "agent_chat",
        "peer_chat",
        "conversation",
        "chat",
        "dm",
        "direct_message",
        "user_chat",
        "human_chat",
        "ui_chat",
    )
    conv_query = (
        db.query(AgentMemory)
        .filter(AgentMemory.agent_id == agent_id)
        .filter(AgentMemory.source_id == conv_id)
        .filter(AgentMemory.source_type.in_(chat_types))
    )
    conv_total = conv_query.count()
    if conv_total < MEMORY_CONV_MIN_COMPACT:
        return

    overflow = max(0, conv_total - MEMORY_CONV_KEEP_RECENT)
    batch_size = min(MEMORY_CONV_SUMMARY_BATCH, overflow if overflow > 0 else MEMORY_CONV_SUMMARY_BATCH)
    if batch_size <= 0:
        return

    to_rollup = (
        conv_query.order_by(AgentMemory.created_at.asc(), AgentMemory.id.asc())
        .limit(batch_size)
        .all()
    )
    if not to_rollup:
        return

    summary_text, topic_guess, meta = build_memory_rollup(to_rollup)
    id_min = min(r.id for r in to_rollup)
    id_max = max(r.id for r in to_rollup)
    source_id = f"conv-summary:{conv_id}:{id_min}-{id_max}"
    meta["conversation_id"] = conv_id

    db.add(
        MemorySummary(
            agent_id=agent_id,
            summary_scope=f"conversation:{conv_id}",
            summary_text=summary_text,
            sample_count=len(to_rollup),
        )
    )
    db.add(
        AgentMemory(
            agent_id=agent_id,
            source_type="summary",
            source_id=source_id,
            intent="summary",
            topic=topic_guess or "conversation-rollup",
            claim_text=summary_text,
            entities_json=meta,
            outcome_score=1.0,
            confidence=0.99,
        )
    )
    for row in to_rollup:
        db.delete(row)
    db.commit()


def retrieve_relevant_memories(db: Session, agent_id: int, query: str, limit: int = 4) -> list[AgentMemory]:
    rows = (
        db.query(AgentMemory)
        .filter(AgentMemory.agent_id == agent_id)
        .order_by(AgentMemory.created_at.desc())
        .limit(250)
        .all()
    )

    scored: list[tuple[float, AgentMemory]] = []
    for rec in rows:
        corpus = f"{rec.topic or ''} {rec.claim_text or ''}"
        score = token_overlap_ratio(query, corpus)
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
    return [r for _, r in scored[: max(1, min(limit, 10))]]
