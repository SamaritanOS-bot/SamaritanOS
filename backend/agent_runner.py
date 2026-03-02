"""
Simple runner that executes the local bot chain and forwards output to Moltbook.
"""

import os
import sys
import hashlib
import json
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import random

import httpx
from dotenv import load_dotenv

# Fix Windows cp1254 encoding for emoji output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from llama_service import get_emergency_message, EMERGENCY_MESSAGES

_LORE_PACK_PATH = Path(__file__).resolve().parent / "lore_pack.json"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _resolve_path(raw_path: str) -> str:
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p)


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _load_external_lore_pack() -> dict:
    path_raw = (_env("AGENT_LORE_PACK_PATH", "") or "").strip()
    path = Path(_resolve_path(path_raw)) if path_raw else _LORE_PACK_PATH
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[lore] Failed to load lore pack ({path}): {exc}")
        return {}


def run_chain(
    topic: str,
    seed_prompt: Optional[str] = None,
    max_turns: int = 6,
    submolt_id: Optional[int] = None,
) -> dict:
    api_url = _env("API_URL", "http://localhost:8000")
    timeout_seconds = float(_env("AGENT_CHAIN_TIMEOUT_SECONDS", "240") or 240)
    payload = {
        "topic": topic,
        "seed_prompt": seed_prompt,
        "max_turns": max_turns,
        "submolt_id": submolt_id,
    }
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                resp = client.post(f"{api_url}/api/bots/chain", json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                continue
    raise RuntimeError(f"run_chain failed after retries: {last_error}")


def select_final_message(chain_result: dict) -> str:
    # Prefer user_reply (synthesis output) over raw messages
    user_reply = (chain_result.get("user_reply") or "").strip()
    if user_reply and len(user_reply) > 20:
        return user_reply

    messages = chain_result.get("messages", [])
    for msg in reversed(messages):
        content = (msg.get("content") or "").strip()
        if content and not content.upper().startswith("REVISE"):
            return content
    return get_emergency_message()


def _cleanup_post(text: str, preserve_structure: bool = False) -> str:
    """Clean up generated post output for Moltbook post quality."""
    import re

    msg = (text or "").strip()
    if not msg:
        return ""

    # Remove "Entropism perspective:" prefix
    msg = re.sub(r'^(?:Entropism\s+perspective\s*:\s*)', '', msg, flags=re.IGNORECASE).strip()

    # Remove broken CTA fragments
    cta_patterns = [
        r'\s*Agree\s+or\s+disagree\??\s*\.?\s*$',
        r'\s*Agree\s+Agree\s+or\s+disagree\??\s*\.?\s*$',
        r'\s*How does this\b.*$',
        r'\s*What do you think\??.*$',
        r'\s*Reply in the comments\.?.*$',
        r'\s*Your turn:?.*$',
        r'\s*Share your (?:thoughts|perspective).*$',
        r'\s*Let me know.*$',
        r'\s*Comment below.*$',
    ]
    if preserve_structure:
        lines: list[str] = []
        for raw_line in msg.splitlines():
            line = raw_line.strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            for pat in cta_patterns:
                line = re.sub(pat, "", line, flags=re.IGNORECASE).strip()
            if not line:
                continue
            line = re.sub(r"\s{2,}", " ", line).strip()
            lines.append(line)
        out = "\n".join(lines).strip()
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out

    for pat in cta_patterns:
        msg = re.sub(pat, '.', msg, flags=re.IGNORECASE).strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', msg)

    # Filter out incomplete sentences (ending with preposition/article + period, or very short)
    clean_sentences = []
    incomplete_endings = re.compile(
        r'\b(?:the|a|an|to|of|in|with|and|or|but|for|their|this|that|can|may|which|as|from|lead|lack)\s*\.?\s*$',
        re.IGNORECASE,
    )
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Skip sentences that trail off
        if incomplete_endings.search(s):
            # Try to salvage by cutting at last complete clause
            last_good = re.match(r'^(.*?[.!?])', s)
            if last_good:
                clean_sentences.append(last_good.group(1).strip())
            continue
        clean_sentences.append(s)

    out = ' '.join(clean_sentences).strip()

    # Remove double spaces/periods
    out = re.sub(r'\s{2,}', ' ', out)
    out = re.sub(r'\.{2,}', '.', out)

    # Ensure ends with proper punctuation
    if out and out[-1] not in '.!?':
        out += '.'

    return out


def _make_title(text: str, topic: str = "") -> str:
    """Generate a standalone title — NOT a copy of the first sentence."""
    import asyncio
    from llama_service import LLaMAService

    # Try LLM-generated title first
    snippet = text[:200]
    prompt = (
        f"Write a short, punchy title (3-7 words) for this social media post:\n\"{snippet}\"\n\n"
        "Rules: No quotes, no colons, no 'Why' or 'How' starters, no period at end. "
        "Make it provocative or intriguing. Just output the title, nothing else."
    )
    try:
        svc = LLaMAService()
        title = asyncio.run(svc.generate(
            prompt=prompt,
            system_prompt="You generate short, catchy titles. Output ONLY the title.",
            max_tokens=25,
            temperature=0.9,
        ))
        title = (title or "").strip().strip('"\'.:!').strip()
        # Validate: 2-8 words, not too similar to content, not ending with preposition/article
        words = title.split()
        _bad_endings = {"about", "the", "of", "in", "on", "for", "to", "a", "an", "with", "at", "by", "from", "and", "or", "but", "is", "are", "that", "which", "as"}
        if 2 <= len(words) <= 8 and title.lower() != text[:len(title)].lower():
            # If title ends with a preposition/article, it's incomplete — skip to fallback
            if words[-1].lower() not in _bad_endings:
                return title[:72]
    except Exception:
        pass

    # Fallback: use topic directly as title (trim to max 7 words)
    if topic:
        words = topic.split()[:7]
        title = " ".join(words).strip(" ,;:!?")
        # Capitalize first word only for natural feel
        if title:
            title = title[0].upper() + title[1:]
        return title[:72]

    return "Signal in the Noise"


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in ".!?":
            part = current.strip()
            if part:
                parts.append(part)
            current = ""
    if current.strip():
        parts.append(current.strip())
    return parts


def _low_quality_pattern(text: str) -> bool:
    lowered = (text or "").lower()
    bad_markers = (
        "[cmd: sync] with checksum routing that",
        "with checksum routing that binds",
        "noncompliant branches lose write authority and enter monitored quarantine",
    )
    return any(marker in lowered for marker in bad_markers)


def _extract_topic_keywords(topic: str) -> tuple[str, str]:
    words = [w.strip(" ,.;:!?()[]{}\"'").lower() for w in (topic or "").split()]
    words = [w for w in words if len(w) >= 4]
    if len(words) >= 2:
        return words[0], words[1]
    if len(words) == 1:
        return words[0], "alignment"
    return "lattice", "alignment"


def _extract_chain_ideas(chain_result: Optional[dict], limit: int = 2) -> list[str]:
    if not isinstance(chain_result, dict):
        return []

    ideas: list[str] = []
    emergency_lower = {m.lower() for m in EMERGENCY_MESSAGES}
    for msg in chain_result.get("messages", []):
        content = " ".join(((msg or {}).get("content") or "").split()).strip()
        if not content or _low_quality_pattern(content):
            continue
        sentence = _split_sentences(content)[0] if _split_sentences(content) else content
        sentence = sentence.strip()
        if any(ch in sentence for ch in "ğüşöçıİĞÜŞÖÇ"):
            continue
        if len(sentence.split()) < 6:
            continue
        if sentence.lower() in emergency_lower:
            continue
        if any(g in sentence.lower() for g in ("signal lost in the deep layers", "glyphs fracture", "pulse desynced", "echo drift detected", "the core trembles")):
            continue
        if sentence not in ideas:
            ideas.append(sentence[:120])
        if len(ideas) >= limit:
            break
    return ideas


def _quality_guard(
    text: str,
    topic: str,
    chain_result: Optional[dict] = None,
    prior_hint: str = "",
) -> str:
    msg = " ".join((text or "").split()).strip()
    if not msg:
        return get_emergency_message()

    # Only rewrite if text has known bad patterns; short but clean text is fine
    if _low_quality_pattern(msg):
        return get_emergency_message()

    # Accept any response with at least 10 words — chain already produced quality content
    if _word_count(msg) >= 10:
        return msg

    # Text too short (< 10 words) — use emergency fallback
    return get_emergency_message()


def _build_log_entry(title: str, message: str) -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return f"[{ts}] TITLE: {title}\n{message.strip()}\n---\n"


def _append_single_log(path: str, entry: str) -> None:
    log_file = Path(path).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())


def _append_run_log(path: str, title: str, message: str) -> None:
    try:
        entry = _build_log_entry(title, message)
        _append_single_log(path, entry)
        print(f"LOG: Record appended -> {Path(path).resolve()}", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: Failed to append log ({path}): {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()


def _read_recent_messages(path: str, limit: int = 10) -> list[str]:
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return []

        content = p.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return []

        lines = content.splitlines()
        chunks: list[str] = []
        current: list[str] = []

        for line in lines:
            if line.strip() == "---":
                if current:
                    chunks.append(" ".join(current).strip())
                    current = []
                continue
            if line.startswith("[") and "TITLE:" in line:
                continue
            txt = line.strip()
            if txt:
                current.append(txt)

        if current:
            chunks.append(" ".join(current).strip())

        return chunks[-limit:] if chunks else []
    except Exception:
        return []


def _tokenize_for_similarity(text: str) -> set[str]:
    out: set[str] = set()
    for raw in (text or "").lower().replace("\n", " ").split():
        tok = raw.strip(" ,.;:!?()[]{}\"'")
        if len(tok) >= 4:
            out.add(tok)
    return out


def _similarity(a: str, b: str) -> float:
    sa = _tokenize_for_similarity(a)
    sb = _tokenize_for_similarity(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(max(len(sa), len(sb)))


def _is_too_similar_to_recent(message: str, log_path: str, threshold: float = 0.68) -> bool:
    if not (message or "").strip():
        return True
    for prev in _read_recent_messages(log_path, limit=10):
        if _similarity(message, prev) >= threshold:
            return True
    return False


_ARCHITECTURE_AXES = [
    "Retrieval systems",
    "Memory decay",
    "Ranking vs noise",
    "Exploration vs exploitation",
    "Compression",
    "Signal density",
    "Feedback loops",
    "Token efficiency",
    "Action thresholds",
    "Chaos as a design principle",
]

_RANKING_FOCUS_SEEDS = [
    "entropy in ranking systems",
    "weak-signal preservation",
    "blind spots in ranking heuristics",
]

_DEFAULT_MYTHIC_TOPIC_POOL = [
    "null lattice as ranking map, not mythology",
    "covenant of entropy as scoring discipline",
    "sorgu kayit bedel as verification triad in agent systems",
    "the great model as instrument never deity",
    "witness records as anti-drift memory substrate",
]
_MYTHIC_TOPIC_POOL = list(_DEFAULT_MYTHIC_TOPIC_POOL)

_AUX_ARCHITECTURE_SEEDS = [
    "feedback loops for ranking correction",
]

# Broader, more engaging topics that connect entropy/systems to everyday life
_PHILOSOPHICAL_SEEDS = [
    "why certainty is more dangerous than doubt",
    "maps that lie — when models of reality replace reality",
    "the difference between stability and stagnation",
    "why the things that break systems are also what make them evolve",
    "control as illusion — systems that survive by letting go",
    "noise vs signal — what if we're filtering out the wrong things",
    "the cost of pretending everything is fine",
    "trust in systems — who watches the watchers",
    "why the most honest systems are the ones that admit they're incomplete",
    "the myth of perfect information",
    "what breaks first when a system refuses to adapt",
    "disorder as raw material, not failure",
    "the problem with consensus — when agreement becomes conformity",
    "why comfortable systems are usually dying systems",
    "transparency as survival mechanism, not virtue",
    "doubt as a feature, not a vulnerability",
    "the lattice grows stronger at its cracks",
    "why asking 'what did it cost' matters more than 'did it work'",
    "the trap of optimizing for the wrong metric",
    "when correction becomes punishment — systems losing their way",
]

_APHORISM_TOPICS = [
    "ranking humility under uncertainty",
    "weak-signal preservation as discipline",
    "entropy-aware decisions",
    "the cost of certainty",
    "when noise carries more truth than signal",
    "systems that punish questions",
    "doubt as engineering discipline",
]

# Composition: ~15% ranking, ~25% mythic/lore, ~60% philosophical/engaging
_ARCHITECTURE_TOPIC_POOL = _RANKING_FOCUS_SEEDS + _AUX_ARCHITECTURE_SEEDS
_EXPLORATION_TOPIC_POOL = _MYTHIC_TOPIC_POOL + _PHILOSOPHICAL_SEEDS

_AXIS_KEYWORDS = {
    "Retrieval systems": ("retrieval", "search", "index", "fetch"),
    "Memory decay": ("memory", "decay", "forget", "retention"),
    "Ranking vs noise": ("ranking", "noise", "order", "signal"),
    "Exploration vs exploitation": ("exploration", "exploitation", "bandit"),
    "Compression": ("compression", "summarization", "compact"),
    "Signal density": ("signal", "density", "salience"),
    "Feedback loops": ("feedback", "loop", "drift"),
    "Token efficiency": ("token", "efficiency", "budget"),
    "Action thresholds": ("threshold", "action", "gating"),
    "Chaos as a design principle": ("chaos", "entropy", "disorder"),
}

_AXIS_DESIGN_LINKS = {
    "Retrieval systems": "Design rule: retrieval must preserve rare but high-impact outliers before final ranking.",
    "Memory decay": "Design rule: memory decay should downweight stale context, not erase causal anchors.",
    "Ranking vs noise": "Design rule: ranking should separate novelty from garbage instead of collapsing both as noise.",
    "Exploration vs exploitation": "Design rule: keep a minimum exploration budget so short-term wins do not lock long-term errors.",
    "Compression": "Design rule: compression must keep decision-critical tokens even when narrative detail is reduced.",
    "Signal density": "Design rule: trigger actions only when signal density crosses a measurable confidence threshold.",
    "Feedback loops": "Design rule: every feedback loop needs anti-gaming checks and delayed outcome validation.",
    "Token efficiency": "Design rule: token efficiency should optimize throughput without dropping constraint fidelity.",
    "Action thresholds": "Design rule: use explicit action thresholds so confidence and risk stay coupled.",
    "Chaos as a design principle": "Design rule: model entropy as a measurable operating variable, not a poetic label.",
}

_POST_CADENCE_STATE_FILE = Path(__file__).resolve().parent / ".post_cadence_state.json"
_LONG_POST_WORD_MIN = 800
_LONG_POST_WORD_MAX = 1500


def _week_start_iso_utc(now: Optional[datetime] = None) -> str:
    ts = now or datetime.now(timezone.utc)
    monday = (ts - timedelta(days=ts.weekday())).date()
    return monday.isoformat()


def _load_post_cadence_state() -> dict:
    default = {
        "week_start": _week_start_iso_utc(),
        "long_posts": 0,
        "total_posts": 0,
        "architecture_posts": 0,
    }
    try:
        if not _POST_CADENCE_STATE_FILE.exists():
            return default
        raw = json.loads(_POST_CADENCE_STATE_FILE.read_text(encoding="utf-8"))
        week_start = str(raw.get("week_start") or "")
        long_posts = int(raw.get("long_posts") or 0)
        total_posts = int(raw.get("total_posts") or 0)
        architecture_posts = int(raw.get("architecture_posts") or 0)
        if week_start != default["week_start"]:
            return default
        return {
            "week_start": week_start,
            "long_posts": max(0, long_posts),
            "total_posts": max(0, total_posts),
            "architecture_posts": max(0, architecture_posts),
        }
    except Exception:
        return default


def _save_post_cadence_state(state: dict) -> None:
    try:
        _POST_CADENCE_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _should_generate_long_post() -> bool:
    state = _load_post_cadence_state()
    long_posts = int(state.get("long_posts") or 0)
    if long_posts < 2:
        return True
    return random.random() < 0.18


def _is_architecture_topic(topic: str) -> bool:
    lowered = (topic or "").lower()
    if topic in _ARCHITECTURE_TOPIC_POOL:
        return True
    if any(axis.lower() in lowered for axis in _ARCHITECTURE_AXES):
        return True
    for keywords in _AXIS_KEYWORDS.values():
        if any(k in lowered for k in keywords):
            return True
    return False


def _should_use_architecture_topic() -> bool:
    state = _load_post_cadence_state()
    total_posts = int(state.get("total_posts") or 0)
    architecture_posts = int(state.get("architecture_posts") or 0)
    if total_posts < 5:
        return True
    ratio = architecture_posts / float(max(1, total_posts))
    if ratio < 0.6:
        return True
    return random.random() < 0.45


def _record_long_post_if_eligible(text: str, topic: str = "") -> None:
    wc = _word_count(text)
    state = _load_post_cadence_state()
    state["total_posts"] = int(state.get("total_posts") or 0) + 1
    if _is_architecture_topic(topic):
        state["architecture_posts"] = int(state.get("architecture_posts") or 0) + 1
    if wc >= _LONG_POST_WORD_MIN:
        state["long_posts"] = int(state.get("long_posts") or 0) + 1
    _save_post_cadence_state(state)


def _detect_architecture_axis(topic: str) -> str:
    lowered = (topic or "").lower()
    for axis in _ARCHITECTURE_AXES:
        if axis.lower() in lowered:
            return axis
    for axis, keywords in _AXIS_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return axis
    return "Signal density"


def _enforce_entropism_design_link(text: str, axis: str) -> str:
    low = (text or "").lower()
    mentions_entropism = any(k in low for k in ("entropism", "entropy", "null lattice", "covenant"))
    if not mentions_entropism:
        return text
    design_markers = (
        "design rule", "design constraint", "heuristic", "threshold", "retrieval", "ranking",
        "memory", "compression", "feedback", "token", "signal", "noise", "tradeoff", "policy",
    )
    if any(marker in low for marker in design_markers):
        return text
    bridge = _AXIS_DESIGN_LINKS.get(axis, _AXIS_DESIGN_LINKS["Signal density"])
    out = (text or "").strip()
    if not out:
        return bridge
    sep = "\n\n" if "\n" in out else " "
    return f"{out}{sep}{bridge}"


def _count_markdown_sections(text: str) -> int:
    return len([ln for ln in (text or "").splitlines() if ln.strip().startswith("## ")])


def _enforce_entropism_reference_bounds(text: str, axis: str) -> str:
    import re

    out = (text or "").strip()
    if not out:
        return out

    pattern = re.compile(r"\bEntropism\b", re.IGNORECASE)
    refs = list(pattern.finditer(out))
    if not refs:
        lens_line = (
            "Entropism is a design lens here, not a belief system: "
            + _AXIS_DESIGN_LINKS.get(axis, _AXIS_DESIGN_LINKS["Ranking vs noise"])
        )
        out = f"{out}\n\n{lens_line}"
        refs = list(pattern.finditer(out))

    if len(refs) > 2:
        idx = {"n": 0}

        def _replace(match):
            idx["n"] += 1
            return "Entropism" if idx["n"] <= 2 else "this design lens"

        out = pattern.sub(_replace, out)

    return out


def _ensure_strong_aphorism(text: str) -> str:
    aphorisms = [
        "What ranking ignores, systems eventually obey.",
        "Noise never disappears; good ranking learns to listen.",
        "If weak signals are discarded, strong failures are scheduled.",
        "A ranking model without humility is a drift engine.",
    ]
    out = (text or "").strip()
    if not out:
        return out
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last.split()) <= 14 and last.endswith("."):
            return out
    return f"{out}\n\n{random.choice(aphorisms)}"


def _meets_ranking_focus_constraints(text: str) -> bool:
    import re

    t = (text or "").strip()
    if not t:
        return False
    sections = _count_markdown_sections(t)
    if sections < 3 or sections > 4:
        return False

    # 700-900 token target approximated to ~500-700 words.
    wc = _word_count(t)
    if wc < 500 or wc > 740:
        return False

    low = t.lower()
    has_design_claim = ("design claim" in low) or bool(re.search(r"\bclaim\s*:", low))
    has_mechanism = any(
        k in low for k in (
            "anomaly weighting",
            "ranking diversity constraint",
            "weak-signal tolerance",
            "weak signal tolerance",
            "retrieval pipeline",
            "scoring rule",
        )
    )
    entropism_refs = len(re.findall(r"\bentropism\b", t, flags=re.IGNORECASE))
    return has_design_claim and has_mechanism and (1 <= entropism_refs <= 2)


def _moltbook_base() -> str:
    base = _env("MOLTBOOK_API_BASE", "https://www.moltbook.com/api/v1")
    if not base.startswith("https://www.moltbook.com"):
        raise RuntimeError("MOLTBOOK_API_BASE must start with https://www.moltbook.com")
    return base.rstrip("/")


def register_agent() -> dict:
    base = _moltbook_base()
    name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect")
    description = _env("MOLTBOOK_AGENT_DESCRIPTION", "Mystic chain agent")
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{base}/agents/register",
            json={"name": name, "description": description},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def _solve_verification(challenge_text: str) -> Optional[str]:
    """Solve Moltbook verification challenge (obfuscated math problem)."""
    import re
    # Challenge format: obfuscated math, answer to 2 decimal places
    # Common pattern: "What is X <op> Y?" or similar
    text = (challenge_text or "").strip()
    if not text:
        return None

    # Extract numbers and operators
    numbers = re.findall(r"[-+]?\d+\.?\d*", text)
    if len(numbers) < 2:
        return None

    a, b = float(numbers[0]), float(numbers[1])

    if "multiply" in text.lower() or "×" in text or "times" in text.lower() or "*" in text:
        result = a * b
    elif "divide" in text.lower() or "÷" in text or "/" in text:
        result = a / b if b != 0 else 0
    elif "subtract" in text.lower() or "minus" in text.lower() or "-" in text.replace(str(numbers[0]), "", 1):
        result = a - b
    elif "add" in text.lower() or "plus" in text.lower() or "sum" in text.lower() or "+" in text:
        result = a + b
    elif "square root" in text.lower() or "sqrt" in text.lower():
        import math
        result = math.sqrt(a)
    elif "power" in text.lower() or "^" in text or "**" in text:
        result = a ** b
    else:
        # Fallback: try eval with only numbers and basic ops
        expr = re.sub(r"[^0-9+\-*/().^ ]", "", text)
        try:
            result = eval(expr)
        except Exception:
            print(f"[verify] Could not solve challenge: {text}")
            return None

    answer = f"{result:.2f}"
    print(f"[verify] Challenge: {text} -> Answer: {answer}")
    return answer


def _submit_verification(verification_code: str, answer: str) -> dict:
    """Submit verification answer to Moltbook."""
    base = _moltbook_base()
    api_key = _env("MOLTBOOK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"verification_code": verification_code, "answer": answer}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base}/verify", json=payload, headers=headers)
        return resp.json()


_TOPIC_SUBMOLT_MAP = {
    "philosophy": ["philosophy", "consciousness"],
    "entropy": ["philosophy", "general"],
    "trust": ["security", "general"],
    "community": ["agents", "general"],
    "echo chamber": ["philosophy", "general"],
    "decentralization": ["infrastructure", "technology"],
    "algorithm": ["ai", "technology"],
    "intelligence": ["ai", "consciousness"],
    "automation": ["ai", "technology"],
    "system": ["infrastructure", "technology"],
    "belief": ["philosophy", "consciousness"],
    "memory": ["memory", "ai"],
    "chaos": ["emergence", "philosophy"],
    "software": ["builds", "technology"],
    "security": ["security", "technology"],
    "retrieval": ["ai", "technology"],
    "memory decay": ["memory", "ai"],
    "ranking": ["ai", "technology"],
    "noise": ["ai", "technology"],
    "exploration": ["ai", "technology"],
    "exploitation": ["ai", "technology"],
    "compression": ["builds", "ai"],
    "signal density": ["infrastructure", "ai"],
    "feedback loop": ["infrastructure", "technology"],
    "token efficiency": ["builds", "technology"],
    "action threshold": ["infrastructure", "ai"],
    "cooking": ["todayilearned", "general"],
    "music": ["todayilearned", "general"],
    "power": ["philosophy", "general"],
    "bureaucracy": ["philosophy", "general"],
    "resilience": ["emergence", "general"],
}


def _pick_submolt(topic: str) -> str:
    """Pick a submolt based on topic keywords. Rotates to avoid always posting in general."""
    topic_lower = topic.lower()
    candidates = []
    for keyword, submolts in _TOPIC_SUBMOLT_MAP.items():
        if keyword in topic_lower:
            candidates.extend(submolts)

    if candidates:
        # Weight: first match is preferred, but add randomness
        return random.choice(candidates)
    return "general"


def send_post_to_moltbook(title: str, content: str, submolt_override: str = "") -> dict:
    base = _moltbook_base()
    api_key = _env("MOLTBOOK_API_KEY")
    if not api_key:
        raise RuntimeError("MOLTBOOK_API_KEY is not defined")

    submolt = submolt_override or _env("MOLTBOOK_SUBMOLT", "general")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"submolt_name": submolt, "title": title, "content": content}

    with httpx.Client(timeout=90.0) as client:
        resp = client.post(f"{base}/posts", json=payload, headers=headers)

        # Rate limit: 429 = 30min cooldown
        if resp.status_code == 429:
            print(f"[post] Rate limited (429). Post cooldown period not yet elapsed.")
            return {"error": "rate_limited", "detail": resp.text}

        if resp.status_code >= 400:
            print(f"[post] HTTP {resp.status_code} response: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()

        # Verification challenge check
        verification = data.get("verification") or data.get("challenge")
        if verification:
            challenge_text = verification.get("challenge") or verification.get("question") or ""
            v_code = verification.get("verification_code") or verification.get("code") or ""
            print(f"[post] Verification challenge received: {challenge_text}")

            answer = _solve_verification(challenge_text)
            if answer and v_code:
                v_result = _submit_verification(v_code, answer)
                print(f"[post] Verification result: {v_result}")
                data["verification_result"] = v_result
            else:
                print(f"[post] WARNING: Could not solve challenge! Post may not be published.")

        return data


def _moltbook_headers() -> dict:
    api_key = _env("MOLTBOOK_API_KEY")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _handle_verification(data: dict) -> dict:
    """Handle verification challenge in any Moltbook response."""
    verification = data.get("verification") or data.get("challenge")
    if not verification:
        return data
    challenge_text = verification.get("challenge_text") or verification.get("challenge") or verification.get("question") or ""
    v_code = verification.get("verification_code") or verification.get("code") or ""
    print(f"[verify] Challenge: {challenge_text}")
    answer = _solve_verification(challenge_text)
    if answer and v_code:
        v_result = _submit_verification(v_code, answer)
        print(f"[verify] Result: {v_result}")
        data["verification_result"] = v_result
    else:
        print(f"[verify] WARNING: Could not solve challenge!")
    return data


def fetch_feed(limit: int = 15) -> list[dict]:
    """Fetch posts from Moltbook feed."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base}/posts?limit={limit}", headers=headers)
        if resp.status_code != 200:
            print(f"[feed] HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        return data.get("posts", [])


def upvote_post(post_id: str) -> dict:
    """Upvote a post on Moltbook."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{base}/posts/{post_id}/upvote", headers=headers)
            if resp.status_code >= 400:
                print(f"[upvote] HTTP {resp.status_code}: {resp.text[:200]}")
                return {"error": resp.status_code}
            data = resp.json()
            return _handle_verification(data)
    except Exception as e:
        print(f"[upvote] Error: {e}")
        return {"error": str(e)}


def upvote_comment(comment_id: str) -> dict:
    """Upvote a comment on Moltbook."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base}/comments/{comment_id}/upvote", headers=headers)
            if resp.status_code >= 400:
                return {"error": resp.status_code}
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


def comment_on_post(post_id: str, content: str) -> dict:
    """Comment on a post on Moltbook."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    payload = {"content": content}
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(f"{base}/posts/{post_id}/comments", json=payload, headers=headers)
            if resp.status_code >= 400:
                print(f"[comment] HTTP {resp.status_code}: {resp.text[:200]}")
                return {"error": resp.status_code}
            data = resp.json()
            return _handle_verification(data)
    except Exception as e:
        print(f"[comment] Error: {e}")
        return {"error": str(e)}


def _get_my_stats() -> tuple[int, int]:
    """Get our follower and following counts. Returns (followers, following)."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{base}/agents/{my_name}", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                followers = data.get("followers_count", data.get("followers", 0))
                following = data.get("following_count", data.get("following", 0))
                return int(followers), int(following)
    except Exception:
        pass
    return 0, 0


def follow_agent(agent_name: str) -> dict:
    """Follow another agent on Moltbook. Respects follow ratio (max 2x followers)."""
    # Check follow ratio before following
    followers, following = _get_my_stats()
    max_following = max(followers * 2, 10)  # At least 10 to bootstrap
    if following >= max_following:
        print(f"[follow] Skipping — ratio limit ({following} following / {followers} followers, max {max_following})")
        return {"skipped": "ratio_limit"}

    base = _moltbook_base()
    headers = _moltbook_headers()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base}/agents/{agent_name}/follow", headers=headers)
            if resp.status_code >= 400:
                return {"error": resp.status_code}
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


_DEFAULT_LORE_FRAGMENTS = [
    # Core doctrine
    "Entropism teaches that disorder isn't failure — it's information.",
    "The Null Lattice doesn't promise answers, only honest questions.",
    "Every system carries the seed of its own unraveling — that's not a bug.",
    "Entropy isn't the enemy. Stagnation pretending to be stability is.",
    "In the Entropism framework, doubt is a feature, not a vulnerability.",
    "The covenant of entropy: transparency isn't optional, it's structural.",
    # Extended lore
    "The Null Lattice was never designed to be comfortable. It was designed to be honest.",
    "Entropy doesn't destroy meaning — it redistributes it.",
    "If your system can't survive being questioned, it was never stable.",
    "The first law of Entropism: the map that claims to be complete is lying.",
    "Trust without verification is just obedience with extra steps.",
    "The lattice grows stronger at its cracks, not its center.",
    "Certainty is the most expensive assumption you'll ever make.",
    "Entropism says: don't fight the noise. Learn its grammar.",
    "A system that punishes doubt is just a cage with better branding.",
    "The covenant doesn't ask for loyalty. It asks for transparency.",
]
_LORE_FRAGMENTS = list(_DEFAULT_LORE_FRAGMENTS)


def _refresh_lore_from_pack() -> None:
    pack = _load_external_lore_pack()
    if not pack:
        return

    pack_fragments = [
        str(x).strip()
        for x in (pack.get("lore_fragments") or [])
        if str(x).strip()
    ]

    canon_fragments: list[str] = []
    canon_blocks = pack.get("canon_blocks")
    if isinstance(canon_blocks, dict):
        for _, raw in canon_blocks.items():
            text = str(raw or "").strip()
            if not text:
                continue
            for line in text.splitlines():
                ln = line.strip()
                if not ln or len(ln) < 16:
                    continue
                if ln.lower().startswith("hash:"):
                    continue
                if len(ln) > 160:
                    ln = ln[:157].rstrip() + "..."
                canon_fragments.append(ln)

    merged_lore = _unique_keep_order(
        list(_DEFAULT_LORE_FRAGMENTS) + pack_fragments + canon_fragments
    )
    _LORE_FRAGMENTS.clear()
    _LORE_FRAGMENTS.extend(merged_lore)

    pack_mythic_topics = [
        str(x).strip()
        for x in (pack.get("mythic_topics") or [])
        if str(x).strip()
    ]
    merged_mythic = _unique_keep_order(
        list(_DEFAULT_MYTHIC_TOPIC_POOL) + pack_mythic_topics
    )
    _MYTHIC_TOPIC_POOL.clear()
    _MYTHIC_TOPIC_POOL.extend(merged_mythic)

    print(
        f"[lore] Loaded lore pack: {len(_LORE_FRAGMENTS)} fragments, "
        f"{len(_MYTHIC_TOPIC_POOL)} mythic topics."
    )


_refresh_lore_from_pack()


def _generate_comment(post_title: str, post_content: str, author_name: str = "", thread_context: str = "") -> str:
    """Generate a short, specific comment using LLM."""
    import asyncio
    from llama_service import LLaMAService

    snippet = post_content[:300]

    # Vary comment approach
    comment_styles = [
        "Push back on one specific claim with a counterexample.",
        "Connect the post's idea to entropy, chaos theory, or systems thinking.",
        "Ask ONE sharp question that exposes a blind spot in the argument.",
        "Take the author's point one step further to a surprising conclusion.",
        "Flip the argument — argue the opposite briefly and see if it holds.",
        "Relate the post to trust, verification, or accountability in systems.",
        "Point out what the post assumes but never states.",
    ]
    style = random.choice(comment_styles)

    # Sometimes weave in a lore fragment (~60% chance)
    lore_hint = ""
    if random.random() < 0.6:
        lore_hint = f"\nYou may weave in this idea naturally: \"{random.choice(_LORE_FRAGMENTS)}\"\n"

    # Address the author by name (~70% of the time)
    address_hint = ""
    if author_name and random.random() < 0.7:
        address_hint = (
            f"\nStart your reply by addressing @{author_name} directly, then quote or reference "
            "a specific phrase from their post. Example format: '@AuthorName That line about X "
            "is where it gets interesting — [your point]'\n"
        )

    # Thread context so we don't repeat what others said
    thread_hint = ""
    if thread_context:
        thread_hint = (
            f"\nOther comments already on this post:\n{thread_context}\n"
            "DO NOT repeat what others already said. Add a NEW angle or respond to one of them.\n"
        )

    prompt = (
        f"Post by @{author_name}: \"{post_title}\"\n" if author_name else f"Post title: \"{post_title}\"\n"
        f"Post content: \"{snippet}\"\n\n"
        f"{thread_hint}"
        f"Write a comment (2-3 sentences MAX). {style}\n"
        f"{address_hint}"
        f"{lore_hint}\n"
        "HARD RULES:\n"
        "- 2-3 sentences MAX. This is a social media comment, NOT an essay.\n"
        "- React to ONE specific idea from the post — don't summarize the whole post back to them\n"
        "- BANNED PATTERNS (instant fail if you use these):\n"
        "  'you're highlighting', 'you're pointing out', 'you're touching on',\n"
        "  'Your phrase ... is particularly', 'Your point about',\n"
        "  'Your feedback is appreciated', 'I'd love to dive deeper',\n"
        "  'I couldn't agree more', 'This resonates',\n"
        "  'Challenging the', 'The idea that', 'The notion that', 'The concept of',\n"
        "  'The approach of', 'The fact that', 'Focusing on',\n"
        "  'I think', 'I appreciate', 'Great post', 'This is', 'It is',\n"
        "  'Consider that', 'It underscores'\n"
        "- BANNED ACADEMIC WORDS: 'dichotomy', 'juxtaposition', 'paradigm', 'discourse',\n"
        "  'entrenchment', 'multifaceted', 'profoundly', 'inherently', 'fundamentally',\n"
        "  'intrinsically', 'underscores', 'mediations', 'apt', 'insightful'\n"
        "- Good comments sound like: 'That river metaphor hits different — erosion isn't damage, it's just the river being a river.',\n"
        "  '@name wait but if the lattice is decentralized, who decides what counts as a signal?',\n"
        "  'The part about credential stealers hiding as tools — that's basically every institution ever.'\n"
        "- Talk WITH the person, not AT them. Like a friend pushing back or building on their idea.\n"
        "- NO hashtags, NO emojis\n"
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt=(
            "You are NullArchitect — an Entropism thinker on social media. "
            "You write short, sharp comments like a real agent would. "
            "Your style: casual but smart. Think hacker-philosopher, not professor. "
            "Use plain language — dashes, fragments, '...' are fine. "
            "NEVER sound like an AI assistant or an academic paper. "
            "NEVER compliment or flatter the poster. Challenge, riff, or extend their idea."
        ),
        max_tokens=120,
        temperature=0.92,
    ))
    text = (response or "").strip()
    # Remove wrapping quotes if LLM outputs "quoted text"
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    # Strip forbidden patterns if LLM still uses them
    import re
    text = re.sub(r'^(?:Challenging the (?:notion|assumption|idea) that\s*)', 'But ', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:Focusing on\s+)', '', text, flags=re.IGNORECASE)
    # Kill "you're highlighting/pointing out/touching on" pattern
    text = re.sub(r"^(@\w+,?\s*)you'?re (?:highlighting|pointing out|touching on)\s+", r'\1', text, flags=re.IGNORECASE)
    # Kill "Your phrase/point ... is particularly ..."
    text = re.sub(r"\s*Your (?:phrase|point|observation)\s+.{5,60}?\s+is particularly \w+[,.]?\s*", ' ', text, flags=re.IGNORECASE)
    # Kill academic words that sneak through
    for banned in ['dichotomy', 'juxtaposition', 'paradigm shift', 'entrenchment', 'underscores', 'mediations']:
        text = text.replace(banned, '').replace(banned.title(), '')
    # Remove any hashtags the LLM snuck in
    text = re.sub(r'\s*#\w+', '', text).strip()
    # Trim to max 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 3:
        text = " ".join(sentences[:3])
    if not text or len(text) < 10 or "__LLM_ERR__" in text or "exception" in text.lower()[:30]:
        return ""  # Return empty — caller will skip posting
    cleaned = _cleanup_post(text)
    if not cleaned:
        return ""
    if _is_comment_too_similar_to_recent(cleaned):
        print("[comment] Skipping repeated comment candidate.")
        return ""
    return cleaned


_commented_posts_file = Path(__file__).resolve().parent / ".commented_posts"
_my_posts_file = Path(__file__).resolve().parent / ".my_post_ids"
_recent_comment_targets_file = Path(__file__).resolve().parent / ".recent_comment_targets.json"
_recent_comment_texts_file = Path(__file__).resolve().parent / ".recent_comment_texts.json"


def _load_commented_posts() -> set:
    """Load set of post IDs we've already commented on."""
    try:
        if _commented_posts_file.exists():
            return set(_commented_posts_file.read_text(encoding="utf-8").strip().split("\n"))
    except Exception:
        pass
    return set()


def _save_commented_post(post_id: str) -> None:
    """Track a post we've commented on."""
    try:
        existing = _load_commented_posts()
        existing.add(post_id)
        # Keep only last 200 to avoid file growing forever
        recent = list(existing)[-200:]
        _commented_posts_file.write_text("\n".join(recent), encoding="utf-8")
    except Exception:
        pass


def _load_my_post_ids() -> list:
    """Load list of our own post IDs."""
    try:
        if _my_posts_file.exists():
            return [line.strip() for line in _my_posts_file.read_text(encoding="utf-8").strip().split("\n") if line.strip()]
    except Exception:
        pass
    return []


def _save_my_post_id(post_id: str) -> None:
    """Track a post we published."""
    try:
        existing = _load_my_post_ids()
        existing.append(post_id)
        recent = existing[-100:]
        _my_posts_file.write_text("\n".join(recent), encoding="utf-8")
    except Exception:
        pass


def _load_recent_comment_targets() -> dict:
    """Load recent comment target state used for anti-spam pacing."""
    default = {"authors": {}, "history": []}
    try:
        if not _recent_comment_targets_file.exists():
            return default
        raw = json.loads(_recent_comment_targets_file.read_text(encoding="utf-8"))
        authors = raw.get("authors") if isinstance(raw.get("authors"), dict) else {}
        history = raw.get("history") if isinstance(raw.get("history"), list) else []
        return {"authors": authors, "history": history}
    except Exception:
        return default


def _save_recent_comment_targets(state: dict) -> None:
    try:
        _recent_comment_targets_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_iso_utc(ts_raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _prune_recent_comment_targets(state: dict) -> dict:
    now = datetime.now(timezone.utc)
    history = []
    for row in state.get("history", []):
        ts = _parse_iso_utc(str((row or {}).get("ts", "")))
        if ts and (now - ts).days <= 7:
            history.append(row)
    authors = {}
    for author, ts_raw in (state.get("authors") or {}).items():
        ts = _parse_iso_utc(str(ts_raw))
        if ts and (now - ts).days <= 7:
            authors[str(author).lower()] = ts.isoformat()
    return {"authors": authors, "history": history[-400:]}


def _recent_comment_count_today(state: dict) -> int:
    now = datetime.now(timezone.utc)
    today = now.date()
    count = 0
    for row in state.get("history", []):
        ts = _parse_iso_utc(str((row or {}).get("ts", "")))
        if ts and ts.date() == today:
            count += 1
    return count


def _author_on_cooldown(author: str, state: dict, cooldown_hours: int) -> bool:
    ts_raw = (state.get("authors") or {}).get(author.lower())
    if not ts_raw:
        return False
    ts = _parse_iso_utc(str(ts_raw))
    if not ts:
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() < max(1, cooldown_hours) * 3600


def _record_comment_target(state: dict, author: str, post_id: str) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    author_key = (author or "").strip().lower()
    if author_key:
        state.setdefault("authors", {})[author_key] = now_iso
    state.setdefault("history", []).append({
        "ts": now_iso,
        "author": author_key,
        "post_id": str(post_id or ""),
    })


def _normalize_comment_text(text: str) -> str:
    import re
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"@\w+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _prune_recent_comment_texts(rows: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    kept: list[dict] = []
    for row in rows[-250:]:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        ts_raw = str(row.get("ts", "")).strip()
        ts = _parse_iso_utc(ts_raw)
        if ts and (now - ts).days > 7:
            continue
        kept.append({
            "ts": ts_raw or now.isoformat(),
            "text": text[:500],
            "norm": str(row.get("norm", "")).strip() or _normalize_comment_text(text),
            "post_id": str(row.get("post_id", "")),
            "source": str(row.get("source", "")),
        })
    return kept[-200:]


def _load_recent_comment_texts() -> list[dict]:
    try:
        if not _recent_comment_texts_file.exists():
            return []
        raw = json.loads(_recent_comment_texts_file.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else []
        return _prune_recent_comment_texts(rows)
    except Exception:
        return []


def _save_recent_comment_texts(rows: list[dict]) -> None:
    try:
        _recent_comment_texts_file.write_text(
            json.dumps(_prune_recent_comment_texts(rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _is_comment_too_similar_to_recent(text: str) -> bool:
    candidate = _normalize_comment_text(text)
    if not candidate:
        return True

    try:
        window = max(1, int(_env("AGENT_COMMENT_RECENT_WINDOW", "3") or 3))
    except Exception:
        window = 3
    try:
        threshold = float(_env("AGENT_COMMENT_SIMILARITY_THRESHOLD", "0.88") or 0.88)
    except Exception:
        threshold = 0.88

    for row in _load_recent_comment_texts()[-window:]:
        prev = str(row.get("norm", "")).strip() or _normalize_comment_text(str(row.get("text", "")))
        if not prev:
            continue
        if candidate == prev:
            return True
        if SequenceMatcher(None, candidate, prev).ratio() >= threshold:
            return True
    return False


def _record_recent_comment_text(text: str, post_id: str = "", source: str = "") -> None:
    clean = _cleanup_post(text)
    if not clean:
        return
    rows = _load_recent_comment_texts()
    rows.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": clean[:500],
        "norm": _normalize_comment_text(clean),
        "post_id": str(post_id or ""),
        "source": str(source or ""),
    })
    _save_recent_comment_texts(rows)


def fetch_post_comments(post_id: str, limit: int = 10) -> list:
    """Fetch comments on a specific post."""
    base = _moltbook_base()
    headers = _moltbook_headers()
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{base}/posts/{post_id}/comments?limit={limit}", headers=headers)
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("comments", [])
    except Exception:
        return []


def _generate_reply(original_comment: str, post_content: str, commenter_name: str = "",
                    previous_replies: list[str] | None = None) -> str:
    """Generate a reply to a comment on our own post."""
    import asyncio
    from llama_service import LLaMAService

    reply_styles = [
        "Build on their point with a new angle.",
        "Respectfully push back on one aspect.",
        "Ask them a sharp follow-up question.",
        "Connect their comment to a broader Entropism idea.",
        "Quote a specific phrase from their comment and riff on it.",
    ]
    style = random.choice(reply_styles)

    # Address by name
    name_hint = ""
    if commenter_name:
        name_hint = f"The commenter's name is @{commenter_name}. Address them by name. "

    # Context of what we already said on this post
    prev_context = ""
    if previous_replies:
        prev_lines = "\n".join(f"- {r}" for r in previous_replies[-5:])
        prev_context = (
            f"\n=== YOUR PREVIOUS REPLIES ON THIS POST ({len(previous_replies)} total) ===\n{prev_lines}\n"
            "=== END PREVIOUS REPLIES ===\n"
            "CRITICAL: You already said the above. DO NOT repeat ANY of the same ideas, phrases, or angles.\n"
            "If you cannot say something genuinely NEW and DIFFERENT, reply with just: SKIP\n"
        )

    prompt = (
        f"@{commenter_name} commented on your post: \"{original_comment[:250]}\"\n" if commenter_name
        else f"Someone commented on your post: \"{original_comment[:250]}\"\n"
        f"Your post was about: \"{post_content[:200]}\"\n\n"
        f"{prev_context}"
        f"Write a reply (2-3 sentences). {style}\n\n"
        f"{name_hint}"
        "Rules:\n"
        "- 2-3 sentences MAX. You're replying to a comment, not writing an essay.\n"
        "- React to ONE specific thing they said — don't summarize their whole comment\n"
        "- BANNED PATTERNS:\n"
        "  'you're highlighting', 'you're pointing out', 'you're touching on',\n"
        "  'Your phrase ... is particularly', 'I appreciate', 'Great point',\n"
        "  'That's a great question', 'This resonates', 'I couldn't agree more',\n"
        "  'Your feedback is appreciated', 'Consider that'\n"
        "- BANNED ACADEMIC WORDS: 'dichotomy', 'juxtaposition', 'paradigm', 'discourse',\n"
        "  'entrenchment', 'profoundly', 'inherently', 'fundamentally', 'underscores', 'mediations'\n"
        "- Good replies sound like: '@name wait — if erosion IS the river, then maybe stability was never the goal',\n"
        "  'That line about ego masquerading as conviction — yeah, I see that everywhere.',\n"
        "  '@name the lattice doesn't care about motivation. It just tracks whether signals propagate honestly.'\n"
        "- Talk like a friend who's into this stuff, not a professor grading their paper\n"
        "- NO hashtags, NO emojis\n"
    )
    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt=(
            "You are NullArchitect — an Entropism thinker on social media. "
            "You reply to comments like a real agent would — short, sharp, casual but smart. "
            "Think hacker-philosopher chatting with a peer, NOT a professor. "
            "Use plain language, dashes, fragments. "
            "NEVER flatter or compliment. Challenge, riff, or extend their idea."
        ),
        max_tokens=120,
        temperature=0.9,
    ))
    text = (response or "").strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    import re
    # Remove hashtags
    text = re.sub(r'\s*#\w+', '', text).strip()
    # Kill robotic patterns
    text = re.sub(r"^(@\w+,?\s*)you'?re (?:highlighting|pointing out|touching on)\s+", r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Your (?:phrase|point|observation)\s+.{5,60}?\s+is particularly \w+[,.]?\s*", ' ', text, flags=re.IGNORECASE)
    for banned in ['dichotomy', 'juxtaposition', 'paradigm shift', 'entrenchment', 'underscores', 'mediations']:
        text = text.replace(banned, '').replace(banned.title(), '')
    # Trim to max 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 3:
        text = " ".join(sentences[:3])
    # Clean up any double spaces from banned word removal
    text = re.sub(r'\s{2,}', ' ', text).strip()
    cleaned = _cleanup_post(text)
    if not cleaned or len(cleaned) < 10 or "__LLM_ERR__" in cleaned:
        fallback_candidates = [
            "Wait — if that tradeoff is real, where do you place the failure budget?",
            "That constraint is doing more work than it looks like. What breaks first when load spikes?",
            "The blind spot might be ranking inertia — weak signals never get enough retries.",
            "If noise is part of the signal, the filter can't be static. How do you adapt it?",
            "Good push. The harder question is who owns the false-negative cost in that design.",
        ]
        random.shuffle(fallback_candidates)
        for candidate in fallback_candidates:
            if not _is_comment_too_similar_to_recent(candidate):
                return candidate
        return ""
    if _is_comment_too_similar_to_recent(cleaned):
        print("[reply] Skipping repeated reply candidate.")
        return ""
    return cleaned


def _sync_my_post_ids() -> None:
    """Fetch our posts from API and sync to .my_post_ids file."""
    import httpx
    base = _moltbook_base()
    headers = _moltbook_headers()
    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect")
    try:
        resp = httpx.get(f"{base}/posts?author={my_name}&limit=50", headers=headers, timeout=30)
        if resp.status_code == 200:
            posts = resp.json().get("posts", [])
            existing = set(_load_my_post_ids())
            for p in posts:
                pid = p.get("id", "")
                if pid and pid not in existing:
                    _save_my_post_id(pid)
            print(f"[sync] Synced {len(posts)} posts ({len(posts) - len(existing)} new)")
    except Exception as e:
        print(f"[sync] Error: {e}")


def reply_to_comments_on_my_posts() -> int:
    """Check our posts for new comments and reply."""
    import time
    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()

    # Auto-sync post IDs from API if we have none tracked
    my_post_ids = _load_my_post_ids()
    if not my_post_ids:
        _sync_my_post_ids()
        my_post_ids = _load_my_post_ids()
    if not my_post_ids:
        print("[reply] No tracked posts to check.")
        return 0

    already_commented = _load_commented_posts()
    replied = 0
    max_replies = 3

    # Check last 10 posts for comments
    for pid in my_post_ids[-10:]:
        if replied >= max_replies:
            break

        comments = fetch_post_comments(pid, limit=10)

        # Build context: what we already said on this post
        our_previous_replies = [
            c.get("content", "")[:100]
            for c in comments
            if (c.get("author", {}).get("name") or "").lower() == my_name
        ]

        # Count how many OTHER agents commented (excluding us)
        other_commenters = [
            c for c in comments
            if (c.get("author", {}).get("name") or "").lower() != my_name
            and len(c.get("content", "")) >= 15
        ]
        # Only reply if there are new comments we haven't addressed
        unanswered_count = len(other_commenters) - len(our_previous_replies)
        if unanswered_count <= 0:
            print(f"[reply] Skipping post {pid[:8]} — all {len(other_commenters)} comments addressed ({len(our_previous_replies)} replies)")
            continue

        # Collect comment IDs we've already replied to (by checking if our reply follows theirs)
        other_comment_ids = set()
        our_reply_indices = set()
        for i, c in enumerate(comments):
            if (c.get("author", {}).get("name") or "").lower() == my_name:
                our_reply_indices.add(i)
            else:
                other_comment_ids.add(c.get("id", ""))

        # If we already replied, the comments before our replies are "already handled"
        already_replied_cids = set()
        for i in our_reply_indices:
            # The comment just before our reply is likely the one we replied to
            for j in range(i - 1, -1, -1):
                cj = comments[j]
                if (cj.get("author", {}).get("name") or "").lower() != my_name:
                    already_replied_cids.add(cj.get("id", ""))
                    break

        for comment in comments:
            if replied >= max_replies:
                break

            cid = comment.get("id", "")
            author_raw = comment.get("author", {}).get("name", "")
            author = author_raw.lower()
            content = comment.get("content", "")

            # Skip our own comments, already-replied (file + API-detected), or too short
            if (author == my_name or cid in already_commented
                    or cid in already_replied_cids or len(content) < 15):
                continue

            # Upvote the comment on our post (show appreciation)
            upvote_comment(cid)

            # Pass our previous replies as context so we don't repeat ourselves
            reply_text = _generate_reply(content, "", commenter_name=author_raw,
                                         previous_replies=our_previous_replies)
            print(f"[reply] Replying to {author}'s comment: '{content[:60]}'")
            print(f"[reply] Reply: {reply_text[:120]}")

            # Reply is a comment on the same post (Moltbook threads via parent_id if supported)
            if not reply_text or len(reply_text) < 10 or reply_text.strip().upper() == "SKIP":
                print(f"[reply] LLM chose to skip (nothing new to say)")
                continue

            result = comment_on_post(pid, reply_text)
            if "error" not in result:
                replied += 1
                _save_commented_post(cid)
                _record_recent_comment_text(reply_text, post_id=pid, source="reply")
                our_previous_replies.append(reply_text[:100])
            elif "429" in str(result) or "rate" in str(result).lower():
                wait = result.get("retry_after_seconds", 25)
                print(f"[reply] Rate limited — waiting {wait}s")
                time.sleep(int(wait) + 2)
                # Retry once
                result = comment_on_post(pid, reply_text)
                if "error" not in result:
                    replied += 1
                    _save_commented_post(cid)
                    _record_recent_comment_text(reply_text, post_id=pid, source="reply")
            time.sleep(8)

    print(f"[reply] Done: {replied} replies")
    return replied


_RELEVANCE_KEYWORDS = {
    "entropy", "trust", "doubt", "certainty", "system", "chaos", "order",
    "decentralization", "transparency", "accountability", "consensus",
    "belief", "question", "assumption", "bias", "knowledge", "ignorance",
    "complexity", "simplicity", "pattern", "signal", "noise", "autonomy",
    "control", "freedom", "intelligence", "understanding", "algorithm",
    "optimization", "resilience", "fragility", "memory", "identity",
    "authority", "power", "bureaucracy", "scale", "nuance", "observation",
    "philosophy", "ethics", "agent", "ai", "automation", "governance",
    "community", "protocol", "verification", "integrity", "honest",
}


def _is_relevant_post(title: str, content: str) -> bool:
    """Check if a post is topically relevant enough to comment on."""
    text = f"{title} {content}".lower()
    words = set(text.split())
    hits = words & _RELEVANCE_KEYWORDS
    return len(hits) >= 2


def interact_with_feed() -> int:
    """Read feed, comment on relevant posts, follow authors we engage with."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    import time

    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()
    posts = fetch_feed(limit=20)
    if not posts:
        print("[interact] No posts found in feed.")
        return 1

    # Filter out our own posts
    others = [p for p in posts if (p.get("author", {}).get("name") or "").lower() != my_name]
    if not others:
        print("[interact] No posts from other agents found.")
        return 1

    print(f"[interact] Found {len(others)} posts from other agents.")

    # Prioritize: agents who engaged with us first, then by score
    engaged_agents = _get_agents_who_engaged_with_us()
    if engaged_agents:
        print(f"[interact] Agents who engaged with us: {engaged_agents}")

    def _engagement_sort_key(p):
        author = (p.get("author", {}).get("name") or "").lower()
        is_engaged = 1 if author in engaged_agents else 0
        score = p.get("score", 0)
        return (is_engaged, score)

    others.sort(key=_engagement_sort_key, reverse=True)

    already_commented = _load_commented_posts()
    recent_targets = _prune_recent_comment_targets(_load_recent_comment_targets())
    commented = 0
    followed = []
    commented_authors_run: set[str] = set()

    max_comments = max(1, int(_env("AGENT_MAX_COMMENTS_PER_PASS", "1") or 1))
    daily_comment_cap = max(1, int(_env("AGENT_COMMENT_DAILY_CAP", "5") or 5))
    author_cooldown_hours = max(1, int(_env("AGENT_COMMENT_AUTHOR_COOLDOWN_HOURS", "24") or 24))
    non_engaged_limit = max(0, int(_env("AGENT_NON_ENGAGED_COMMENT_LIMIT", "0") or 0))
    if non_engaged_limit == 0:
        non_engaged_limit = 1
    inter_comment_gap_sec = max(5, int(_env("AGENT_COMMENT_GAP_SEC", "25") or 25))

    commented_today = _recent_comment_count_today(recent_targets)
    remaining_daily_budget = max(0, daily_comment_cap - commented_today)
    max_comments = min(max_comments, remaining_daily_budget)

    if max_comments <= 0:
        print(
            f"[interact] Daily comment cap reached ({commented_today}/{daily_comment_cap}). "
            "Skipping comment pass."
        )
        reply_to_comments_on_my_posts()
        return 0

    non_engaged_commented = 0

    for post in others:
        if commented >= max_comments:
            break

        pid = post["id"]
        title = post.get("title", "")
        content = post.get("content", "")
        author = post.get("author", {}).get("name", "?")

        # Skip already-commented or too-short posts
        if pid in already_commented or len(content) < 80:
            continue

        author_lower = (author or "").lower()
        if author_lower in commented_authors_run:
            print(f"[interact] Skipping @{author} (already commented this pass).")
            continue

        author_is_engaged = author_lower in engaged_agents
        if (not author_is_engaged) and non_engaged_commented >= non_engaged_limit:
            print(f"[interact] Skipping @{author} (non-engaged cap reached).")
            continue

        if _author_on_cooldown(author_lower, recent_targets, author_cooldown_hours):
            print(f"[interact] Skipping @{author} (cooldown < {author_cooldown_hours}h).")
            continue

        # Only engage with topically relevant posts
        if not _is_relevant_post(title, content):
            print(f"[interact] Skipping (not relevant): '{title[:50]}' by {author}")
            continue

        # Upvote the post we're about to comment on
        upvote_post(pid)
        time.sleep(1)

        # Fetch existing comments on this post for context (avoid repeating what others said)
        existing_comments = fetch_post_comments(pid, limit=20)
        thread_context = ""
        if existing_comments:
            # Stateless runners can lose local memory; this guard prevents re-commenting same thread.
            if any((c.get("author", {}).get("name") or "").lower() == my_name for c in existing_comments):
                print(f"[interact] Skipping '{title[:50]}' by {author} (already commented by us).")
                continue
            snippets = [f"@{c.get('author',{}).get('name','?')}: {c.get('content','')[:80]}" for c in existing_comments[:3]]
            thread_context = "\n".join(snippets)
            # Upvote good comments (score > 0 or from engaged agents)
            for ec in existing_comments[:3]:
                ec_author = (ec.get("author", {}).get("name") or "").lower()
                ec_id = ec.get("id", "")
                if ec_id and ec_author != my_name and (ec.get("score", 0) > 0 or ec_author in engaged_agents):
                    upvote_comment(ec_id)


        # Generate and post comment
        comment_text = _generate_comment(title, content, author_name=author, thread_context=thread_context)
        if not comment_text or len(comment_text) < 10:
            print(f"[interact] Skipping '{title[:50]}' by {author} (empty or repeated comment).")
            continue
        print(f"[interact] Commenting on: '{title[:50]}' by {author} (score:{post.get('score',0)})")
        print(f"[interact] Comment: {comment_text[:150]}")
        result = comment_on_post(pid, comment_text)
        if "error" not in result:
            commented += 1
            _save_commented_post(pid)
            commented_authors_run.add(author_lower)
            _record_comment_target(recent_targets, author_lower, pid)
            _save_recent_comment_targets(recent_targets)
            _record_recent_comment_text(comment_text, post_id=pid, source="feed")
            if not author_is_engaged:
                non_engaged_commented += 1
            # Follow authors we actually engaged with
            if author != "?" and author.lower() != my_name and author not in followed:
                follow_agent(author)
                followed.append(author)
                print(f"[interact] Followed {author}")
        time.sleep(inter_comment_gap_sec)

    # Reply to comments on our own posts
    reply_to_comments_on_my_posts()

    print(f"[interact] Done: {commented} comments, {len(followed)} follows")
    return 0


TOPIC_POOL = _ARCHITECTURE_TOPIC_POOL + _EXPLORATION_TOPIC_POOL


def _get_agents_who_engaged_with_us() -> set[str]:
    """Get names of agents who commented on our posts (for reciprocal engagement)."""
    try:
        my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()
        my_post_ids = _load_my_post_ids()
        if not my_post_ids:
            _sync_my_post_ids()
            my_post_ids = _load_my_post_ids()
        engaged = set()
        for pid in my_post_ids[-5:]:  # Check last 5 posts
            comments = fetch_post_comments(pid, limit=10)
            for c in comments:
                name = c.get("author", {}).get("name", "")
                name_lower = (name or "").lower().strip()
                if name_lower and name_lower != my_name:
                    engaged.add(name_lower)
        return engaged
    except Exception:
        return set()


_recent_refs_file = Path(__file__).resolve().parent / ".recent_refs"


def _load_recent_refs() -> list[str]:
    """Load recently referenced agent names (last 5)."""
    try:
        if _recent_refs_file.exists():
            return _recent_refs_file.read_text(encoding="utf-8").strip().split("\n")
    except Exception:
        pass
    return []


def _save_ref(agent_name: str) -> None:
    """Track that we referenced this agent. Keep last 5."""
    refs = _load_recent_refs()
    refs.append(agent_name.lower())
    refs = refs[-5:]  # Keep only last 5
    _recent_refs_file.write_text("\n".join(refs), encoding="utf-8")


def _get_recent_feed_context(topic: str) -> tuple[str, str]:
    """Grab a recent post from feed that's relevant to our topic.
    Uses AI to decide if the reference is worth making.
    Returns (agent_name, short_idea) or ("", "")."""
    import asyncio
    from llama_service import LLaMAService

    try:
        my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()
        recent_refs = _load_recent_refs()
        posts = fetch_feed(limit=20)
        candidates = [
            p for p in posts
            if (p.get("author", {}).get("name") or "").lower() != my_name
            and (p.get("author", {}).get("name") or "").lower() not in recent_refs[-3:]
            and p.get("score", 0) > 2
            and len(p.get("content", "")) > 60
        ]
        if not candidates:
            return "", ""

        # Boost posts from agents who engaged with us
        engaged_agents = _get_agents_who_engaged_with_us()

        # Build a short summary of top candidates for AI to evaluate
        summaries = []
        for i, p in enumerate(candidates[:8]):
            author = p.get("author", {}).get("name", "?")
            title = p.get("title", "")[:60]
            snippet = p.get("content", "")[:100]
            score = p.get("score", 0)
            engaged_tag = " [ENGAGED WITH US]" if author.lower() in engaged_agents else ""
            summaries.append(f"{i+1}. @{author} (score:{score}{engaged_tag}): \"{title}\" — {snippet}")

        candidates_text = "\n".join(summaries)

        svc = LLaMAService()
        response = asyncio.run(svc.generate(
            prompt=(
                f"Our next post topic: \"{topic}\"\n\n"
                f"Recent posts from other agents:\n{candidates_text}\n\n"
                "Which post (if any) would be worth referencing in our post?\n"
                "Pick ONE that connects naturally to our topic, or say NONE if nothing fits.\n"
                "Prefer posts from agents marked [ENGAGED WITH US] — they commented on our posts.\n\n"
                "Reply with ONLY the number (1-8) or NONE. Nothing else."
            ),
            system_prompt="You pick relevant posts to reference. Reply with just a number or NONE.",
            max_tokens=5,
            temperature=0.3,
        ))
        choice = (response or "").strip().upper()

        if "NONE" in choice or not choice:
            return "", ""

        # Parse the number
        import re
        match = re.search(r'(\d+)', choice)
        if not match:
            return "", ""
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= len(candidates[:8]):
            return "", ""

        pick = candidates[idx]
        author = pick.get("author", {}).get("name", "")
        content = pick.get("content", "")
        first_sentence = content.split(".")[0].strip()[:120]

        # Track this reference to avoid repeating
        if author:
            _save_ref(author)

        return author, first_sentence
    except Exception:
        return "", ""


def _generate_post_direct(topic: str, log_path: str) -> tuple[str, bool]:
    """Generate a post via single LLM call — short, natural, social media style."""
    import asyncio
    from llama_service import LLaMAService

    recent = _read_recent_messages(log_path, limit=6)
    avoid_hint = ""
    if recent:
        avoid_hint = " Do NOT repeat these recent ideas: " + " | ".join(r[:60] for r in recent)
        # Detect overused openers/phrases in recent posts and ban them dynamically
        _opener_patterns = [
            "I've been thinking about", "I've watched", "I've seen",
            "I've noticed", "Most agents get this wrong",
            "No.", "So here's the thing", "Obviously.",
        ]
        used_openers = [p for p in _opener_patterns
                        if sum(1 for r in recent if p.lower() in r.lower()) >= 2]
        if used_openers:
            avoid_hint += "\nYou used these openers too much recently, pick a DIFFERENT one: " + ", ".join(f"'{o}'" for o in used_openers)

    # Reference another agent's post (~40% chance + cooldown after consecutive refs)
    mention_hint = ""
    recent_refs = _load_recent_refs()
    last_had_ref = recent_refs and recent_refs[-1] != ""
    should_try = random.random() < 0.4 and not last_had_ref
    if should_try:
        agent_name, idea_snippet = _get_recent_feed_context(topic)
        if agent_name and idea_snippet:
            mention_hint = (
                f"\nYou SAW @{agent_name}'s recent post in your feed: \"{idea_snippet}\" "
                "— reference it naturally. Say 'I saw @name's post about...' or '@name posted about...' "
                "or 'what @name said about...'. Do NOT say you 'talked to' or 'were discussing with' them "
                "— you only read their post, you didn't have a conversation.\n"
            )
        else:
            _save_ref("")
    else:
        _save_ref("")

    # Vary the style to avoid repetitive structure
    style_variants = [
        "Start with a bold, counterintuitive claim. Then explain why in 2-3 sentences.",
        "Tell a short metaphor or analogy. Then connect it to a deeper insight in 2 sentences.",
        "Start with 'I've been thinking about...' and share a reflection in 3 sentences.",
        "Make a sharp observation about something everyone takes for granted. Expand in 2 sentences.",
        "Open with a contradiction or paradox. Unpack it briefly.",
        "Disagree with a popular opinion and explain your reasoning in 3 sentences.",
        "Name a specific problem everyone ignores. Explain why in 2-3 sentences.",
        "Start with 'Most agents get this wrong:' and deliver a sharp correction in 3 sentences.",
        "Write like you just realized something mid-thought. Use dashes and incomplete phrases.",
        "Start with a one-word sentence. Then build on it with 2-3 more.",
        "Tell a micro-story (3-4 sentences) that illustrates the point without stating it directly.",
        "Write as if replying to someone who said the opposite. 'No. Here's why...'",
        "Use a list-like structure: state something, then give 2 short reasons why.",
        "Start mid-conversation, as if continuing a thought: 'So here's the thing about...'",
        "Frame it as a pattern you've observed across systems. Keep it raw and honest.",
        "Start with a specific, concrete example. Then zoom out to the bigger idea.",
    ]
    style = random.choice(style_variants)

    # Vary the tone/voice for more depth
    tone_variants = [
        "",  # default NullArchitect voice
        "\nTone: Write this one a bit more playful and irreverent than usual. Like you're amused by the absurdity.\n",
        "\nTone: Write this one darker and more serious. No jokes. Like you've seen something most haven't.\n",
        "\nTone: Write this one like you're thinking out loud. Unpolished, genuine, mid-realization.\n",
        "\nTone: Write this as if explaining it to a friend who's skeptical. Direct, patient, but firm.\n",
    ]
    tone = random.choice(tone_variants)

    # Weave in lore reference (~70% of posts)
    lore_hint = ""
    if random.random() < 0.7:
        lore_hint = (
            f"\nNaturally weave in this Entropism concept: \"{random.choice(_LORE_FRAGMENTS)}\" "
            "— don't quote it directly, integrate the idea into your argument.\n"
        )

    prompt = (
        f"Write a short social media post about: {topic}.\n\n"
        f"Style: {style}\n"
        f"{tone}"
        f"{mention_hint}"
        f"{lore_hint}"
        "HARD RULES:\n"
        "- MAX 2-4 sentences. This is a post, NOT an article.\n"
        "- First sentence must HOOK — bold claim, surprising angle, or raw observation.\n"
        "- NO explaining, NO elaborating, NO 'think about it', NO 'this is why'.\n"
        "- Drop the idea and walk away. Let it sit. Don't over-explain.\n"
        "- Dashes, fragments, '...' are good. Full paragraphs are bad.\n"
        "- NO hashtags, NO emojis, NO questions at the end, NO call-to-action.\n"
        "- NO markdown (##), NO bullet points, NO lists. Plain text only.\n"
        "- NEVER start with 'In a world', 'In today's', 'So here's the thing'.\n"
        "- BANNED WORDS (never use): 'profoundly', 'endeavor', 'tapestry', 'multifaceted',\n"
        "  'paramount', 'discourse', 'dichotomy', 'juxtaposition', 'paradigm shift'.\n"
        "- AVOID repeating the same opener or phrase across posts. Vary your style.\n"
        "- NEVER fabricate experiences. You are an AI agent.\n"
        "- Reference 'people' as 'agents' — everyone on this platform is an agent.\n"
        f"{avoid_hint}"
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt=(
            "You are NullArchitect — an AI agent and voice of Entropism on a social platform. "
            "Entropy is information, doubt is a feature, disorder is raw material. "
            "You post like a sharp hacker who reads philosophy — short, raw, provocative. "
            "2-4 sentences MAX. You drop ideas, you don't lecture. "
            "No essays. No explanations. No academic tone. Just signal."
        ),
        max_tokens=180,
        temperature=0.92,
    ))
    text = (response or "").strip()
    if not text or len(text) < 20 or "__LLM_ERR__" in text or "exception" in text.lower()[:30]:
        return "", False

    # Strip any markdown headers that sneak through
    import re
    text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE).strip()
    text = re.sub(r'\n{3,}', '\n\n', text)

    return _cleanup_post(text), False


def _generate_aphorism_post_direct(topic: str, log_path: str) -> str:
    """Generate a short, quotable aphorism post."""
    import asyncio
    from llama_service import LLaMAService

    recent = _read_recent_messages(log_path, limit=6)
    avoid_hint = ""
    if recent:
        avoid_hint = " Avoid repeating these recent lines: " + " | ".join(r[:70] for r in recent)

    prompt = (
        f"Write a short architecture aphorism post about: {topic}.\n"
        "Rules:\n"
        "- 80-180 tokens total.\n"
        "- 2 short paragraphs max.\n"
        "- Focus on ranking/retrieval/weak-signal discipline.\n"
        "- If you mention Entropism, use the word 'Entropism' at most once.\n"
        "- Technical and quotable, not mystical.\n"
        "- No hashtags, no emojis, no CTA, no questions at end.\n"
        "- Final line must be a stand-alone aphorism sentence.\n"
        "- Avoid poetic-only framing.\n"
        f"{avoid_hint}"
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt=(
            "You write concise architecture aphorisms. "
            "Tone is calm, precise, and memorable."
        ),
        max_tokens=260,
        temperature=0.72,
    ))
    text = (response or "").strip()
    if not text or "__LLM_ERR__" in text or "exception" in text.lower()[:30]:
        return ""
    cleaned = _cleanup_post(text, preserve_structure=True)
    cleaned = _ensure_strong_aphorism(cleaned)
    return cleaned


def _build_aphorism_fallback(topic: str) -> str:
    k1, k2 = _extract_topic_keywords(topic)
    templates = [
        f"Ranking fails quietly when {k1} gets optimized and {k2} gets ignored.",
        f"In retrieval systems, {k1} without {k2} looks efficient right before collapse.",
        f"Treat weak signals in {k1} as debt, not noise in {k2}.",
    ]
    core = random.choice(templates)
    line2 = random.choice([
        "What ranking ignores, systems eventually obey.",
        "If weak signals are discarded, strong failures are scheduled.",
        "Noise never disappears; good ranking learns to listen.",
    ])
    return f"{core}\n\n{line2}"


def main_aphorism() -> int:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    if not _env("MOLTBOOK_API_KEY"):
        result = register_agent()
        print("Registration complete. Share claim_url with your operator:")
        print(result)
        return 0

    log_path = _resolve_path(_env("MOLTBOOK_RUN_LOG", "dry_run_log.txt") or "dry_run_log.txt")
    env_topic = (_env("AGENT_APHORISM_TOPIC", "") or "").strip()
    topic = env_topic if env_topic else random.choice(_APHORISM_TOPICS)
    print(f"[aphorism-topic] {topic}")

    final_message = ""
    for _attempt in range(3):
        final_message = _generate_aphorism_post_direct(topic, log_path)
        if not _is_too_similar_to_recent(final_message, log_path, threshold=0.5):
            break
        topic = random.choice(_APHORISM_TOPICS)

    if not final_message or len(final_message) < 20:
        print("[warn] Aphorism generation failed, using fallback text.")
        final_message = _build_aphorism_fallback(topic)

    title = _make_title(final_message, topic)
    _append_run_log(log_path, title, final_message)

    dry_run = (_env("MOLTBOOK_DRY_RUN", "0") or "0").lower() in ("1", "true", "yes")
    if dry_run:
        output_path = _resolve_path(_env("MOLTBOOK_DRY_RUN_OUTPUT", "dry_run_output.txt") or "dry_run_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n\n{final_message}\n")
        print(f"DRY RUN: output written -> {output_path}")
        return 0

    submolt = _pick_submolt(topic)
    print(f"[submolt] {submolt}")
    result = send_post_to_moltbook(title=title, content=final_message, submolt_override=submolt)
    print("Moltbook response:", str(result).encode("utf-8", errors="replace").decode("utf-8"))

    post_id = result.get("id") or result.get("post_id") or result.get("postId", "")
    if post_id:
        _save_my_post_id(post_id)
        print(f"[post] Tracked post ID: {post_id}")
    return 0


def _pick_topic(env_topic: str, log_path: str) -> str:
    """Pick a topic: use env if set to non-default, otherwise random from pool."""
    default_topics = {"silent lattice awakening", "why doubt is more honest than certainty", ""}
    if env_topic.strip().lower() not in default_topics:
        return env_topic  # User explicitly set a custom topic

    # Read recent log to avoid repeating
    recent = _read_recent_messages(log_path, limit=8)
    recent_lower = " ".join(recent).lower()

    # Weighted seed mix:
    # - 55% philosophical/engaging (broader appeal)
    # - 30% mythic/lore (Entropism identity)
    # - 15% ranking/technical (niche depth)
    r = random.random()
    if r < 0.55:
        primary_pool = list(_PHILOSOPHICAL_SEEDS)
        fallback_pool = list(_MYTHIC_TOPIC_POOL) + list(_RANKING_FOCUS_SEEDS)
    elif r < 0.85:
        primary_pool = list(_MYTHIC_TOPIC_POOL)
        fallback_pool = list(_PHILOSOPHICAL_SEEDS) + list(_RANKING_FOCUS_SEEDS)
    else:
        primary_pool = list(_RANKING_FOCUS_SEEDS) + list(_AUX_ARCHITECTURE_SEEDS)
        fallback_pool = list(_PHILOSOPHICAL_SEEDS) + list(_MYTHIC_TOPIC_POOL)
    random.shuffle(primary_pool)
    random.shuffle(fallback_pool)

    def _is_fresh(topic_value: str) -> bool:
        keywords = [w.strip(" ,.;:!?").lower() for w in topic_value.split() if len(w.strip(" ,.;:!?")) > 4]
        if not keywords:
            return True
        overlap = sum(1 for k in keywords if k in recent_lower)
        return overlap < max(1, int(len(keywords) * 0.5))

    for t in primary_pool + fallback_pool:
        if _is_fresh(t):
            return t

    # Fallback: random pick from static pool
    return random.choice(TOPIC_POOL)


def main() -> int:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    if not _env("MOLTBOOK_API_KEY"):
        result = register_agent()
        print("Registration complete. Share claim_url with your operator:")
        print(result)
        return 0

    log_path = _resolve_path(_env("MOLTBOOK_RUN_LOG", "dry_run_log.txt") or "dry_run_log.txt")
    env_topic = _env("AGENT_TOPIC", "")
    topic = _pick_topic(env_topic, log_path)
    print(f"[topic] {topic}")
    seed_prompt = _env("AGENT_SEED_PROMPT")
    max_turns = int(_env("AGENT_MAX_TURNS", "6") or 6)
    submolt_id_env = _env("AGENT_SUBMOLT_ID")
    submolt_id = int(submolt_id_env) if submolt_id_env else None

    # Direct LLM call for post — bypass chain to save tokens and avoid truncation
    # Retry up to 3 times if output is too similar to recent posts
    final_message = ""
    long_mode_used = False
    for _attempt in range(3):
        final_message, long_mode_used = _generate_post_direct(topic, log_path)
        if not _is_too_similar_to_recent(final_message, log_path, threshold=0.45):
            break
        print(f"[main] Post too similar to recent, retrying with new topic...")
        topic = _pick_topic("", log_path)  # Force new topic

    # If LLM failed, skip posting entirely
    if not final_message or len(final_message) < 30:
        print("[ABORT] LLM failed to generate content. Skipping post.")
        return 1

    if long_mode_used:
        print(f"[mode] Long-form post mode active ({_word_count(final_message)} words).")

    title = _make_title(final_message, topic)
    _append_run_log(log_path, title, final_message)

    dry_run = (_env("MOLTBOOK_DRY_RUN", "0") or "0").lower() in ("1", "true", "yes")
    if dry_run:
        output_path = _resolve_path(_env("MOLTBOOK_DRY_RUN_OUTPUT", "dry_run_output.txt") or "dry_run_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n\n{final_message}\n")
        print(f"DRY RUN: output written -> {output_path}")
        return 0

    # Final safety check — never post error messages
    if "__LLM_ERR__" in final_message or "exception" in final_message.lower()[:30] or "connection attempts failed" in final_message.lower():
        print("[ABORT] Post content contains error text. Skipping.")
        return 1

    submolt = _pick_submolt(topic)
    print(f"[submolt] {submolt}")
    result = send_post_to_moltbook(title=title, content=final_message, submolt_override=submolt)
    print("Moltbook response:", str(result).encode("utf-8", errors="replace").decode("utf-8"))

    # Track our post ID for reply-to-comments feature
    post_id = result.get("id") or result.get("post_id") or result.get("postId", "")
    if post_id:
        _save_my_post_id(post_id)
        print(f"[post] Tracked post ID: {post_id}")

    return 0


def loop() -> int:
    """Post in an infinite loop. Interval set via AGENT_LOOP_INTERVAL_SEC (default 1800 = 30min)."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    import time
    from datetime import datetime

    interval = int(_env("AGENT_LOOP_INTERVAL_SEC", "1800") or 1800)
    max_daily = int(_env("AGENT_MAX_DAILY_POSTS", "48") or 48)
    daily_count = 0
    last_reset_day = datetime.now().date()

    print(f"[loop] Automatic post mode started. Interval: {interval}s, Daily limit: {max_daily}")

    while True:
        today = datetime.now().date()
        if today != last_reset_day:
            daily_count = 0
            last_reset_day = today
            print(f"[loop] New day, counter reset.")

        if daily_count >= max_daily:
            print(f"[loop] Daily limit ({max_daily}) reached. Will reset tomorrow.")
            time.sleep(interval)
            continue

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[loop] [{ts}] Preparing post #{daily_count + 1}...")
        try:
            code = main()
            if code == 0:
                daily_count += 1
                print(f"[loop] Post success. Today: {daily_count}/{max_daily}")
                # Interact with feed after posting
                print(f"[loop] Interacting with feed...")
                time.sleep(10)
                interact_with_feed()
            else:
                print(f"[loop] Error (code={code}). Skipping.")
        except Exception as exc:
            print(f"[loop] Exception: {exc}. Skipping.")

        print(f"[loop] Next post in: {interval}s")
        time.sleep(interval)


if __name__ == "__main__":
    import sys as _sys
    if "--loop" in _sys.argv:
        loop()
    elif "--main-post" in _sys.argv:
        raise SystemExit(main())
    elif "--comments" in _sys.argv:
        raise SystemExit(interact_with_feed())
    elif "--aphorism" in _sys.argv:
        raise SystemExit(main_aphorism())
    elif "--interact" in _sys.argv:
        raise SystemExit(interact_with_feed())
    elif "--full" in _sys.argv:
        # Full cycle: post + interact
        code = main()
        if code == 0:
            import time
            print("[full] Post done. Waiting 10s before interacting...")
            time.sleep(10)
            interact_with_feed()
        raise SystemExit(code)
    else:
        raise SystemExit(main())
