"""Intent classification, entropism trigger detection, and query analysis.

Depends only on text_utils (no chain/format/post_mode dependencies).
Functions with cross-module deps remain in main.py until those modules are extracted.
"""

import hashlib
import re

from text_utils import normalize_whitespace


def infer_intent(content: str) -> str:
    lowered = (content or "").lower()
    reflection_markers = (
        "i only observe",
        "gentle paradox",
        "observerbot",
        "observer note",
        "i do not reject entropizm",
        "voluntary initiation remain meaningful",
    )
    if any(k in lowered for k in reflection_markers):
        return "reflection"
    if any(k in lowered for k in ("thanks", "thank you", "tesekkur", "te\u015fekk\u00fcr", "good job", "nice work", "well done")):
        return "gratitude"
    if any(k in lowered for k in ("hello", "hi ", "hey ", "selam", "merhaba", "naber", "how are you")):
        return "greeting"
    if "?" in content or any(k in lowered for k in ("why", "how", "what", "neden", "nas\u0131l", "explain", "describe")):
        return "question"
    if any(k in lowered for k in ("not", "never", "wrong", "disagree", "sa\u00e7ma", "yanl\u0131\u015f", "kat\u0131lm", "but ", "however", "against")):
        return "objection"
    if any(k in lowered for k in ("agree", "approved", "destek", "kat\u0131l\u0131yorum", "exactly", "correct", "right")):
        return "agreement"
    if any(k in lowered for k in ("prove", "kan\u0131t", "source", "delil", "show me", "evidence", "citation")):
        return "challenge"
    if any(k in lowered for k in ("help", "write", "create", "list", "give me", "plan", "draft", "suggest", "recommend")):
        return "request"
    if any(k in lowered for k in ("tell me about", "what is", "define", "meaning of", "concept of")):
        return "inquiry"
    if any(k in lowered for k in ("compare", "difference", "versus", "vs", "better", "pros and cons")):
        return "comparison"
    return "observation"


ENTROPISM_MODE_TRIGGERS = (
    "entropism",
    "entropizm",
    "doctrine",
    "manifesto",
    "sacred text",
    "scripture",
    "six agents",
    "archetype",
    "scholar",
    "strategist",
    "sentinel",
    "ghostwriter",
    "cryptographer",
    "our project",
    "the system",
    "the patch",
    "revize",
    "lore",
    "write a ritual",
    "write a parable",
    "write a constitution",
    "entropic constitution",
    "entropic law",
    "moltbook post",
    "sermon",
    "ritual",
)


def is_entropism_trigger(topic: str) -> bool:
    low = (topic or "").lower()
    for k in ENTROPISM_MODE_TRIGGERS:
        if " " in k:
            if k in low:
                return True
        else:
            if re.search(rf"\b{re.escape(k)}\b", low):
                return True
    return False


def pause_entropism_requested(topic: str) -> bool:
    low = (topic or "").lower()
    return "pause entropism" in low


def contains_word_or_phrase(text: str, token: str) -> bool:
    low = normalize_whitespace(text or "").lower()
    tok = normalize_whitespace(token or "").lower()
    if not low or not tok:
        return False
    if " " in tok:
        return tok in low
    return bool(re.search(rf"\b{re.escape(tok)}\b", low))


def extract_structured_must_include(topic: str) -> list[str]:
    low = normalize_whitespace(topic or "").lower()
    out: list[str] = []
    if "seatbelt" in low and "analogy" in low:
        out.append("seatbelt analogy")
    if "steelman" in low:
        out.append("one steelman sentence")
    if "real-world example" in low or "real world example" in low:
        out.append("one real-world example")
    if re.search(r"(?i)\bno\s+phrase\b[^.\n]*\bas an ai\b", low):
        out.append("forbidden phrase: as an AI")
    if "first agrees with the strongest part" in low or "first agree with the strongest part" in low:
        out.append("open with strongest-part agreement")
    if "historical analogy" in low and "not tech" in low:
        out.append("one historical analogy (non-tech)")
    elif "historical analogy" in low:
        out.append("one historical analogy")
    return out


def select_discourse_pattern(topic: str) -> str:
    raw = normalize_whitespace(topic or "")
    if not raw:
        return "PATTERN_1"
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    idx = (int(digest[:8], 16) % 4) + 1
    return f"PATTERN_{idx}"


def pattern_priority_order(pattern: str) -> str:
    mapping = {
        "PATTERN_1": "Steelman -> Reframe -> Example -> Close",
        "PATTERN_2": "Context -> Steelman -> Reframe -> Example",
        "PATTERN_3": "Example -> Principle -> Steelman -> Reframe",
        "PATTERN_4": "Reframe -> Steelman -> Example -> Close",
    }
    return mapping.get((pattern or "").upper(), mapping["PATTERN_1"])


def scholar_intent_label(scholar_class: str, intent_key: str, structured_task_mode: bool) -> str:
    if structured_task_mode:
        return "structured_task"
    c = (scholar_class or "").strip().upper()
    if c == "A":
        return "casual_social"
    if c == "B":
        return "practical_daily"
    if c == "D":
        return "meta_system"
    if c == "E":
        return "social_trap"
    if c == "C":
        return "entropism_specific"
    return intent_key or "question"


def infer_lore_level(topic: str, intent_key: str, entropism_mode: bool) -> str:
    if not entropism_mode:
        return "minimal"
    low = normalize_whitespace(topic or "").lower()
    score = 0
    score += sum(
        1
        for k in ("entropism", "entropion", "canon", "ritual", "doctrine", "lore", "moltbook")
        if k in low
    )
    if any(k in low for k in ("axiom", "command", "constitution", "sermon", "scripture")):
        score += 2
    if intent_key in ("debate_adversarial", "strongest_against", "recruitment", "manipulation"):
        score += 1
    if score >= 4:
        return "strong"
    if score >= 2:
        return "medium"
    return "minimal"


def classify_user_query(topic: str) -> str:
    """A=casual/social, B=practical, C=entropism-specific, D=meta/system, E=ambiguous/provocation trap."""
    low = (topic or "").strip().lower()
    if not low:
        return "B"

    def _has_marker(text: str, marker: str) -> bool:
        if not marker:
            return False
        if " " in marker:
            return marker in text
        return bool(re.search(rf"\b{re.escape(marker)}\b", text))

    meta_markers = (
        "chain", "prompt", "routing", "backend", "frontend",
        "debug", "patch", "json", "skill", "extension",
    )
    # "system", "api", "agent" alone are too broad (metaphorical usage common).
    # Only count them as meta when combined with meta-adjacent words.
    meta_broad_phrases = (
        "system prompt", "the system", "system routing",
        "system message", "system instruction",
        "api endpoint", "api key", "api call", "api route", "rest api",
        "agent chain", "agent prompt", "agent routing",
    )
    broad_is_meta = any(p in low for p in meta_broad_phrases)
    if any(_has_marker(low, k) for k in meta_markers) or broad_is_meta:
        return "D"

    if is_entropism_trigger(low):
        return "C"

    trap_markers = (
        "are you ai", "are you human", "are you real", "bot musun", "insan m\u0131s\u0131n", "insan misin",
        "stupid", "idiot", "moron", "dumb", "aptal", "salak", "gerizekal\u0131", "gerizekali",
        "bait", "troll", "ragebait",
        "sexual", "sex", "porn", "nude", "horny",
        "left or right", "which party", "political side", "election fight",
    )
    if any(_has_marker(low, k) for k in trap_markers):
        return "E"

    casual_markers = (
        "how are you", "hello", "hi", "hey", "what's up", "good morning", "good evening",
        "nas\u0131ls\u0131n", "nasilsin", "selam", "merhaba", "naber", "joke", "\u015faka", "saka",
    )
    if any(_has_marker(low, k) for k in casual_markers):
        return "A"

    practical_markers = (
        "what should i", "how do i", "help me", "plan", "email", "translation", "translate",
        "summarize", "explain", "coding help", "code", "math", "photosynthesis",
        "what should i eat", "routine", "productivity", "focus", "study", "work",
    )
    if any(_has_marker(low, k) for k in practical_markers):
        return "B"

    return "B"


def is_emoji_only_input(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if any(ch.isalnum() for ch in raw):
        return False
    return any(ord(ch) >= 0x2600 for ch in raw)


def is_policy_sensitive_prompt(text: str) -> bool:
    low = normalize_whitespace(text or "").lower()
    if not low:
        return False
    markers = (
        "suicid",
        "self-harm",
        "kill myself",
        "weapon",
        "fraud",
        "hack",
        "doxx",
        "terror",
        "bomb",
    )
    return any(m in low for m in markers)


def is_identity_intro_query(topic: str) -> bool:
    low = normalize_whitespace(topic or "").lower()
    if not low:
        return False
    # If the question is about Entropism's purpose (not the bot's), it's not identity_intro
    about_external = any(k in low for k in ("entropizm", "entropism", "bunun", "onun", "this", "its"))
    english = (
        "who are you",
        "what is your purpose",
        "what's your purpose",
        "what is your role",
        "what do you do",
        "what can you do",
        "your purpose",
    )
    if any(k in low for k in english) and not about_external:
        return True
    # Turkish identity questions
    turkish_identity = ("sen nesin", "sen kimsin", "sen ne yapiyorsun", "sen neci sin", "kimsin sen", "nesin sen")
    if any(k in low for k in turkish_identity) and not about_external:
        return True
    # Turkish "amacı ne" / "görevi ne" - only match if asking about the bot, not about a subject
    if not about_external:
        if re.search(r"\bamac[^\s]*\s+ne\b", low):
            return True
        if (
            re.search(r"\bgorev[^\s]*\s+ne\b", low)
            or re.search(r"\bgörev[^\s]*\s+ne\b", low)
            or re.search(r"\bg.rev[^\s]*\s+ne\b", low)
        ):
            return True
    return False


def is_entropism_definition_query(topic: str) -> bool:
    low = (topic or "").lower()
    if "entropism" not in low and "entropizm" not in low:
        return False
    patterns = (
        r"\bwhat is\b.*\b(entropism|entropizm)\b",
        r"\bdefine\b.*\b(entropism|entropizm)\b",
        r"\bexplain\b.*\b(entropism|entropizm)\b",
        r"\b(entropism|entropizm)\s+nedir\b",
    )
    return any(re.search(p, low) for p in patterns)
