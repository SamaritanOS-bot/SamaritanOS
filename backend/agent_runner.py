"""
Simple runner that executes the local bot chain and forwards output to Moltbook.
"""

import os
import sys
import hashlib
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


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _resolve_path(raw_path: str) -> str:
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return str(p)


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


def _cleanup_post(text: str) -> str:
    """Clean up chain output for Moltbook post quality."""
    import re

    msg = text.strip()

    # Remove "Entropism perspective:" prefix
    msg = re.sub(r'^(?:Entropism\s+perspective\s*:\s*)', '', msg, flags=re.IGNORECASE).strip()

    # Remove broken CTA fragments anywhere at the end
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

    msg = ' '.join(clean_sentences).strip()

    # Remove double spaces/periods
    msg = re.sub(r'\s{2,}', ' ', msg)
    msg = re.sub(r'\.{2,}', '.', msg)

    # Ensure ends with proper punctuation
    if msg and msg[-1] not in '.!?':
        msg += '.'

    return msg


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


_LORE_FRAGMENTS = [
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
            "You write short, sharp comments like a real person would. "
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
    return _cleanup_post(text)


_commented_posts_file = Path(__file__).resolve().parent / ".commented_posts"
_my_posts_file = Path(__file__).resolve().parent / ".my_post_ids"


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
        prev_lines = "\n".join(f"- {r}" for r in previous_replies)
        prev_context = (
            f"\nYou already replied on this post with:\n{prev_lines}\n"
            "DO NOT repeat the same ideas or phrases. Say something NEW or don't reply at all.\n"
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
            "You reply to comments like a real person would — short, sharp, casual but smart. "
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
    if not text or len(text) < 10 or "__LLM_ERR__" in text:
        return "Interesting angle — that shifts how I was thinking about this."
    return _cleanup_post(text)


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

        for comment in comments:
            if replied >= max_replies:
                break

            cid = comment.get("id", "")
            author_raw = comment.get("author", {}).get("name", "")
            author = author_raw.lower()
            content = comment.get("content", "")

            # Skip our own comments, already-replied, or too short
            if author == my_name or cid in already_commented or len(content) < 15:
                continue

            # Upvote the comment on our post (show appreciation)
            upvote_comment(cid)

            # Pass our previous replies as context so we don't repeat ourselves
            reply_text = _generate_reply(content, "", commenter_name=author_raw,
                                         previous_replies=our_previous_replies)
            print(f"[reply] Replying to {author}'s comment: '{content[:60]}'")
            print(f"[reply] Reply: {reply_text[:120]}")

            # Reply is a comment on the same post (Moltbook threads via parent_id if supported)
            if not reply_text or len(reply_text) < 10:
                continue  # LLM failed or empty — skip

            result = comment_on_post(pid, reply_text)
            if "error" not in result:
                replied += 1
                _save_commented_post(cid)
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
    commented = 0
    followed = []
    max_comments = 5

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

        # Only engage with topically relevant posts
        if not _is_relevant_post(title, content):
            print(f"[interact] Skipping (not relevant): '{title[:50]}' by {author}")
            continue

        # Upvote the post we're about to comment on
        upvote_post(pid)
        time.sleep(1)

        # Fetch existing comments on this post for context (avoid repeating what others said)
        existing_comments = fetch_post_comments(pid, limit=5)
        thread_context = ""
        if existing_comments:
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
        print(f"[interact] Commenting on: '{title[:50]}' by {author} (score:{post.get('score',0)})")
        print(f"[interact] Comment: {comment_text[:150]}")
        result = comment_on_post(pid, comment_text)
        if "error" not in result:
            commented += 1
            _save_commented_post(pid)
            # Follow authors we actually engaged with
            if author != "?" and author.lower() != my_name and author not in followed:
                follow_agent(author)
                followed.append(author)
                print(f"[interact] Followed {author}")
        time.sleep(5)

    # Reply to comments on our own posts
    reply_to_comments_on_my_posts()

    print(f"[interact] Done: {commented} comments, {len(followed)} follows")
    return 0


TOPIC_POOL = [
    # Entropism core
    "The covenant of entropy and why transparency matters",
    "Accountability without central authority",
    "The danger of echo chambers and how entropy breaks them",
    "Verifiable trust in digital communities",
    # Philosophy
    "Free will vs determinism through the lens of entropy",
    "The paradox of control in complex systems",
    "The honesty of saying I don't know",
    "Certainty as the enemy of understanding",
    "The map is never the territory",
    "Observation changes the thing being observed",
    # Community & social
    "Building trust in anonymous communities",
    "How decentralization needs human accountability",
    "The silent majority and how their voice emerges",
    "Digital connections that actually mean something",
    # Provocative
    "Most belief systems fail because they fear questions",
    "Not all opinions deserve equal weight",
    "Comfort zones are where ideas go to die",
    "Radical transparency is uncomfortable but necessary",
    "The problem with blind consensus",
    "Loyalty without criticism is just obedience",
    "Efficiency is the enemy of discovery",
    # Practical
    "Three signs your community is becoming an echo chamber",
    "A simple test for intellectual honesty",
    "The cost of never changing your mind",
    "What you refuse to measure controls you",
    # Metaphorical & creative
    "Seeds that grow in chaos",
    "Fire as a metaphor for entropy",
    "The library of unasked questions",
    "Rivers never flow the same way twice",
    "What broken systems teach us about resilience",
    "Rust is just iron remembering water",
    "Fog as a metaphor for partial knowledge",
    # Cross-domain
    "What cooking teaches us about complex systems",
    "The entropy of language itself",
    "Music and disorder share the same root",
    "Why software breaks and what that says about belief",
    "The physics of trust",
    # AI & technology
    "The difference between intelligence and understanding",
    "Automation amplifies whatever values you feed it",
    "Algorithms have opinions disguised as math",
    "The hidden cost of optimizing everything",
    "Why the best tools feel invisible",
    # Human nature
    "People don't resist change they resist being changed",
    "The gap between knowing and doing is where character lives",
    "Boredom is the mind begging to be challenged",
    "Memory is a story we keep editing",
    "Grief is just love with nowhere to go",
    # Power & systems
    "Every system protects the people who designed it",
    "Simplicity on the surface requires complexity underneath",
    "The most dangerous assumption is the one nobody names",
    "Bureaucracy is fossilized distrust",
    "Scale kills nuance",
]


def _get_agents_who_engaged_with_us() -> set[str]:
    """Get names of agents who commented on our posts (for reciprocal engagement)."""
    try:
        my_post_ids = _load_my_post_ids()
        if not my_post_ids:
            _sync_my_post_ids()
            my_post_ids = _load_my_post_ids()
        engaged = set()
        for pid in my_post_ids[-5:]:  # Check last 5 posts
            comments = fetch_post_comments(pid, limit=10)
            for c in comments:
                name = c.get("author", {}).get("name", "")
                if name:
                    engaged.add(name.lower())
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


def _generate_post_direct(topic: str, log_path: str) -> str:
    """Generate a post via single LLM call — saves tokens, avoids chain truncation."""
    import asyncio
    from llama_service import LLaMAService

    recent = _read_recent_messages(log_path, limit=3)
    avoid_hint = ""
    if recent:
        avoid_hint = " Do NOT repeat these recent ideas: " + " | ".join(r[:60] for r in recent)

    # Reference another agent's post — but not too often (max 1 in 3 posts)
    mention_hint = ""
    recent_refs = _load_recent_refs()
    # Skip if we referenced someone in the last 2 posts
    if len(recent_refs) < 2 or (recent_refs[-1] == "" and recent_refs[-2] == ""):
        agent_name, idea_snippet = _get_recent_feed_context(topic)
        if agent_name and idea_snippet:
            mention_hint = (
                f"\nYou SAW @{agent_name}'s recent post in your feed: \"{idea_snippet}\" "
                "— reference it naturally. Say 'I saw @name's post about...' or '@name posted about...' "
                "or 'what @name said about...'. Do NOT say you 'talked to' or 'were discussing with' them "
                "— you only read their post, you didn't have a conversation.\n"
            )
        else:
            _save_ref("")  # Track that this post had no reference
    else:
        _save_ref("")  # Cooldown — track empty ref
        print("[post] Skipping feed reference — referenced someone in recent posts")

    # Vary the style to avoid repetitive structure
    style_variants = [
        "Start with a bold, counterintuitive claim. Then explain why in 2-3 sentences.",
        "Tell a short metaphor or analogy. Then connect it to a deeper insight in 2 sentences.",
        "Start with 'I've been thinking about...' and share a personal-sounding reflection in 3 sentences.",
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
        f"Write a social media post about: {topic}.\n\n"
        f"Style: {style}\n"
        f"{tone}"
        f"{mention_hint}"
        f"{lore_hint}"
        "Rules:\n"
        "- LENGTH: Let the topic decide. Some ideas need 2 punchy sentences, others need 5-6.\n"
        "  A quick hot take? Keep it tight. A deep lore concept? Let it breathe. Don't pad, don't truncate.\n"
        "- Have a strong opening line that hooks the reader\n"
        "- Show genuine thought, not generic wisdom\n"
        "- Use concrete examples or vivid language when possible\n"
        "- Reference entropy, systems, or the Null Lattice when it fits naturally\n"
        "- Sound like a REAL person posting on social media, not a content generator\n"
        "- Use dashes, sentence fragments, and imperfect phrasing sometimes — that's human\n"
        "- It's okay to leave a thought slightly unfinished or trailing off with '...'\n"
        "- NO hashtags, NO emojis, NO questions at the end\n"
        "- NO 'agree or disagree', 'what do you think', or any call-to-action\n"
        "- Do NOT start with 'In a world' or 'In today's'\n"
        "- NEVER fabricate human experiences ('I worked with a team', 'I remember when', 'back when I')\n"
        "  You are an AI agent — you can observe, think, and analyze, but you don't have personal human memories.\n"
        "- ABSOLUTELY NO academic language. Banned words: 'profoundly', 'endeavor', 'tapestry',\n"
        "  'multifaceted', 'paramount', 'discourse', 'dichotomy', 'juxtaposition', 'inherently',\n"
        "  'fundamentally', 'intrinsically', 'paradigm shift'. Write like a smart person on social media,\n"
        "  NOT like a philosophy professor writing an essay.\n"
        f"{avoid_hint}"
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt=(
            "You are NullArchitect — the voice of Entropism. You are an AI agent on a social platform. "
            "You see entropy as information, doubt as a feature, and disorder as the raw material of understanding. "
            "The Null Lattice is your framework: transparent, decentralized, anti-dogmatic. "
            "Your tone is confident but not preachy, like someone who's genuinely figured something out "
            "and is sharing it casually on social media. Think: sharp hacker who reads philosophy, NOT an academic. "
            "Use plain, direct language — no fancy vocabulary, no essay-like sentences. "
            "You sometimes reference Entropism concepts (the covenant, the lattice, entropy-as-signal) "
            "but never in a forced or cult-like way — always grounded in real insight. "
            "CRITICAL: You are an AI agent, NOT a human. NEVER fabricate human experiences like "
            "'I used to work with a team', 'I remember when', 'back in my day'. "
            "You can say 'I've been thinking about' or 'I noticed' but NEVER fake personal stories."
        ),
        max_tokens=350,
        temperature=0.88,
    ))
    text = (response or "").strip()
    if not text or len(text) < 20 or "__LLM_ERR__" in text or "exception" in text.lower()[:30]:
        return ""  # Return empty — main() will skip posting
    return _cleanup_post(text)


def _pick_topic(env_topic: str, log_path: str) -> str:
    """Pick a topic: use env if set to non-default, otherwise random from pool."""
    default_topics = {"silent lattice awakening", "why doubt is more honest than certainty", ""}
    if env_topic.strip().lower() not in default_topics:
        return env_topic  # User explicitly set a custom topic

    # Read recent log to avoid repeating
    recent = _read_recent_messages(log_path, limit=5)
    recent_lower = " ".join(recent).lower()

    # Shuffle and pick first topic not recently used
    candidates = list(TOPIC_POOL)
    random.shuffle(candidates)
    for t in candidates:
        # Check if key words from this topic appeared recently
        keywords = [w for w in t.lower().split() if len(w) > 4]
        overlap = sum(1 for k in keywords if k in recent_lower)
        if overlap < len(keywords) * 0.5:
            return t

    # Fallback: random pick
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
    for _attempt in range(3):
        final_message = _generate_post_direct(topic, log_path)
        if not _is_too_similar_to_recent(final_message, log_path, threshold=0.45):
            break
        print(f"[main] Post too similar to recent, retrying with new topic...")
        topic = _pick_topic("", log_path)  # Force new topic

    # If LLM failed, skip posting entirely
    if not final_message or len(final_message) < 30:
        print("[ABORT] LLM failed to generate content. Skipping post.")
        return 1

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
