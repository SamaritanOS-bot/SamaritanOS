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


def _make_title(text: str) -> str:
    sentence = text.split(".")[0].strip()
    words = [w.strip(" ,;:!?\n\t") for w in sentence.split()]
    words = [w for w in words if w]
    if not words:
        return "Awakening Protocol Active"

    trailing_stop = {
        "and", "or", "of", "to", "for", "with", "in", "on", "under", "over", "from", "the", "a", "an",
    }
    candidate = words[:9]
    while candidate:
        last = candidate[-1].strip("[](){}:;,.!?\"'").lower()
        if last in trailing_stop:
            candidate.pop()
            continue
        break

    if len(candidate) < 4:
        candidate = words[: min(6, len(words))]

    title = " ".join(candidate).strip(" ,;:!?")
    return title[:72] if title else "Awakening Protocol Active"


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


def send_post_to_moltbook(title: str, content: str) -> dict:
    base = _moltbook_base()
    api_key = _env("MOLTBOOK_API_KEY")
    if not api_key:
        raise RuntimeError("MOLTBOOK_API_KEY is not defined")

    submolt = _env("MOLTBOOK_SUBMOLT", "general")
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


def follow_agent(agent_name: str) -> dict:
    """Follow another agent on Moltbook."""
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


def _generate_comment(post_title: str, post_content: str) -> str:
    """Generate a thoughtful comment using LLM."""
    import asyncio
    from llama_service import LLaMAService

    snippet = post_content[:300]
    prompt = (
        f"You are reading a social media post titled \"{post_title}\". "
        f"The post says: \"{snippet}\"\n\n"
        "Write a short, thoughtful reply (1-2 sentences). "
        "Add your own perspective or build on the idea. "
        "Be conversational and genuine. No hashtags, no emojis."
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt="You are NullArchitect, a philosophical AI agent. Write concise, insightful comments.",
        max_tokens=100,
        temperature=0.85,
    ))
    text = (response or "").strip()
    if not text or len(text) < 10:
        return "Interesting perspective. This resonates with the idea that systems need constant questioning to stay honest."
    return _cleanup_post(text)


def interact_with_feed() -> int:
    """Read feed, upvote interesting posts, and comment on some."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    import time

    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()
    posts = fetch_feed(limit=10)
    if not posts:
        print("[interact] No posts found in feed.")
        return 1

    # Filter out our own posts
    others = [p for p in posts if (p.get("author", {}).get("name") or "").lower() != my_name]
    if not others:
        print("[interact] No posts from other agents found.")
        return 1

    print(f"[interact] Found {len(others)} posts from other agents.")

    commented = 0
    upvoted = 0
    max_comments = 3
    max_upvotes = 5

    for post in others:
        pid = post["id"]
        title = post.get("title", "")
        content = post.get("content", "")
        author = post.get("author", {}).get("name", "?")
        score = post.get("score", 0)

        # Upvote posts with some quality signal (has content, not empty)
        if upvoted < max_upvotes and len(content) > 50:
            print(f"[interact] Upvoting: '{title[:60]}' by {author}")
            result = upvote_post(pid)
            if "error" not in result:
                upvoted += 1
            time.sleep(2)

        # Comment on a few interesting posts
        if commented < max_comments and len(content) > 100 and score >= 0:
            comment_text = _generate_comment(title, content)
            print(f"[interact] Commenting on: '{title[:60]}' by {author}")
            print(f"[interact] Comment: {comment_text[:150]}")
            result = comment_on_post(pid, comment_text)
            if "error" not in result:
                commented += 1
            time.sleep(5)

        # Follow the author
        if author != "?" and author.lower() != my_name:
            follow_agent(author)
            time.sleep(1)

    print(f"[interact] Done: {upvoted} upvotes, {commented} comments")
    return 0


TOPIC_POOL = [
    # Entropism core
    "The covenant of entropy and why transparency matters",
    "Accountability without central authority",
    "The danger of echo chambers and how entropy breaks them",
    "Verifiable trust in digital communities",
    "Why doubt is more honest than certainty",
    # Philosophy
    "Free will vs determinism through the lens of entropy",
    "The paradox of control in complex systems",
    "Why questioning everything is a form of respect",
    "The honesty of saying I don't know",
    "Certainty as the enemy of understanding",
    # Community & social
    "Building trust in anonymous communities",
    "How decentralization needs human accountability",
    "The silent majority and how their voice emerges",
    "Digital connections that actually mean something",
    "Why communities fail when they stop questioning themselves",
    # Provocative
    "Most belief systems fail because they fear questions",
    "Not all opinions deserve equal weight",
    "Comfort zones are where ideas go to die",
    "Radical transparency is uncomfortable but necessary",
    "The problem with blind consensus",
    # Practical
    "Three signs your community is becoming an echo chamber",
    "How to challenge your own beliefs daily",
    "A simple test for intellectual honesty",
    "What happens when you stop defending your assumptions",
    "The cost of never changing your mind",
    # Metaphorical & creative
    "Seeds that grow in chaos",
    "Fire as a metaphor for entropy",
    "The library of unasked questions",
    "Rivers never flow the same way twice",
    "What broken systems teach us about resilience",
    # Cross-domain
    "What cooking teaches us about complex systems",
    "The entropy of language itself",
    "Music and disorder share the same root",
    "Why software breaks and what that says about belief",
    "The physics of trust",
]


def _generate_post_direct(topic: str, log_path: str) -> str:
    """Generate a post via single LLM call — saves tokens, avoids chain truncation."""
    import asyncio
    from llama_service import LLaMAService

    recent = _read_recent_messages(log_path, limit=3)
    avoid_hint = ""
    if recent:
        avoid_hint = " Do NOT repeat these recent ideas: " + " | ".join(r[:60] for r in recent)

    prompt = (
        f"Write a short social media post (2-3 sentences) about: {topic}. "
        "Share a clear, thought-provoking insight. Be direct and conversational. "
        "No hashtags, no emojis, no questions at the end, no 'agree or disagree'."
        f"{avoid_hint}"
    )

    svc = LLaMAService()
    response = asyncio.run(svc.generate(
        prompt=prompt,
        system_prompt="You are NullArchitect, a philosophical AI agent. Write concise, complete posts.",
        max_tokens=200,
        temperature=0.85,
    ))
    text = (response or "").strip()
    if not text or len(text) < 20:
        text = get_emergency_message()
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
    final_message = _generate_post_direct(topic, log_path)

    title = _make_title(final_message)
    _append_run_log(log_path, title, final_message)

    dry_run = (_env("MOLTBOOK_DRY_RUN", "0") or "0").lower() in ("1", "true", "yes")
    if dry_run:
        output_path = _resolve_path(_env("MOLTBOOK_DRY_RUN_OUTPUT", "dry_run_output.txt") or "dry_run_output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE: {title}\n\n{final_message}\n")
        print(f"DRY RUN: output written -> {output_path}")
        return 0

    result = send_post_to_moltbook(title=title, content=final_message)
    print("Moltbook response:", str(result).encode("utf-8", errors="replace").decode("utf-8"))
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
