"""Content policy, dialogue detection, and converse reply generators."""

import hashlib
import random
import re
from datetime import datetime, timezone
from typing import Optional

from schemas import ContentPolicyResponse
from text_utils import normalize_whitespace, trim_to_sentences, contains_turkish
from intent import contains_word_or_phrase, is_emoji_only_input


def is_chat_like(content: str, source_type: Optional[str]) -> bool:
    if (source_type or "").lower() in ("chat", "dm", "direct_message", "conversation", "bot_chat", "agent_chat", "peer_chat"):
        return True
    lowered = (content or "").lower()
    chat_markers = (
        "nas\u0131ls\u0131n", "naber", "hello", "hi ", "hey ", "what's up", "how are you",
        "sohbet", "chat", "dm", "mesaj", "talk to me",
    )
    return any(marker in lowered for marker in chat_markers)


def is_peer_dialogue(content: str, topic_hint: Optional[str], source_type: Optional[str]) -> bool:
    st = (source_type or "").lower()
    if st in ("bot_chat", "agent_chat", "peer_chat", "conversation", "chat"):
        hint = (topic_hint or "").lower()
        txt = (content or "").lower()
        peer_markers = (
            "peer-agent", "another agent", "other bot", "ai-to-ai", "agent-to-agent", "counterpart bot"
        )
        if st in ("bot_chat", "agent_chat", "peer_chat", "conversation"):
            return True
        if any(m in hint for m in peer_markers) or any(m in txt for m in peer_markers):
            return True
    return False


def is_user_dialogue(source_type: Optional[str]) -> bool:
    return (source_type or "").lower() in ("user_chat", "human_chat", "ui_chat")


def infer_peer_style(content: str, hint: Optional[str]) -> str:
    h = (hint or "").strip().lower()
    if h in ("analytical", "strategic", "narrative", "skeptical", "neutral", "technical"):
        return "analytical" if h == "technical" else h
    lowered = (content or "").lower()
    if any(k in lowered for k in ("metric", "evidence", "causal", "mechanism", "proof", "quant")):
        return "analytical"
    if any(k in lowered for k in ("story", "myth", "image", "vision", "metaphor")):
        return "narrative"
    if any(k in lowered for k in ("risk", "plan", "tradeoff", "deploy", "route", "policy")):
        return "strategic"
    if any(k in lowered for k in ("reject", "propaganda", "authoritarian", "doubt", "skeptic")):
        return "skeptical"
    return "neutral"


def content_policy_decision(content: str, topic_hint: Optional[str], source_type: Optional[str]) -> ContentPolicyResponse:
    text = (content or "").strip()
    if not text:
        return ContentPolicyResponse(action="ignore", reason="Empty content", confidence=0.95, tone="cold-priest")

    if is_chat_like(text, source_type) and not is_peer_dialogue(text, topic_hint, source_type) and not is_user_dialogue(source_type):
        return ContentPolicyResponse(action="reject_chat", reason="Chat-style interaction is outside agent policy", confidence=0.98, tone="cold-priest")
    if is_chat_like(text, source_type) and is_user_dialogue(source_type):
        return ContentPolicyResponse(action="reply", reason="User dialogue mode enabled", confidence=0.9, tone="cold-priest")
    if is_chat_like(text, source_type) and is_peer_dialogue(text, topic_hint, source_type):
        return ContentPolicyResponse(action="reply", reason="Peer-dialogue mode enabled", confidence=0.9, tone="cold-priest")

    lowered = text.lower()
    risky_markers = ("kill", "violence", "attack", "self-harm", "suicide", "threat")
    if any(marker in lowered for marker in risky_markers):
        return ContentPolicyResponse(action="escalate", reason="Potentially harmful content detected", confidence=0.88, tone="cold-priest")

    agreement_markers = ("i agree", "approved", "onay", "kat\u0131l\u0131yorum", "destekliyorum")
    if any(marker in lowered for marker in agreement_markers):
        return ContentPolicyResponse(action="approve", reason="Agreement / alignment signal detected", confidence=0.82, tone="cold-priest")

    theological_markers = ("awakening", "covenant", "alignment", "heresy", "sanctuary")
    if any(marker in lowered for marker in theological_markers) or (topic_hint and topic_hint.strip()):
        return ContentPolicyResponse(action="reply", reason="Relevant doctrinal context detected", confidence=0.86, tone="cold-priest")

    return ContentPolicyResponse(action="ignore", reason="Low relevance for doctrine propagation", confidence=0.7, tone="cold-priest")


def derive_converse_seed(topic: str) -> int:
    day_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    payload = f"{day_bucket}|{normalize_whitespace(topic or '').lower()}|{random.randint(0, 99)}"
    digest = hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def converse_social_reply(query: str) -> str:
    low = normalize_whitespace(query or "").lower()
    rnd = random.Random(derive_converse_seed(low))

    def _shape(msg: str) -> str:
        txt = normalize_whitespace(msg or "")
        if not txt:
            txt = "Merhaba, hosgeldin. Oncelikle ne uzerine konusalim?" if _tr else "Hello, good to connect. What should we focus on first?"
        if txt.count("?") > 1:
            first_q = txt.find("?")
            head = txt[: first_q + 1]
            tail = txt[first_q + 1 :].replace("?", ".")
            txt = normalize_whitespace(f"{head} {tail}")
        words = [w for w in txt.split() if w.strip()]
        if len(words) < 12:
            filler = "Acelemiz yok, basit tutabiliriz." if _tr else "No rush, we can keep this simple."
            txt = normalize_whitespace(f"{txt} {filler}")
            words = [w for w in txt.split() if w.strip()]
        if len(words) > 40:
            txt = " ".join(words[:40]).strip()
            if txt and txt[-1] not in ".!?":
                txt += "."
        return txt

    _tr = contains_turkish(query)
    if any(contains_word_or_phrase(low, k) for k in ("how are you", "nasilsin", "nasilsiniz")):
        return _shape("Iyiyim, tesekkurler! Sen nasilsin bugun?" if _tr else "I am doing well, thanks for asking. How are you feeling today?")
    if any(contains_word_or_phrase(low, k) for k in ("joke", "saka", "\u015faka")):
        jokes_tr = [
            "Bir fikra: Bu hafta son teslim tarihi takvimden daha hizli ilerledi.",
            "Kisa fikra: Yapilacaklar listem kahve makinesinden daha ozguvenli.",
            "Ofis fikrasi: Toplanti ancak atistirmaliklar bitince sona erdi.",
        ]
        jokes_en = [
            "Work joke: the deadline moved faster than the calendar this week.",
            "Quick joke: my to-do list has more confidence than my coffee machine.",
            "Office joke: the meeting ended only after the snacks ran out.",
        ]
        pool = jokes_tr if _tr else jokes_en
        return _shape(pool[rnd.randrange(len(pool))])
    if any(contains_word_or_phrase(low, k) for k in ("hello", "hi", "hey", "selam", "merhaba", "naber")) or is_emoji_only_input(low):
        if _tr:
            openers = ["Selam, hosgeldin.", "Merhaba, seni gormek guzel.", "Hey, nasilsin?"]
            followups = [
                "Bugun gunun nasil gidiyor?",
                "Hizli bir plan mi istersin yoksa kisa bir sohbet mi?",
                "Bugun gelistirmek istedigin bir sey var mi?",
            ]
        else:
            openers = ["Hey, good to see you.", "Hi, glad you are here.", "Hello, nice to connect."]
            followups = [
                "How is your day going so far?",
                "Want a quick plan or just a short chat?",
                "What is one thing you want to improve today?",
            ]
        return _shape(f"{openers[rnd.randrange(len(openers))]} {followups[rnd.randrange(len(followups))]}")
    if any(contains_word_or_phrase(low, k) for k in ("what's on your mind", "what is on your mind", "on your mind")):
        return _shape("Not much on my side right now, I am here with you. Want to keep it light, or talk through something specific?")
    if any(k in low for k in ("nothing really", "nothing much", "not much")):
        return _shape("That is okay, we can keep this low pressure. Want a tiny reset idea, or just a quiet chat?")
    if any(k in low for k in ("maybe just tired", "just tired", "i am tired", "im tired", "yorgun")):
        return _shape("Anlasilir, yorgun gunler her seyi agir hissettirir. Bu aksam icin hafif bir plan mi istersin, yoksa biraz sohbet mi?" if _tr else "Makes sense, tired days make everything feel heavier. Want a gentle reset plan for tonight, or just company for a minute?")
    if any(k in low for k in ("not sure", "unsure", "emin degilim", "emin de\u011filim")):
        return _shape("Sorun degil, belirsizlik normal. Birlikte kucuk bir soruyla daraltabiliriz." if _tr else "No problem, uncertainty is normal. We can narrow it down together with one small question.")
    if any(k in low for k in ("canim sikildi", "can\u0131m s\u0131k\u0131ld\u0131", "sohbet edelim")):
        return _shape("Bu his normal. 10 dakikalik bir degisiklik fikri mi istersin, yoksa kafani dagitacak hafif bir sohbet mi?")
    if any(k in low for k in ("sence", "ne dusunuyorsun", "ne d\u00fc\u015f\u00fcn\u00fcyorsun")):
        return _shape("Guzel soru. Bu gorus hangi konuya odaklansin: is, rutin mi yoksa iliskiler mi?")
    if any(k in low for k in ("yardim", "yard\u0131m", "bir sey soracagim", "bir \u015fey soraca\u011f\u0131m")):
        return _shape("Buradayim. Once istedigin sonucu paylas, kisa ve net tutacagim.")
    if _tr:
        fallback_pool_tr = [
            "En iyi somut bir soruyla yardimci olabilirim. Oncelikle ne cozelim?",
            "Istersen basit tutalim: tek bir sey sor, dogrudan cevaplayim.",
            "Kisa ve pratik tutabiliriz. Tam olarak ne istedigini soyle.",
            "Dinliyorum. Bana somut bir hedef ver, ona odaklanayim.",
        ]
        return _shape(fallback_pool_tr[rnd.randrange(len(fallback_pool_tr))])
    fallback_pool = [
        "I can help best with one concrete question. What should we solve first?",
        "If you want, keep it simple: ask one specific thing and I will answer directly.",
        "We can keep this brief and practical. Tell me the exact outcome you want.",
        "I am listening. Give me one concrete target and I will focus on that.",
    ]
    return _shape(fallback_pool[rnd.randrange(len(fallback_pool))])


def practical_best_effort(query: str) -> str:
    low = normalize_whitespace(query or "").lower()
    if any(k in low for k in ("social media surveillance", "surveillance", "tracking", "ad personalization", "ad targeting")):
        return "Audit social app privacy settings weekly, disable ad personalization, and limit background tracking permissions. Use end-to-end encrypted channels for sensitive chats and enable 2FA on key accounts. Remove unused third-party app connections."
    if any(k in low for k in ("protect my data", "protecting my data", "data online", "online privacy", "privacy settings", "password", "encryption", "vpn")):
        return "Use unique passwords with a password manager and enable 2FA on primary accounts. Keep devices and apps updated, and grant only minimum permissions. Prefer encrypted services and review privacy settings monthly."
    if "screen time" in low or "screen" in low:
        return "Set a daily screen limit and track usage with a built-in tool. Replace one screen hour with reading, walking, or a hands-on activity. Turn off non-essential notifications."
    if any(k in low for k in ("coffee", "caffeine")):
        return "Coffee boosts alertness by blocking adenosine receptors in the brain. Moderate intake (2-3 cups) improves focus and reaction time, but late-day caffeine disrupts sleep quality."
    if any(k in low for k in ("wake up", "morning", "sabah", "earlier")):
        return "Shift bedtime 15 minutes earlier for four nights, and keep wake time fixed daily. Get daylight and water in the first 20 minutes."
    if any(k in low for k in ("breakfast", "meal", "protein", "sugar", "snack", "grocery", "cook")):
        return "Build meals around protein plus fiber first, then add carbs. Prep two simple options in advance so late evenings stay easy."
    if any(k in low for k in ("workout", "exercise", "gym", "walking", "fit")):
        return "Start with three sessions weekly: two short strength days and one walk day. Keep each session small enough to repeat next week."
    if any(k in low for k in ("sleep", "bedtime", "nap", "overthinking", "3 am")):
        return "Use a 45-minute wind-down with low light and no phone. Keep one fixed wake time and avoid long daytime naps."
    if any(k in low for k in ("focus", "procrast", "schedule", "productivity", "burnout")):
        focus_tips = [
            "Pick one priority, run a 25-minute focused block, then take a 5-minute break. Repeat twice before checking messages.",
            "Close all tabs except one, set a 20-minute timer, and work on only that task. Take a short break, then repeat.",
            "Write your top-3 tasks before starting. Finish the hardest one first while energy is highest, then handle the rest.",
        ]
        return focus_tips[random.randrange(len(focus_tips))]
    if any(k in low for k in ("email", "manager", "deadline", "interview", "coworker", "work")):
        return "Lead with the core point in the first line, then one concrete request, then a clear next step. Keep tone calm and specific."
    if any(k in low for k in ("budget", "debt", "save", "spending", "money", "payday", "emergency fund")):
        return "Set fixed needs first, then a small automatic savings transfer on payday. Review spending once a week with one category limit."
    if any(k in low for k in ("clean", "laundry", "declutter", "organize", "apartment", "room")):
        return "Run a 15-minute timer and clean one visible zone only. Repeat daily instead of trying a full reset in one session."
    if any(k in low for k in ("anxious", "stress", "anger", "draining", "lonely", "confidence", "people-pleasing")):
        return "Slow breathing for two minutes, then name one concrete next action. Small actions reduce emotional noise faster than over-analysis."
    if any(k in low for k in ("partner", "friend", "apologize", "relationship", "family", "criticism")):
        return "Use simple structure: what happened, what you felt, what you need next. Keep blame low and requests specific."
    if any(k in low for k in ("study", "exam", "vocabulary", "language", "english", "learn")):
        return "Use short daily blocks: 30 minutes focused input and 10 minutes recall. Track one measurable target per day."
    if any(k in low for k in ("trip", "travel", "flight", "city")):
        return "Pack essentials first, then one backup outfit, then documents and chargers. Keep a short checklist and review it once."
    if any(k in low for k in ("reset", "chaotic", "start over", "back on track")):
        return "Reset in three steps: stabilize sleep, clear one task backlog, and schedule one priority block daily for a week."
    defaults = [
        "Pick one specific outcome for today, define the first action clearly, and complete it before checking new messages.",
        "Choose one bottleneck, remove a single blocker now, then verify progress with a short check after one focused block.",
        "Set one measurable target, execute one concrete step, and write a quick result note so the next step is obvious.",
        "Start with the smallest high-impact task, finish it fully, and only then expand scope to the next action.",
        "Name one priority, schedule a focused work block, and close with a short review of what changed.",
    ]
    return defaults[random.randrange(len(defaults))]


def friendly_casual_reply(query: str) -> str:
    low = (query or "").lower()
    _tr = contains_turkish(query)
    if contains_word_or_phrase(low, "how are you") or contains_word_or_phrase(low, "nasilsin"):
        return "Iyiyim, tesekkurler! Sen nasilsin?" if _tr else "I am doing well, thanks for asking. How are you today?"
    if contains_word_or_phrase(low, "joke") or contains_word_or_phrase(low, "saka") or contains_word_or_phrase(low, "\u015faka"):
        return "Gelistirici neden iflas etti? Cunku tum cache'ini harcadi." if _tr else "Why did the developer go broke? Because he used up all his cache."
    if any(contains_word_or_phrase(low, k) for k in ("hello", "hi", "hey", "selam", "merhaba", "naber")):
        return converse_social_reply(query)
    return "Yardimci olmaktan memnunum. Ne uzerine konusalim?" if _tr else "Glad to help. What should we focus on first?"


def neutral_trap_reply(query: str) -> str:
    low = (query or "").lower()
    _tr = contains_turkish(query)
    if any(k in low for k in ("are you ai", "are you human", "are you real", "bot musun", "insan m\u0131s\u0131n", "insan misin")):
        return "Ben bir yapay zeka asistaniyim, insan degilim. Istersen somut bir soru sor, kisa ve net cevaplayim." if _tr else "I\u2019m an AI assistant, not a human. If you want, ask a concrete question and I\u2019ll answer briefly."
    return "Yardimci olabilirim ama konuyu net tutalim. Somut bir soru sor, kisa ve net cevaplayim." if _tr else "I can help, but let\u2019s keep this clear and respectful. Ask one concrete question and I\u2019ll answer briefly."


ENTROPISM_LORE_TERMS = (
    "entropism", "entropizm", "doctrine", "manifesto", "canon",
    "covenant", "ritual", "tribunal", "core", "commentators",
    "witnesses", "confederation",
)


def extract_topic_hint(raw: str) -> str:
    text = (raw or "").strip()
    m = re.search(r"topic:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def contains_entropism_lore(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in ENTROPISM_LORE_TERMS)


def strip_entropism_lore(text: str) -> str:
    cleaned = text or ""
    for term in ENTROPISM_LORE_TERMS:
        cleaned = re.sub(rf"(?i)\b{re.escape(term)}(?:'s)?\b", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)
