"""Low-level text helpers used across the chain engine.

No dependency on schemas, models, or any other project module.
"""

import json
import re
from typing import Optional


def split_sentences(text: str) -> list[str]:
    parts = []
    current = ""
    for ch in text:
        current += ch
        if ch in ".!?":
            parts.append(current.strip())
            current = ""
    if current.strip():
        parts.append(current.strip())
    return parts


def trim_to_sentences(text: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def contains_banned(text: str) -> bool:
    banned = [
        "moltbook",
        "api",
        "python",
        "ben bir botum",
        "analiz ediyorum",
        "elden ring",
    ]
    lowered = text.lower()
    return any(term in lowered for term in banned)


def contains_first_person(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered.split() for token in ("i", "me", "my", "mine"))


def contains_turkish(text: str) -> bool:
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    if any(ch in turkish_chars for ch in text):
        return True
    low = (text or "").strip().lower()
    # Single-word or short inputs that are uniquely Turkish (no English meaning)
    _definite_turkish = (
        "merhaba", "selam", "naber", "nasilsin", "nasilsiniz", "hosgeldin",
        "tesekkurler", "sagol", "evet", "hayir", "tamam", "peki",
        "gunaydın", "gunaydin", "iyi geceler", "iyi aksamlar",
        "nesin", "kimsin", "nereden", "nerelisin",
        "kac", "bugun", "nerede", "soyle", "nasil",
    )
    import re as _re
    if any(_re.search(rf"\b{_re.escape(w)}\b", low) for w in _definite_turkish):
        return True
    # Also detect ASCII Turkish by checking common Turkish words
    low = (text or "").lower()
    turkish_words = (
        # Temel baglaçlar ve edatlar
        "bir", "ile", "icin", "nasil", "neden", "niye", "ama", "veya", "eger",
        "olan", "gibi", "kadar", "sonra", "once", "daha", "cok", "hic",
        "var", "yok", "hem", "iste", "zaten", "ancak", "yani", "fakat",
        # Zamirler ve soru kelimeleri
        "bana", "sana", "bunu", "sunu", "onun", "bunun", "nedir", "misin", "midir",
        # Yaygın fiiller (ASCII)
        "yaz", "ver", "gel", "git", "bak", "bul", "yap", "olan", "eden",
        "yapilir", "olarak", "olmak", "etmek", "vermek", "almak",
        "bulmak", "hafife", "anlam", "gucunu", "alma",
        # Yaygın sifatlar ve isimler
        "guzel", "gercek", "degil", "buyuk", "kucuk", "yeni", "eski",
        "hayat", "dunya", "insan", "zaman", "bilgi",
        # Gunluk kelimeler
        "tamam", "peki", "evet", "hayir", "selam", "merhaba", "tekrar",
        "kisaca", "bakalim", "simdi", "bugun", "biraz", "acaba", "galiba", "belki",
        # Post / konu kelimeleri
        "hakkinda", "arasinda", "arasindaki", "uzerine", "konusu", "kavram",
        "ilke", "temel", "denge", "adalet", "topluluk", "korku", "ozgurluk",
        "postu", "gonder", "paylas", "baslik",
        # Ek yaygın kelimeler
        "kusura", "bakma", "baskan", "yazilimci", "kediler", "felsefe",
        "olsa", "olurdu",
        # Soru edatlari ve yaygın ekli kelimeler
        "misin", "musun", "nedir", "neden", "nasil",
        "tavuk", "yumurta", "cikar", "gelir", "gider", "olur", "olmaz",
        "sence", "bence", "herkes", "kimse", "hicbir", "birsey", "bisey",
        "canim", "canin", "sikildi", "sohbet", "edelim", "yapalim", "gidelim",
        "yapay", "zeka", "robot", "makine", "bilgisayar",
        "yemek", "kahve", "cay", "su", "ekmek", "para", "okul", "araba",
        "anne", "baba", "kardes", "arkadas", "dost", "sevgi", "mutlu", "uzgun",
        "dogru", "yanlis", "onemli", "kolay", "zor", "hizli", "yavas",
        "burada", "orada", "buraya", "oraya", "neresi", "burasi", "orasi",
    )
    import re as _re
    hits = sum(1 for w in turkish_words if _re.search(rf"\b{_re.escape(w)}\b", low))
    if hits >= 2:
        return True
    # Turkish question suffixes: standalone mi/mu/mı/mü are strong Turkish signals
    question_particles = _re.findall(r"\b(mi|mu|mı|mü)\b", low)
    if len(question_particles) >= 1 and hits >= 1:
        return True
    if len(question_particles) >= 2:
        return True
    return False


def word_count(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


def has_system_tag(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("["):
        return False
    close_idx = stripped.find("]")
    return close_idx > 2 and close_idx < 40


def normalize_for_similarity(text: str) -> list[str]:
    cleaned = []
    for token in text.lower().replace("\n", " ").split():
        t = token.strip(" ,.;:!?()[]{}\"'")
        if len(t) >= 4:
            cleaned.append(t)
    return cleaned


def token_overlap_ratio(a: str, b: str) -> float:
    sa = set(normalize_for_similarity(a))
    sb = set(normalize_for_similarity(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(max(len(sa), len(sb)))


def extract_keywords(text: str, limit: int = 3) -> list[str]:
    tokens = [
        w.strip(" ,.;:!?()[]{}\"'").lower()
        for w in text.replace("\n", " ").split()
    ]
    stop = {
        "the", "and", "or", "of", "to", "in", "a", "an", "is", "are", "for", "on", "with",
        "that", "this", "from", "into", "your", "their", "have", "will", "must", "node", "system",
        "user", "users", "about", "challenges", "challenge", "test", "reply", "give", "actionable",
        "command", "contextual", "doctrinal", "rebuttal", "state", "concrete", "consequence",
        "write", "moltbook", "post", "thread", "gonderi", "paylasim", "create", "draft",
        "topic", "please", "should", "would", "could", "using", "think", "make",
        "matters", "important", "explain", "describe", "discuss", "does", "related",
        "what", "why", "how", "when", "where", "which", "there", "here", "very", "just",
        "also", "some", "more", "most", "much", "many", "each", "every", "other", "only",
    }
    keywords = [w for w in tokens if w and w not in stop and len(w) > 3 and not w.isdigit()]
    seen: list[str] = []
    for w in keywords:
        if w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def extract_json_object(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        chunk = raw[start : end + 1]
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def token_count_heuristic(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+", normalize_whitespace(text or "")))
