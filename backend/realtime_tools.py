# -*- coding: utf-8 -*-
"""Lightweight real-time data helpers: time, weather, web search.

No API keys required. All functions return plain-text context strings
that can be injected into LLM prompts.
"""

import datetime
import re
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Time / Date
# ---------------------------------------------------------------------------

def get_current_datetime(tz_name: str = "Europe/Istanbul") -> dict:
    """Return current date/time info as a dict."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = None
    now = datetime.datetime.now(tz=tz)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "day_of_week": now.strftime("%A"),
        "timezone": tz_name,
        "iso": now.isoformat(),
    }


def format_datetime_context(tz_name: str = "Europe/Istanbul") -> str:
    """Human-readable datetime string for prompt injection."""
    info = get_current_datetime(tz_name)
    return (
        f"Current date: {info['date']} ({info['day_of_week']}), "
        f"Time: {info['time']} ({info['timezone']})"
    )


# ---------------------------------------------------------------------------
# Weather  (wttr.in - free, no key)
# ---------------------------------------------------------------------------

def fetch_weather(location: str = "Istanbul", lang: str = "tr") -> Optional[str]:
    """Fetch a short weather summary from wttr.in."""
    safe_loc = re.sub(r"[^a-zA-Z0-9\s,]", "", location).strip() or "Istanbul"
    url = f"https://wttr.in/{safe_loc}?format=j1&lang={lang}"
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "curl/7.68.0"})
            if r.status_code != 200:
                return None
            data = r.json()
    except Exception:
        return None

    try:
        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        feels = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        desc_list = current.get("lang_tr", current.get("weatherDesc", [{}]))
        if isinstance(desc_list, list) and desc_list:
            desc = desc_list[0].get("value", "")
        else:
            desc = ""
        wind_kmph = current.get("windspeedKmph", "?")

        # Forecast (today + tomorrow)
        forecasts = data.get("weather", [])
        forecast_lines = []
        for fc in forecasts[:2]:
            date = fc.get("date", "")
            max_t = fc.get("maxtempC", "?")
            min_t = fc.get("mintempC", "?")
            forecast_lines.append(f"{date}: {min_t}C - {max_t}C")

        result = (
            f"Weather for {safe_loc}: {desc}, {temp_c}C (feels like {feels}C), "
            f"humidity {humidity}%, wind {wind_kmph} km/h."
        )
        if forecast_lines:
            result += " Forecast: " + "; ".join(forecast_lines) + "."
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Web Search  (DuckDuckGo - free, no key)
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 3, region: str = "tr-tr") -> Optional[str]:
    """Run a DuckDuckGo search and return a compact summary string."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return None

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, region=region, max_results=max_results))
        # Fix encoding: ensure proper UTF-8 strings
        results = []
        for r in raw_results:
            fixed = {}
            for k, v in r.items():
                if isinstance(v, str):
                    try:
                        fixed[k] = v.encode("latin-1").decode("utf-8")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        fixed[k] = v
                else:
                    fixed[k] = v
            results.append(fixed)
    except Exception:
        return None

    if not results:
        return None

    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        href = (r.get("href") or "").strip()
        if title and body:
            lines.append(f"{i}. {title}: {body[:200]}")
    if not lines:
        return None
    return "Web search results:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Query detection: does the user need real-time data?
# ---------------------------------------------------------------------------

_TIME_PATTERNS = (
    r"\bsaat\s*(ka[cç]|kac|ne)\b",
    r"\bsaat\b.*\b(ka[cç]|kac)\b",
    r"\bwhat time\b",
    r"\bcurrent time\b",
    r"\bsaat\s*ne\b",
    r"\bbugun\s*(tarih|gun|ne)\b",
    r"\bbug[uü]n\s*(tarih|gun|ne)\b",
    r"\bgun\s*ne\b",
    r"\bhangi\s*gun\b",
    r"\bwhat day\b",
    r"\bwhat date\b",
    r"\btoday'?s?\s*date\b",
    r"\btarih\s*ne\b",
    r"\btarih\s*ka[cç]\b",
    r"\btarih\s*kac\b",
)

_WEATHER_PATTERNS = (
    r"\bhava\s*(durumu|nasil|nas[ıi]l|nas[iı]l)\b",
    r"\bhava\b.*\b(sicak|soguk|s[ıi]cak|so[gğ]uk|yagmur|ya[gğ]mur|kar|derece)\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\btemperature\b",
    r"\bhava\b.*\bnasil\b",
    r"\bdisarisi\s*nasil\b",
    r"\bdisarida\b.*\b(sicak|soguk)\b",
    r"\bhava\s*durumu\b",
)

_SEARCH_PATTERNS = (
    # Explicit search commands
    r"\b(ara|arat|bak|search|google|look\s*up)\b",
    # News / current events
    r"\bson\s*(haberler|gelismeler|dakika)\b",
    r"\bgundeme?\b",
    r"\blatest\s*news\b",
    r"\bcurrent\s*events?\b",
    r"\brecent\b.*\bnews\b",
    r"\bneler\s*oluyor\b",
    r"\bne\s*oldu\b",
    r"\bson\s*durum\b",
    # Finance / currency (real-time data needed)
    r"\b(dolar|euro|sterlin|bitcoin|btc|eth|kripto)\s*(ka[cç]|kac|ne\s*kadar|fiyat|kur)\b",
    r"\b(dolar|euro|sterlin)\b.*\b(kac|ka[cç]|tl|kur)\b",
    r"\b(bitcoin|btc|eth|kripto)\b.*\b(ne\s*durumda|fiyat|kac|ka[cç]|price)\b",
    r"\b(borsa|bist|nasdaq|dow)\b.*\b(ne\s*durumda|nasil|kac|ka[cç])\b",
    r"\bcurrent\s*(price|rate|value)\b",
    r"\bexchange\s*rate\b",
    # Sports (real-time scores)
    r"\b(mac|maç|maclar)\w*\s*(sonuc|sonu[cç]|skor|ne\s*oldu)\b",
    r"\bbugunku\s*(mac|maç|maclar)\w*\b",
    r"\bbugunun\s*(mac|maç|maclar)\w*\b",
    r"\b(super\s*lig|champions\s*league|premier\s*league)\b.*\b(sonuc|skor|puan)\b",
    r"\b(galatasaray|fenerbahce|besiktas|trabzonspor)\b.*\b(mac|maç|skor|sonuc)\b",
    r"\bmaclar\w*\s*(soyle|ver|listele)\b",
    # Current events queries (disasters, elections, etc.)
    r"\bson\s*(deprem|sel|yangin|kaza)\b",
    r"\b(deprem|sel|yangin)\b.*\b(nerede|ne\s*zaman|son)\b",
    r"\b(secim|se[cç]im)\s*(sonuc|sonu[cç])\b",
    r"\bson\s*(secim|se[cç]im)\b",
    # "what's happening" / latest developments
    r"\bson\s*gelismeler\b",
    r"\bne\s*degisti\b",
    r"\byapay\s*zeka\b.*\b(son|yeni|gelism)\b",
)


def detect_realtime_need(query: str) -> dict:
    """Detect which real-time data sources a query needs.

    Returns dict with keys: time, weather, search.
    Each is True/False, plus weather_location if detected.
    """
    low = (query or "").lower().strip()
    result = {"time": False, "weather": False, "search": False, "weather_location": "Istanbul"}

    if not low:
        return result

    # Time detection
    for p in _TIME_PATTERNS:
        if re.search(p, low):
            result["time"] = True
            break

    # Weather detection
    for p in _WEATHER_PATTERNS:
        if re.search(p, low):
            result["weather"] = True
            # Try to extract location
            loc_match = re.search(
                r"(?:hava\s*durumu|weather|forecast|hava)\s+(?:in\s+|icin\s+|için\s+)?([A-Za-z\u00c0-\u024f]+(?:\s+[A-Za-z\u00c0-\u024f]+)?)",
                low,
            )
            if loc_match:
                candidate = loc_match.group(1).strip()
                # Filter out non-location words
                skip = {"nasil", "nasildir", "ne", "durumu", "bugün", "bugun", "yarin", "how", "today"}
                if candidate.lower() not in skip and len(candidate) > 2:
                    result["weather_location"] = candidate.title()
            break

    # Search detection
    for p in _SEARCH_PATTERNS:
        if re.search(p, low):
            result["search"] = True
            break

    return result


def _optimize_search_query(query: str) -> str:
    """Transform user query into a better web search query."""
    low = query.strip().lower()
    # Remove filler words that hurt search quality
    noise = ("soyle", "söyle", "ver", "bana", "neler", "nedir", "ne", "nasil",
             "lütfen", "lutfen", "bir", "biraz", "acaba")
    tokens = low.split()
    clean = [t for t in tokens if t not in noise and len(t) > 1]
    search_q = " ".join(clean) if clean else low

    # Domain-specific query optimization
    if re.search(r"\b(dolar|euro|sterlin)\b", low) and re.search(r"\b(kac|ka[cç]|tl|kur)\b", low):
        search_q = re.sub(r"\b(kac|ka[cç])\b", "kuru", search_q)
        if "bugun" not in search_q:
            search_q += " güncel kur"
    elif re.search(r"\b(deprem)\b", low):
        search_q = re.sub(r"\b(nerede|oldu)\b", "", search_q).strip()
        search_q += " Türkiye son deprem AFAD"
    elif re.search(r"\bmaclar\w*\b", low):
        search_q += " fikstür bugün"
    elif re.search(r"\b(bitcoin|btc|eth|kripto)\b", low):
        search_q += " fiyat bugün"

    # Add year for recency
    if not re.search(r"\b202[4-9]\b", search_q):
        search_q = f"{search_q} 2026"
    return search_q.strip()


def build_realtime_context(query: str) -> str:
    """Build a real-time context block to inject into the LLM prompt.

    Returns empty string if no real-time data is needed.
    """
    needs = detect_realtime_need(query)
    parts = []

    if needs["time"]:
        parts.append(format_datetime_context())

    if needs["weather"]:
        from text_utils import contains_turkish
        lang = "tr" if contains_turkish(query) else "en"
        weather = fetch_weather(location=needs["weather_location"], lang=lang)
        if weather:
            parts.append(weather)
        # Always include time with weather
        if not needs["time"]:
            parts.append(format_datetime_context())

    if needs["search"]:
        # Optimize search query for better results
        search_query = _optimize_search_query(query)
        search_result = web_search(search_query, max_results=4)
        if search_result:
            parts.append(search_result)

    if not parts:
        return ""

    return "REAL-TIME DATA (use this to answer the user's question):\n" + "\n".join(parts) + "\n"
