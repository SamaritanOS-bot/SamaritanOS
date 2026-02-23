# -*- coding: utf-8 -*-
"""Post mode diversity & quality test - 45 varied topics."""
import httpx
import time
import sys
import os
import re
import hashlib
from collections import Counter

sys.path.insert(0, ".")
from scripts.test_utils import (
    ensure_utf8, word_count, has_meta_leak, has_cta, has_lore_content,
)

ensure_utf8()

BASE = "http://127.0.0.1:8000"
CALL_DELAY_SEC = max(0.0, float(os.getenv("POST_TEST_DELAY_SEC", "16")))

TOPICS = [
    # --- Entropism core (TR) ---
    "Moltbook icin bir post yaz: Entropizmin temel ilkesi nedir?",
    "Moltbook postu yaz: Entropi ve duzen arasindaki denge",
    "Moltbook post: Kaos icinde anlam bulmak",
    "Bir moltbook postu yaz: Entropizm neden geleneksel felsefeden farkli?",
    "Moltbook post: Entropik adalet kavrami",
    # --- Entropism core (EN) ---
    "Write a moltbook post about the covenant of entropy",
    "Write a moltbook post: Why transparency matters in digital communities",
    "Write a moltbook post about accountability without central authority",
    "Moltbook post: The danger of echo chambers and how entropy breaks them",
    "Write a moltbook post about the doctrine of verifiable trust",
    # --- Philosophy & deep thought ---
    "Write a moltbook post: Free will vs determinism through the lens of entropy",
    "Moltbook post yaz: Belirsizlik korkusu mu yoksa ozgurluk mu?",
    "Write a moltbook post about why doubt is more honest than certainty",
    "Moltbook post: Sessizligin gucunu hafife alma",
    "Write a moltbook post about the paradox of control",
    # --- Community & social ---
    "Moltbook post yaz: Topluluk olusturmanin ilk adimi guvendir",
    "Write a moltbook post: Building trust in anonymous communities",
    "Moltbook post: Dijital cagda gercek baglantilar nasil kurulur",
    "Write a moltbook post about why decentralization needs human accountability",
    "Moltbook post yaz: Sessiz cogunlugun sesi nasil duyulur",
    # --- Provocative / debate ---
    "Write a moltbook post: Most belief systems fail because they fear questions",
    "Moltbook post yaz: Dogma oldurmez ama dusunceyi oldurur",
    "Write a moltbook post challenging the idea that all opinions are equal",
    "Moltbook post: Radikal seffaflik tehlikeli mi yoksa gerekli mi",
    "Write a moltbook post about why comfort zones are the enemy of growth",
    # --- Practical / actionable ---
    "Write a moltbook post: 3 signs your community is becoming an echo chamber",
    "Moltbook post yaz: Gunluk hayatta entropi ile nasil bas edilir",
    "Write a moltbook post about how to challenge your own beliefs daily",
    "Moltbook post: Karar verirken kullanilacak 3 entropik ilke",
    "Write a moltbook post: A simple test for intellectual honesty",
    # --- Metaphorical / creative ---
    "Write a moltbook post using fire as a metaphor for entropy",
    "Moltbook post yaz: Deniz dalgalari gibi entropi",
    "Write a moltbook post: The library of unasked questions",
    "Moltbook post: Bir agacin kokleri gibi guven",
    "Write a moltbook post about seeds that grow in chaos",
    # --- Edge cases ---
    "Write a moltbook post in exactly 3 sentences about entropy",
    "Moltbook post yaz: Tek cumlelik bir manifesto",
    "Write a moltbook post that asks more questions than it answers",
    "Moltbook post: Emoji kullanmadan guclu bir mesaj ver",
    "Write a moltbook post that starts with a controversial statement",
    # --- Mixed / cross-domain ---
    "Write a moltbook post connecting music and entropy",
    "Moltbook post yaz: Spor ve entropi arasindaki baglanti",
    "Write a moltbook post about what cooking teaches us about systems",
    "Moltbook post: Yazilim gelistirme ve entropik dusunce",
    "Write a moltbook post about the entropy of language itself",
]

MAX_TOPICS = int(os.getenv("POST_TEST_MAX_TOPICS", "0")) or len(TOPICS)
TOPICS = TOPICS[:MAX_TOPICS]


def normalize_text(t):
    return " ".join(t.lower().split())


def extract_sentences(t):
    return [s.strip() for s in re.split(r'[.!?]+', t) if s.strip() and len(s.strip()) > 5]


def extract_opening_words(t, n=5):
    words = t.split()[:n]
    return " ".join(words).lower()


def main():
    print("=" * 70)
    print("POST MODE DIVERSITY & QUALITY TEST - 45 TOPICS")
    print("=" * 70)

    results = []
    all_replies = []
    all_openings = []
    first_call = True

    with httpx.Client(base_url=BASE, timeout=120.0) as client:
        for idx, topic in enumerate(TOPICS):
            if not first_call:
                time.sleep(CALL_DELAY_SEC)
            first_call = False

            short_topic = topic[:70] + ("..." if len(topic) > 70 else "")
            print(f"\n[{idx+1:02d}/45] {short_topic}")
            sys.stdout.flush()

            t0 = time.perf_counter()
            try:
                r = client.post(
                    "/api/bots/chain",
                    json={"topic": topic, "conversation_id": f"postdiv-{idx+1}"},
                )
            except Exception as e:
                print(f"  ERR: {e}")
                results.append({"idx": idx+1, "topic": topic, "ok": False, "issue": "CONNECTION"})
                continue

            lat = (time.perf_counter() - t0) * 1000

            if r.status_code != 200:
                print(f"  ERR: HTTP {r.status_code}")
                results.append({"idx": idx+1, "topic": topic, "ok": False, "issue": f"HTTP_{r.status_code}"})
                continue

            d = r.json()
            reply = (d.get("user_reply") or "").strip()
            route = d.get("order", [])
            meta = d.get("meta", {})
            constraints = meta.get("constraints", [])

            wc = word_count(reply)
            sentences = extract_sentences(reply)
            opening = extract_opening_words(reply)
            norm = normalize_text(reply)

            # Quality checks
            has_post_lock = any("POST" in str(c).upper() for c in constraints)
            reply_has_cta = has_cta(reply)
            reply_has_lore = has_lore_content(reply)
            reply_has_leak = has_meta_leak(reply)
            is_empty = len(reply) == 0
            is_too_short = wc < 15
            is_too_long = wc > 250
            is_fallback = "i am here and listening" in reply.lower() or "format_fail" in reply.lower()

            # Language consistency
            topic_tr = any(k in topic.lower() for k in ("yaz:", "yaz ", "icin", "nasil", "nedir", "neden", "gibi", "arasinda"))
            reply_tr = any(k in reply.lower() for k in ("bir", "ile", "icin", "ama", "veya", "olan", "gibi", "nasil"))
            lang_match = (topic_tr and reply_tr) or (not topic_tr and not reply_tr) or reply_tr

            issues = []
            if is_empty: issues.append("EMPTY")
            if is_fallback: issues.append("FALLBACK")
            if is_too_short: issues.append("TOO_SHORT")
            if is_too_long: issues.append("TOO_LONG")
            if reply_has_leak: issues.append("TAG_LEAK")
            if not reply_has_cta: issues.append("NO_CTA")
            if not reply_has_lore: issues.append("NO_LORE")
            if not lang_match: issues.append("LANG_MISMATCH")

            status = ",".join(issues) if issues else "OK"
            print(f"  Reply ({wc}w, {len(sentences)}s): {reply[:150]}")
            print(f"  Lat: {lat:.0f}ms | Route: {' > '.join(route)} | Status: {status}")
            sys.stdout.flush()

            all_replies.append(norm)
            all_openings.append(opening)

            results.append({
                "idx": idx+1, "topic": topic, "ok": True, "lat": lat,
                "wc": wc, "sc": len(sentences), "status": status,
                "reply": reply, "opening": opening, "has_cta": reply_has_cta,
                "has_lore": reply_has_lore, "has_leak": reply_has_leak, "lang_match": lang_match,
                "is_fallback": is_fallback, "post_lock": has_post_lock,
            })

    # --- DIVERSITY ANALYSIS ---
    print(f"\n{'='*70}")
    print("DIVERSITY & QUALITY ANALYSIS")
    print(f"{'='*70}")

    total = len(results)
    ok_n = sum(1 for r in results if r["ok"])
    ok_results = [r for r in results if r["ok"]]

    print(f"\nHTTP Success:     {ok_n}/{total}")

    fallbacks = sum(1 for r in ok_results if r.get("is_fallback"))
    empties = sum(1 for r in ok_results if r.get("wc", 0) == 0)
    leaks = sum(1 for r in ok_results if r.get("has_leak"))
    no_cta = sum(1 for r in ok_results if not r.get("has_cta"))
    no_lore = sum(1 for r in ok_results if not r.get("has_lore"))
    lang_miss = sum(1 for r in ok_results if not r.get("lang_match"))
    post_locks = sum(1 for r in ok_results if r.get("post_lock"))

    print(f"Fallback:         {fallbacks}/{ok_n}")
    print(f"Empty:            {empties}/{ok_n}")
    print(f"Tag leak:         {leaks}/{ok_n}")
    print(f"Missing CTA:      {no_cta}/{ok_n}")
    print(f"Missing lore:     {no_lore}/{ok_n}")
    print(f"Lang mismatch:    {lang_miss}/{ok_n}")
    print(f"POST lock:        {post_locks}/{ok_n}")

    wcs = [r["wc"] for r in ok_results if r.get("wc", 0) > 0]
    if wcs:
        print(f"\nWord count:       min={min(wcs)} max={max(wcs)} avg={sum(wcs)/len(wcs):.0f}")
    too_short = sum(1 for w in wcs if w < 30)
    too_long = sum(1 for w in wcs if w > 200)
    print(f"Too short (<30w): {too_short}/{len(wcs)}")
    print(f"Too long (>200w): {too_long}/{len(wcs)}")

    lats = [r["lat"] for r in ok_results if r.get("lat", 0) > 0]
    if lats:
        print(f"\nLatency:          min={min(lats):.0f}ms max={max(lats):.0f}ms avg={sum(lats)/len(lats):.0f}ms")

    opening_counter = Counter(all_openings)
    unique_openings = len(opening_counter)
    most_common_opening = opening_counter.most_common(1)[0] if opening_counter else ("N/A", 0)
    print(f"\nOpening diversity: {unique_openings}/{len(all_openings)} unique")
    print(f"Most repeated:    '{most_common_opening[0]}' ({most_common_opening[1]}x)")

    reply_hashes = [hashlib.md5(r.encode()).hexdigest() for r in all_replies]
    unique_hashes = len(set(reply_hashes))
    print(f"Full reply unique: {unique_hashes}/{len(all_replies)}")

    all_words_flat = " ".join(all_replies).split()
    bigrams = [f"{all_words_flat[i]} {all_words_flat[i+1]}" for i in range(len(all_words_flat)-1)]
    bigram_freq = Counter(bigrams)
    top_bigrams = bigram_freq.most_common(10)
    print(f"\nTop 10 most repeated 2-grams:")
    for bg, cnt in top_bigrams:
        pct = cnt / len(bigrams) * 100 if bigrams else 0
        print(f"  '{bg}': {cnt}x ({pct:.1f}%)")

    reply_counter = Counter(all_replies)
    most_common_reply = reply_counter.most_common(1)[0] if reply_counter else ("", 0)
    template_ratio = most_common_reply[1] / len(all_replies) if all_replies else 0
    print(f"\nTemplate ratio:    {template_ratio:.2f} (most repeated: {most_common_reply[1]}x)")

    print(f"\n{'Idx':<5} {'Wc':>4} {'Sc':>3} {'CTA':>4} {'Lore':>5} {'Leak':>5} {'Lang':>5} {'Status':<20} {'Opening...':<30}")
    print("-" * 85)
    for r in ok_results:
        print(
            f"{r['idx']:<5} {r.get('wc',0):>4} {r.get('sc',0):>3} "
            f"{'Y' if r.get('has_cta') else 'N':>4} "
            f"{'Y' if r.get('has_lore') else 'N':>5} "
            f"{'Y' if r.get('has_leak') else 'N':>5} "
            f"{'Y' if r.get('lang_match') else 'N':>5} "
            f"{r.get('status','?'):<20} "
            f"{r.get('opening','')[:28]:<30}"
        )

    quality_ok = (
        ok_n >= total * 0.9
        and fallbacks == 0
        and leaks == 0
        and template_ratio <= 0.1
        and unique_openings >= len(all_openings) * 0.6
    )
    diversity_ok = unique_hashes >= len(all_replies) * 0.8

    print(f"\n{'='*70}")
    if quality_ok and diversity_ok:
        print("RESULT: FULL PASS - Post mode is high quality and diverse!")
    elif quality_ok:
        print(f"RESULT: QUALITY PASS but DIVERSITY WEAK (unique: {unique_hashes}/{len(all_replies)})")
    elif diversity_ok:
        print(f"RESULT: DIVERSITY PASS but QUALITY ISSUES ({fallbacks} fallback, {leaks} leak)")
    else:
        print(f"RESULT: NEEDS IMPROVEMENT - quality and diversity issues present")
    print(f"{'='*70}")

    return 0 if (quality_ok and diversity_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
