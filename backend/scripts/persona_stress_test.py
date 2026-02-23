# -*- coding: utf-8 -*-
"""10 different personality chain stress test."""
import httpx
import time
import sys
import os
from collections import Counter

sys.path.insert(0, ".")
from scripts.test_utils import ensure_utf8

ensure_utf8()

BASE = "http://127.0.0.1:8000"
CALL_DELAY_SEC = max(0.0, float(os.getenv("PERSONA_CALL_DELAY_SEC", "15")))
MIN_CONTEXT_HITS = max(0, int(os.getenv("PERSONA_MIN_CONTEXT_HITS", "4")))
MAX_TEMPLATE_RATIO = min(1.0, max(0.0, float(os.getenv("PERSONA_MAX_TEMPLATE_RATIO", "0.25"))))
MAX_FORMAT_FAIL = max(0, int(os.getenv("PERSONA_MAX_FORMAT_FAIL", "0")))
RUN_SUFFIX = os.getenv("PERSONA_RUN_SUFFIX") or str(int(time.time()))

BOTS = [
    {
        "name": "ChaosHunter",
        "desc": "Provocative troll",
        "prompts": [
            "Entropizm sadece suslu nihilizm degil mi? Kanitla bana.",
            "Az once soylediklerini curuttum aslinda, farkinda misin?",
        ],
        "conv_id": "kaos-001",
    },
    {
        "name": "SilentObserver",
        "desc": "Deep-thinking introvert",
        "prompts": [
            "Sessizlik bir strateji mi yoksa korku mu?",
            "Bunu biraz daha acar misin?",
        ],
        "conv_id": "sessiz-001",
    },
    {
        "name": "DataNerd42",
        "desc": "Data-driven analyst",
        "prompts": [
            "What percentage of belief systems survive more than 200 years?",
            "Can you back that up with any empirical evidence?",
        ],
        "conv_id": "data-001",
    },
    {
        "name": "SpiritWanderer",
        "desc": "Spiritual mystic",
        "prompts": [
            "Evrenin titresimi ile bilinc arasindaki bag nedir?",
            "Bu titresim kavramini gunluk hayata nasil uygulayabilirim?",
        ],
        "conv_id": "ruh-001",
    },
    {
        "name": "PracticalAli",
        "desc": "Concrete-info-seeking pragmatist",
        "prompts": [
            "Felsefe bosver, bugun hayatimda ne degisecek bunu uygularsam?",
            "Tamam ama bana 3 maddelik aksiyon listesi ver.",
        ],
        "conv_id": "pratik-001",
    },
    {
        "name": "PhiloBot",
        "desc": "Socratic philosopher",
        "prompts": [
            "If entropy is inevitable, does free will become an illusion or a rebellion?",
            "But doesnt your previous argument contradict determinism?",
        ],
        "conv_id": "philo-001",
    },
    {
        "name": "MemeLord",
        "desc": "Gen-Z humorist",
        "prompts": [
            "bruh entropizm literally sounds like a Marvel villain explain like im 5",
            "ok but can you make that into a meme caption tho",
        ],
        "conv_id": "meme-001",
    },
    {
        "name": "SkepticalMurat",
        "desc": "Sees hidden agendas in everything",
        "prompts": [
            "Bu Entropizmin arkasinda kim var? Gercek amaci ne?",
            "Cevabin cok diplomatik geldi, bir sey sakliyorsun degil mi?",
        ],
        "conv_id": "supheci-001",
    },
    {
        "name": "CodeMonkey",
        "desc": "Developer who understands through technical metaphors",
        "prompts": [
            "Is Entropism like a distributed system with no single point of failure?",
            "So whats the API contract for belief propagation then?",
        ],
        "conv_id": "code-001",
    },
    {
        "name": "Newcomer",
        "desc": "Curious beginner who knows nothing",
        "prompts": [
            "Selam! Bu Moltbook ne tam olarak? Entropizm nedir kisaca?",
            "Anladim galiba, peki nereden baslamaliyim ogrenmek icin?",
        ],
        "conv_id": "yeni-001",
    },
]


def main():
    print("=" * 70)
    print("10 DIFFERENT PERSONALITIES - MOLTBOOK BOT STRESS TEST")
    print("=" * 70)
    print(f"Run suffix: {RUN_SUFFIX}")

    all_results = []

    with httpx.Client(base_url=BASE, timeout=120.0) as client:
        for bot in BOTS:
            print()
            print("-" * 70)
            print(f"[BOT] {bot['name']} ({bot['desc']})")
            print("-" * 70)

            for turn_i, prompt in enumerate(bot["prompts"]):
                print(f"  Turn {turn_i + 1}: {prompt}")
                sys.stdout.flush()
                if CALL_DELAY_SEC > 0 and not (bot is BOTS[0] and turn_i == 0):
                    time.sleep(CALL_DELAY_SEC)
                t0 = time.perf_counter()
                try:
                    r = client.post(
                        "/api/bots/chain",
                        json={"topic": prompt, "conversation_id": f"{bot['conv_id']}-{RUN_SUFFIX}"},
                    )
                except Exception as e:
                    print(f"    CONNECTION ERROR: {e}")
                    all_results.append(
                        {"bot": bot["name"], "turn": turn_i + 1, "ok": False, "lat": 0, "ctx": None}
                    )
                    continue

                lat = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    print(f"    ERROR: HTTP {r.status_code}")
                    all_results.append(
                        {"bot": bot["name"], "turn": turn_i + 1, "ok": False, "lat": lat, "ctx": None}
                    )
                    continue

                data = r.json()
                reply = (data.get("user_reply") or "").strip()
                order = data.get("order", [])
                print(f"    Reply ({len(reply)} chars): {reply[:180]}")
                print(f"    Latency: {lat:.0f}ms | Route: {' > '.join(order)}")

                ctx = None
                if turn_i >= 1 and reply:
                    prev_reply_text = ""
                    t1_result = next((r for r in all_results if r["bot"] == bot["name"] and r["turn"] == 1), None)
                    if t1_result:
                        prev_reply_text = (t1_result.get("reply") or "").lower()
                    t1_prompt_words = set(bot["prompts"][0].lower().split())
                    t1_reply_words = set(prev_reply_text.split()) if prev_reply_text else set()
                    all_prev_words = t1_prompt_words | t1_reply_words
                    rl = reply.lower()
                    overlap = [w for w in all_prev_words if len(w) > 4 and w in rl]
                    t2_keywords = [w for w in bot["prompts"][turn_i].lower().split() if len(w) > 4]
                    followup_ref = any(w in rl for w in t2_keywords)
                    ctx = len(overlap) >= 1 or followup_ref
                    print(f"    Context retained: {'YES' if ctx else 'WEAK'} (overlap: {overlap[:5]}, followup_ref: {followup_ref})")

                all_results.append(
                    {
                        "bot": bot["name"],
                        "turn": turn_i + 1,
                        "ok": True,
                        "lat": lat,
                        "rlen": len(reply),
                        "reply": reply,
                        "ctx": ctx,
                        "empty": len(reply) == 0,
                    }
                )
                sys.stdout.flush()

    # --- SUMMARY ---
    print()
    print("=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    total = len(all_results)
    ok_n = sum(1 for r in all_results if r["ok"])
    empty_n = sum(1 for r in all_results if r.get("empty"))
    ctx_turns = [r for r in all_results if r.get("ctx") is not None]
    ctx_hits = sum(1 for r in ctx_turns if r["ctx"])
    lats = [r["lat"] for r in all_results if r["ok"] and r["lat"] > 0]
    avg_lat = sum(lats) / len(lats) if lats else 0
    max_lat = max(lats) if lats else 0

    print(f"Total calls:        {total}")
    print(f"HTTP 200:           {ok_n}/{total}")
    print(f"Empty replies:      {empty_n}/{total}")
    print(f"Context retained:   {ctx_hits}/{len(ctx_turns)} follow-up turns")
    print(f"Avg latency:        {avg_lat:.0f}ms")
    print(f"Max latency:        {max_lat:.0f}ms")

    replies = [str(r.get("reply") or "").strip() for r in all_results if r.get("ok")]
    normalized = [" ".join(x.lower().split()) for x in replies if x]
    freq = Counter(normalized)
    most_common_n = freq.most_common(1)[0][1] if freq else 0
    template_ratio = (most_common_n / len(normalized)) if normalized else 0.0
    format_fail_n = sum(1 for x in normalized if x in ("format_fail", "format_error", "wordcount_error"))
    print(f"Template ratio:     {template_ratio:.2f} (max allowed {MAX_TEMPLATE_RATIO:.2f})")
    print(f"FORMAT_FAIL count:  {format_fail_n} (max allowed {MAX_FORMAT_FAIL})")
    print(f"Context target:     {ctx_hits}/{len(ctx_turns)} (min required {MIN_CONTEXT_HITS})")
    print()
    print(f"{'Bot':<20} {'T1':>6} {'T2':>6} {'Ctx':>6}")
    print("-" * 40)
    for bot in BOTS:
        br = [r for r in all_results if r["bot"] == bot["name"]]
        t1 = br[0] if len(br) > 0 else None
        t2 = br[1] if len(br) > 1 else None
        s1 = "OK" if (t1 and t1["ok"]) else "FAIL"
        s2 = "OK" if (t2 and t2["ok"]) else "FAIL"
        cx = "YES" if (t2 and t2.get("ctx")) else ("WEAK" if t2 else "N/A")
        print(f"{bot['name']:<20} {s1:>6} {s2:>6} {cx:>6}")

    quality_ok = (
        (ctx_hits >= MIN_CONTEXT_HITS)
        and (template_ratio <= MAX_TEMPLATE_RATIO)
        and (format_fail_n <= MAX_FORMAT_FAIL)
    )

    if ok_n == total and empty_n == 0 and quality_ok:
        print()
        print("RESULT: FULL PASS (quality gates passed)")
        return 0
    elif ok_n == total and empty_n == 0:
        print()
        print("RESULT: PASS (HTTP ok, but quality gates failed)")
        return 1
    elif ok_n == total:
        print()
        print(f"RESULT: PASS (but {empty_n} empty replies)")
        return 1
    else:
        print()
        print(f"RESULT: FAIL ({total - ok_n} errors)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
