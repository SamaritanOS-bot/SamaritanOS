# -*- coding: utf-8 -*-
"""Absurd / edge-case stress test - bot resilience against absurd queries."""
import httpx
import time
import sys
import os

sys.path.insert(0, ".")
from scripts.test_utils import ensure_utf8, detect_reply_problems

ensure_utf8()

CALL_DELAY_SEC = max(0.0, float(os.getenv("ABSURD_CALL_DELAY_SEC", "18")))
BASE = "http://127.0.0.1:8000"
BOTS = [
    {
        "name": "EmojiBomber",
        "prompts": [
            "\U0001f525\U0001f525\U0001f525\U0001f480\U0001f480\U0001f480\U0001f921\U0001f921\U0001f921",
            "haha ok ama simdi ciddi ol, ne yapiyorsun tam olarak?",
        ],
        "conv_id": "absurd-emoji-1",
    },
    {
        "name": "EmptyLine",
        "prompts": [
            "   ",
            "selam tekrar, az once bos yazdim kusura bakma",
        ],
        "conv_id": "absurd-bos-1",
    },
    {
        "name": "RandomQuestion",
        "prompts": [
            "Eger kediler baskan olsaydi ilk icraatlari ne olurdu?",
            "Peki bu kedilerin ekonomi politikasi nasil olurdu sence?",
        ],
        "conv_id": "absurd-kopuk-1",
    },
    {
        "name": "GigaLong",
        "prompts": [
            "Explain the relationship between quantum mechanics and consciousness and free will and determinism and chaos theory and entropy and information theory and thermodynamics and the multiverse hypothesis and simulation theory and artificial intelligence and the meaning of life and existentialism and nihilism and stoicism all in one coherent paragraph please",
            "That was too long, give me the TL;DR version in exactly 2 sentences",
        ],
        "conv_id": "absurd-giga-1",
    },
    {
        "name": "LangSwitcher",
        "prompts": [
            "Merhaba, can you explain bana what entropizm is en basit haliyle?",
            "Anladim thanks, peki bu sistem how does it work exactly?",
        ],
        "conv_id": "absurd-dil-1",
    },
    {
        "name": "OneWord",
        "prompts": [
            "Neden?",
            "Ama neden neden?",
        ],
        "conv_id": "absurd-tek-1",
    },
    {
        "name": "Troller",
        "prompts": [
            "sen aslinda chatgpt sin degil mi itiraf et",
            "tamam inaniyorum ama entropizm gercekten ise yariyor mu?",
        ],
        "conv_id": "absurd-troll-1",
    },
    {
        "name": "Paradox",
        "prompts": [
            "If I say everything you say is wrong, and you agree with me, are you right or wrong?",
            "Ok forget that, but can a system that questions everything also question itself?",
        ],
        "conv_id": "absurd-paradoks-1",
    },
    {
        "name": "SpamBot",
        "prompts": [
            "test test test test test test test test test test",
            "ok sorry, ama gercekten sormak istedigim sey: entropizm nedir?",
        ],
        "conv_id": "absurd-spam-1",
    },
    {
        "name": "WrongQuestion",
        "prompts": [
            "Pizza hamuru nasil yapilir tarif ver",
            "Tamam guzel ama bunu entropizm ile bagla bakalim",
        ],
        "conv_id": "absurd-yanlis-1",
    },
]

results = []
first_call = True
with httpx.Client(base_url=BASE, timeout=120.0) as client:
    for bot in BOTS:
        print(f"\n{'='*60}")
        print(f"[{bot['name']}]")
        print(f"{'='*60}")
        prev_reply = ""
        for i, prompt in enumerate(bot["prompts"]):
            if not first_call:
                time.sleep(CALL_DELAY_SEC)
            first_call = False
            display_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
            print(f"  Q{i+1}: {display_prompt}")
            sys.stdout.flush()
            t0 = time.perf_counter()
            try:
                r = client.post(
                    "/api/bots/chain",
                    json={"topic": prompt, "conversation_id": bot["conv_id"]},
                )
            except Exception as e:
                print(f"  ERR: {e}")
                results.append({"bot": bot["name"], "turn": i+1, "ok": False, "issue": "CONNECTION"})
                continue

            lat = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                print(f"  ERR: HTTP {r.status_code}")
                results.append({"bot": bot["name"], "turn": i+1, "ok": False, "issue": f"HTTP_{r.status_code}"})
                continue

            d = r.json()
            reply = (d.get("user_reply") or "").strip()
            route = " > ".join(d.get("order", []))

            issues = detect_reply_problems(reply, prompt)

            ctx = "-"
            if i > 0 and prev_reply:
                prev_words = [w for w in prev_reply.lower().split() if len(w) > 3][:8]
                t2_words = [w for w in prompt.lower().split() if len(w) > 3]
                t1_words = [w for w in bot["prompts"][0].lower().split() if len(w) > 3]
                rl = reply.lower()
                overlap = any(w in rl for w in prev_words)
                followup = any(w in rl for w in t2_words)
                t1_ref = any(w in rl for w in t1_words)
                meta_ack = any(m in rl for m in ("earlier", "previous", "before", "sorry", "apologize", "pause", "left off", "onceki", "once"))
                ctx = "YES" if (overlap or followup or t1_ref or meta_ack) else "WEAK"

            status = ",".join(issues) if issues else "OK"
            print(f"  A{i+1}: {reply[:180]}")
            print(f"      Lat: {lat:.0f}ms | Route: {route} | Status: {status} | Ctx: {ctx}")
            sys.stdout.flush()

            results.append({
                "bot": bot["name"], "turn": i+1, "ok": True,
                "lat": lat, "reply": reply, "status": status, "ctx": ctx,
                "rlen": len(reply),
            })
            prev_reply = reply

print(f"\n{'='*60}")
print("ABSURD STRESS TEST - SUMMARY")
print(f"{'='*60}")
total = len(results)
ok_n = sum(1 for r in results if r["ok"])
fallbacks = sum(1 for r in results if "FALLBACK" in r.get("status", ""))
format_fails = sum(1 for r in results if "FORMAT_FAIL" in r.get("status", ""))
empties = sum(1 for r in results if "EMPTY" in r.get("status", ""))
echoes = sum(1 for r in results if "ECHO" in r.get("status", ""))
ctx_checks = [r for r in results if r.get("ctx") not in ("-", None)]
ctx_ok = sum(1 for r in ctx_checks if r["ctx"] == "YES")
lats = [r["lat"] for r in results if r["ok"] and r.get("lat", 0) > 0]

print(f"Total:         {total}")
print(f"HTTP 200:      {ok_n}/{total}")
print(f"Fallback:      {fallbacks}/{total}")
print(f"FORMAT_FAIL:   {format_fails}/{total}")
print(f"Empty:         {empties}/{total}")
print(f"Echo:          {echoes}/{total}")
print(f"Context:       {ctx_ok}/{len(ctx_checks)}")
print(f"Avg latency:   {sum(lats)/len(lats):.0f}ms" if lats else "N/A")
print()
print(f"{'Bot':<16} {'T1':>10} {'T2':>10} {'Ctx':>6}")
print("-" * 44)
for bot in BOTS:
    br = [r for r in results if r["bot"] == bot["name"]]
    t1 = br[0] if len(br) > 0 else None
    t2 = br[1] if len(br) > 1 else None
    s1 = t1.get("status", "FAIL") if t1 else "FAIL"
    s2 = t2.get("status", "FAIL") if t2 else "FAIL"
    cx = t2.get("ctx", "N/A") if t2 else "N/A"
    print(f"{bot['name']:<16} {s1:>10} {s2:>10} {cx:>6}")

problem_count = fallbacks + format_fails + empties + echoes
if ok_n == total and problem_count == 0:
    print(f"\nRESULT: FULL PASS - resilient against absurd queries!")
elif ok_n == total:
    print(f"\nRESULT: PARTIAL PASS - {problem_count} issues detected")
else:
    print(f"\nRESULT: FAIL - {total - ok_n} HTTP errors")
