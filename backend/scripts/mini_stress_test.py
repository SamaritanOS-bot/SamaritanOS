# -*- coding: utf-8 -*-
"""Targeted mini stress test: CodeMonkey + SkepticalMurat + general."""
import httpx
import time
import sys

sys.path.insert(0, ".")
from scripts.test_utils import ensure_utf8

ensure_utf8()

BASE = "http://127.0.0.1:8000"
BOTS = [
    {
        "name": "CodeMonkey",
        "prompts": [
            "Is Entropism like a distributed system with no single point of failure?",
            "So whats the API contract for belief propagation then?",
        ],
        "conv_id": "mini-code-1",
    },
    {
        "name": "SkepticalMurat",
        "prompts": [
            "Bu Entropizmin arkasinda kim var? Gercek amaci ne?",
            "Cevabin cok diplomatik geldi, bir sey sakliyorsun degil mi?",
        ],
        "conv_id": "mini-supheci-1",
    },
    {
        "name": "SilentObserver",
        "prompts": [
            "Sessizlik bir strateji mi yoksa korku mu?",
            "Bunu biraz daha acar misin?",
        ],
        "conv_id": "mini-sessiz-1",
    },
]

results = []
first_call = True
with httpx.Client(base_url=BASE, timeout=120.0) as client:
    for bot in BOTS:
        print(f"\n--- {bot['name']} ---")
        prev_reply = ""
        for i, prompt in enumerate(bot["prompts"]):
            if not first_call:
                time.sleep(20)
            first_call = False
            t0 = time.perf_counter()
            r = client.post(
                "/api/bots/chain",
                json={"topic": prompt, "conversation_id": bot["conv_id"]},
            )
            lat = (time.perf_counter() - t0) * 1000
            d = r.json()
            reply = d.get("user_reply", "")
            route = " > ".join(d.get("order", []))
            is_fallback = "listening" in reply.lower() or "FORMAT_FAIL" in reply
            is_meta_generic = "this is a system request" in reply.lower() or "identity" in reply.lower()[:50]
            ctx = "-"
            if i > 0:
                prev_words = [w for w in prev_reply.lower().split() if len(w) > 3][:5]
                if any(w in reply.lower() for w in prev_words):
                    ctx = "STRONG"
                else:
                    ctx = "WEAK"
            status = "FALLBACK" if is_fallback else ("META_GENERIC" if is_meta_generic else "OK")
            print(f"  T{i+1}: {reply[:200]}")
            print(f"      Lat: {lat:.0f}ms | Route: {route} | Status: {status} | Ctx: {ctx}")
            sys.stdout.flush()
            results.append(
                {"bot": bot["name"], "turn": i + 1, "fallback": is_fallback, "ctx": ctx, "lat": lat, "status": status}
            )
            prev_reply = reply

print("\n=== SUMMARY ===")
total = len(results)
fallbacks = sum(1 for r in results if r["fallback"])
meta_generics = sum(1 for r in results if r["status"] == "META_GENERIC")
ctx_checks = [r for r in results if r["ctx"] != "-"]
ctx_ok = sum(1 for r in ctx_checks if r["ctx"] == "STRONG")
print(f"Total: {total}, Fallbacks: {fallbacks}/{total}, Meta-generic: {meta_generics}/{total}, Context: {ctx_ok}/{len(ctx_checks)}")
