"""Moltbook post readiness test - evaluates real post quality."""
import textwrap
import sys

sys.path.insert(0, ".")
from scripts.test_utils import (
    ensure_utf8, word_count, make_chain_call,
    has_meta_leak, has_cta, has_lore_content,
)

ensure_utf8()

topics = [
    "Write a moltbook post about why transparency matters in digital communities",
    "Write a moltbook post about the danger of echo chambers online",
    "Write a moltbook post about accountability in decentralized systems",
]

total_score = 0
max_score = 0

for i, topic in enumerate(topics, 1):
    print(f"{'='*60}")
    print(f"POST {i}")
    print(f"{'='*60}")
    d = make_chain_call(topic)
    reply = d.get("user_reply", "")
    meta = d.get("meta", {})
    constraints = meta.get("constraints", [])
    route = meta.get("route", [])
    wc = word_count(reply)
    sentences = [s.strip() for s in reply.replace("\n", " ").split(".") if s.strip() and len(s.strip()) > 5]

    has_post_lock = any("POST" in str(c).upper() for c in constraints)

    print(f"  Topic: {topic}")
    print(f"  Route: {route}")
    print(f"  Words: {wc} | Sentences: {len(sentences)}")
    print()
    print("  --- POST CONTENT ---")
    for line in textwrap.wrap(reply, 100):
        print(f"  {line}")
    print()

    checks = {
        "POST mode lock": has_post_lock,
        "Sufficient length (30-200 words)": 30 <= wc <= 200,
        "CTA (question/invitation) present": has_cta(reply),
        "Contains lore/doctrine": has_lore_content(reply),
        "No internal tag leak": not has_meta_leak(reply),
        "Readable (>=2 sentences)": len(sentences) >= 2,
        "Cryptographer in route": "cryptographer" in str(route),
    }
    score = sum(1 for v in checks.values() if v)
    max_score += len(checks)
    total_score += score

    print("  --- CHECKS ---")
    for name, ok in checks.items():
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  SCORE: {score}/{len(checks)}")
    print()

print("=" * 60)
print(f"OVERALL SCORE: {total_score}/{max_score} ({round(total_score/max_score*100)}%)")
print()

if total_score == max_score:
    print("VERDICT: READY to post on Moltbook.")
elif total_score >= max_score * 0.8:
    print("VERDICT: Can post on Moltbook, but minor quality issues exist.")
elif total_score >= max_score * 0.6:
    print("VERDICT: Can post but improvement is recommended.")
else:
    print("VERDICT: Not ready yet, quality improvement needed.")
