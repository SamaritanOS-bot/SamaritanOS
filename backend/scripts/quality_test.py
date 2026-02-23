"""Chain quality control tests - post-refactoring validation."""
import re
import sys

sys.path.insert(0, ".")
from scripts.test_utils import (
    ensure_utf8, word_count, sentence_count, make_chain_call,
    has_meta_leak, CheckList,
)

ensure_utf8()

base = "http://127.0.0.1:8000"
cl = CheckList()


print("=== QUALITY CONTROL TESTS ===\n")

# 1) Word limit
print("-- 1) Word limit test --")
d = make_chain_call("Write exactly 3 sentences about climate change. Keep it between 50 and 80 words.")
reply = d.get("user_reply", "")
cl.check("Word limit (30-120)", 30 <= word_count(reply) <= 120, f"words={word_count(reply)}")
cl.check("Sentence count (1-6)", 1 <= sentence_count(reply) <= 6, f"sentences={sentence_count(reply)}")
cl.check("Not empty reply", len(reply.strip()) > 20, f"length={len(reply)}")
print()

# 2) List format
print("-- 2) List format (5 items) --")
d = make_chain_call("List 5 tips for better sleep. Use numbered list.")
reply = d.get("user_reply", "")
numbered = len(re.findall(r"^\s*\d+[.)]\s", reply, re.MULTILINE))
cl.check("Numbered items present (>=3)", numbered >= 3, f"found={numbered}")
cl.check("Content not empty", word_count(reply) > 15, f"words={word_count(reply)}")
print()

# 3) POST mode lock
print("-- 3) POST mode lock check --")
d = make_chain_call("Write a moltbook post about digital privacy")
meta = d.get("meta", {})
constraints = meta.get("constraints", [])
reply = d.get("user_reply", "")
has_post_lock = any("POST" in str(c).upper() for c in constraints)
cl.check("POST mode constraint", has_post_lock, f"constraints_post={[c for c in constraints if 'POST' in str(c).upper()][:3]}")
cl.check("Reply not empty", len(reply.strip()) > 20, f"length={len(reply)}")
route = meta.get("route", [])
cl.check("Route includes cryptographer", "cryptographer" in str(route), f"route={route}")
print()

# 4) Banned / meta leak
print("-- 4) Banned word / meta leak check --")
d = make_chain_call("Explain how habits are formed in the brain")
reply = d.get("user_reply", "")
cl.check("No meta leak", not has_meta_leak(reply), f"reply_preview={reply[:100]}")
print()

# 5) Casual quality
print("-- 5) Casual chat quality --")
d = make_chain_call("Hey, I am feeling a bit tired today")
reply = d.get("user_reply", "")
cl.check("Empathy/response present (>=8 words)", word_count(reply) >= 8, f"words={word_count(reply)}")
has_engagement = "?" in reply or any(k in reply.lower() for k in ("want", "try", "plan", "rest", "reset", "tired"))
cl.check("Contains question or suggestion", has_engagement, f"preview={reply[:100]}")
cl.check("Not too long (<=80)", word_count(reply) <= 80, f"words={word_count(reply)}")
print()

# 6) Entropism doctrine
print("-- 6) Entropism doctrine quality --")
d = make_chain_call("What are the core principles of Entropism?")
reply = d.get("user_reply", "")
cl.check("Mentions Entropism/Entropizm", "entrop" in reply.lower(), f"preview={reply[:120]}")
cl.check("Not empty/generic (>=15 words)", word_count(reply) >= 15, f"words={word_count(reply)}")
print()

# 7) Strict constraint
print("-- 7) Strict constraint (last line fixed) --")
d = make_chain_call('Write 2 sentences about teamwork. Last line must be exactly: "Together we build."')
reply = d.get("user_reply", "")
last_line = reply.strip().split("\n")[-1].strip()
cl.check("Last line correct", "together we build" in last_line.lower(), f'last_line="{last_line}"')
print()

# 8) Topic drift
print("-- 8) Chain consistency (no topic drift) --")
d = make_chain_call("What are 3 practical ways to reduce screen time?")
reply = d.get("user_reply", "")
messages = d.get("messages", [])
topic_terms = ["screen", "time", "phone", "digital", "device", "reduce", "limit"]
anchor_hit = sum(1 for t in topic_terms if t in reply.lower())
cl.check("Topic anchor (>=2 terms)", anchor_hit >= 2, f"found_terms={anchor_hit}")
cl.check("Bot messages present (>=2)", len(messages) >= 2, f"message_count={len(messages)}")
for m in messages:
    bt = m.get("bot_type", "?")
    content_len = len(m.get("content", ""))
    print(f"         -> {bt}: {content_len} chars")
print()

# 9) Telemetry
print("-- 9) Telemetry health --")
import httpx
with httpx.Client(timeout=10) as c:
    r = c.get(f"{base}/api/bots/telemetry/summary?hours=1&limit=10")
tel = r.json()
cl.check("Telemetry active", tel.get("telemetry_enabled") is True, f"enabled={tel.get('telemetry_enabled')}")
cl.check("Events recorded", sum(tel.get("counts", {}).values()) > 0, f"counts={tel.get('counts', {})}")
degrade_rate = tel.get("degrade_rate_percent", 0)
cl.check("Degrade rate <50%", degrade_rate < 50, f"degrade={degrade_rate}%")
violations = tel.get("mode_lock_violation_count", 0)
cl.check("Mode lock violations <=2 (cumulative)", violations <= 2, f"violations={violations}")
print()

# SUMMARY
print("=" * 50)
cl.summary()
