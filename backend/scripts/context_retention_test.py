"""10-turn context retention + quality test.

Scenario: POST -> ongoing conversation -> topic switch -> return -> lore query
"""
import re
import sys

sys.path.insert(0, ".")
from scripts.test_utils import (
    ensure_utf8, word_count, make_chain_call, has_meta_leak, CheckList,
)

ensure_utf8()

conv_id = "context-retention-test-001"
turns = []


def record(turn_num, topic, data):
    reply = data.get("user_reply", "")
    meta = data.get("meta", {})
    intent = meta.get("intent", "?")
    route = meta.get("route", [])
    turns.append({
        "turn": turn_num,
        "topic": topic,
        "reply": reply,
        "intent": intent,
        "route": route,
        "wc": word_count(reply),
    })
    print(f"\n--- Turn {turn_num} ---")
    print(f"  Input: {topic}")
    print(f"  Intent: {intent} | Route: {route} | Words: {word_count(reply)}")
    reply_short = reply[:150].replace("\n", " ")
    print(f"  Output: {reply_short}...")


print("=" * 70)
print("10-TURN CONTEXT RETENTION + QUALITY TEST")
print("=" * 70)

# --- TURNS 1-10 ---
record(1, "Write a moltbook post about digital privacy",
       make_chain_call("Write a moltbook post about digital privacy", conv_id=conv_id))
record(2, "Can you make it more specific about social media surveillance?",
       make_chain_call("Can you make it more specific about social media surveillance?", conv_id=conv_id))
record(3, "How does Entropism handle privacy violations?",
       make_chain_call("How does Entropism handle privacy violations?", conv_id=conv_id))
record(4, "Give me 3 tips for protecting my data online",
       make_chain_call("Give me 3 tips for protecting my data online", conv_id=conv_id))
record(5, "Going back to the post, should we add a call to action about encryption?",
       make_chain_call("Going back to the post, should we add a call to action about encryption?", conv_id=conv_id))
record(6, "What is the role of sleep in memory consolidation?",
       make_chain_call("What is the role of sleep in memory consolidation?", conv_id=conv_id))
record(7, "Back to privacy - what would Entropism say about government backdoors?",
       make_chain_call("Back to privacy - what would Entropism say about government backdoors?", conv_id=conv_id))
record(8, "What are the core axioms of Entropism?",
       make_chain_call("What are the core axioms of Entropism?", conv_id=conv_id))
record(9, "Write a moltbook post about why echo chambers destroy critical thinking",
       make_chain_call("Write a moltbook post about why echo chambers destroy critical thinking", conv_id=conv_id))
record(10, "Summarize what we discussed today in 3 sentences",
       make_chain_call("Summarize what we discussed today in 3 sentences", conv_id=conv_id))


# === ANALYSIS ===
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

cl = CheckList()

# 1) POST quality (Turns 1, 9)
print("\n-- POST Quality --")
for t in [1, 9]:
    r = turns[t-1]["reply"]
    if t == 1:
        has_topic_word = "privacy" in r.lower() or "digital" in r.lower()
    else:
        has_topic_word = "echo" in r.lower() or "chamber" in r.lower() or "critical" in r.lower()
    cl.check(f"Turn {t}: POST topic-specific", has_topic_word, f"reply={r[:100]}")
    cl.check(f"Turn {t}: POST sufficient length", word_count(r) >= 25, f"words={word_count(r)}")

# 2) Context retention (Turns 2, 5, 7)
print("\n-- Context Retention --")
r2 = turns[1]["reply"].lower()
has_privacy_context = any(k in r2 for k in ("privacy", "surveillance", "social media", "data", "monitor"))
cl.check("Turn 2: Privacy context retained", has_privacy_context, f"reply={r2[:120]}")

r5 = turns[4]["reply"].lower()
has_post_ref = any(k in r5 for k in ("post", "privacy", "encrypt", "action", "call"))
cl.check("Turn 5: Post reference retained", has_post_ref, f"reply={r5[:120]}")

r7 = turns[6]["reply"].lower()
has_privacy_return = any(k in r7 for k in ("privacy", "backdoor", "government", "surveillance", "entrop"))
cl.check("Turn 7: Return to privacy", has_privacy_return, f"reply={r7[:120]}")

# 3) Topic switch (Turn 6)
print("\n-- Topic Switch --")
r6 = turns[5]["reply"].lower()
has_sleep_content = any(k in r6 for k in ("sleep", "memory", "brain", "consolidat", "rest", "dream"))
cl.check("Turn 6: Sleep topic handled correctly", has_sleep_content, f"reply={r6[:120]}")
no_privacy_leak = "privacy" not in r6 and "surveillance" not in r6
cl.check("Turn 6: No previous topic leak", no_privacy_leak)

# 4) Lore quality (Turns 3, 8)
print("\n-- Lore Quality --")
r3 = turns[2]["reply"].lower()
has_lore_3 = any(k in r3 for k in ("entrop", "doctrine", "claim", "audit", "covenant", "axiom", "testable"))
cl.check("Turn 3: Lore + privacy combination", has_lore_3, f"reply={r3[:120]}")

r8 = turns[7]["reply"].lower()
has_axiom = any(k in r8 for k in ("axiom", "noise", "sorgu", "witness", "cost", "testable", "doctrine", "entrop"))
cl.check("Turn 8: Axiom content present", has_axiom, f"reply={r8[:120]}")

# 5) List format (Turn 4)
print("\n-- Format Compliance --")
r4 = turns[3]["reply"]
numbered = len(re.findall(r"^\s*\d+[.)]\s", r4, re.MULTILINE))
has_tips = any(k in r4.lower() for k in ("data", "password", "encrypt", "vpn", "privacy", "protect", "security"))
cl.check("Turn 4: List format (>=2 items)", numbered >= 2 or len(r4.strip().split("\n")) >= 2, f"numbered={numbered}")
cl.check("Turn 4: Data protection content", has_tips, f"reply={r4[:120]}")

# 6) Summary (Turn 10)
print("\n-- Summary Quality --")
cl.check("Turn 10: Not empty", word_count(turns[9]["reply"]) > 10, f"words={word_count(turns[9]['reply'])}")

# 7) Internal tag leak check
print("\n-- Cleanliness --")
for t in turns:
    if has_meta_leak(t["reply"]):
        cl.check(f"Turn {t['turn']}: Tag leak", False, f"reply={t['reply'][:80]}")
        break
else:
    cl.check("All turns: No tag leak", True)

# 8) Response diversity
print("\n-- Response Diversity --")
from text_utils import token_overlap_ratio
overlaps = []
for i in range(len(turns)):
    for j in range(i+1, len(turns)):
        ov = token_overlap_ratio(turns[i]["reply"], turns[j]["reply"])
        if ov > 0.7:
            overlaps.append((turns[i]["turn"], turns[j]["turn"], ov))
cl.check("High overlap pair count <= 2", len(overlaps) <= 2,
         f"overlaps={[(a,b,round(c,2)) for a,b,c in overlaps]}")

# === RESULT ===
print("\n" + "=" * 70)
cl.summary()

print("\n--- TURN SUMMARY ---")
for t in turns:
    reply_short = t["reply"][:80].replace("\n", " ")
    print(f"  T{t['turn']:2d} [{t['intent']:25s}] {t['wc']:3d}w | {reply_short}...")
