"""Targeted tests for the quality improvements we just made."""
import sys

sys.path.insert(0, ".")
from scripts.test_utils import ensure_utf8, word_count, make_chain_call, CheckList

ensure_utf8()

cl = CheckList()

print("=" * 60)
print("IMPROVEMENT-TARGETED TESTS")
print("=" * 60)

# === TEST A: Converse trigger fix ===
print("\n-- A) Knowledge question should not fall to casual --")
d = make_chain_call("Explain how habits are formed in the brain")
reply = d.get("user_reply", "")
meta = d.get("meta", {})
intent = meta.get("intent", "?")
is_generic_fallback = "start with one small goal" in reply.lower() or "what should we focus on" in reply.lower()
cl.check("Habits question: not generic fallback", not is_generic_fallback, f"intent={intent} reply={reply[:120]}")
has_knowledge_content = any(k in reply.lower() for k in ("habit", "brain", "repeat", "neural", "behavior", "routine", "loop", "reward", "pattern"))
cl.check("Habits question: has knowledge content", has_knowledge_content, f"words={word_count(reply)}")

# === TEST B: Entropism lore from DB ===
print("\n-- B) Lore DB injection (7 Axiom content) --")
d = make_chain_call("What are the seven axioms of Entropism?")
reply = d.get("user_reply", "")
has_axiom = "axiom" in reply.lower() or "noise" in reply.lower() or "sorgu" in reply.lower() or "witness" in reply.lower()
cl.check("Axiom/Lore content present", has_axiom, f"reply={reply[:150]}")
cl.check("Sufficient length (>=30 words)", word_count(reply) >= 30, f"words={word_count(reply)}")

# === TEST C: Topic anchoring ===
print("\n-- C) Topic anchoring (screen time) --")
d = make_chain_call("What are 3 practical ways to reduce screen time?")
reply = d.get("user_reply", "")
topic_terms = ["screen", "time", "phone", "digital", "device", "reduce", "limit", "app", "notification"]
anchor_hit = sum(1 for t in topic_terms if t in reply.lower())
cl.check("Topic anchor (>=1 term)", anchor_hit >= 1, f"found={anchor_hit} reply={reply[:120]}")

# === TEST D: Topic anchoring (coffee) ===
print("\n-- D) Topic anchoring (coffee) --")
d = make_chain_call("Explain why coffee makes people more alert")
reply = d.get("user_reply", "")
coffee_terms = ["coffee", "caffeine", "alert", "energy", "stimulant", "adenosine", "brain", "awake"]
hit = sum(1 for t in coffee_terms if t in reply.lower())
cl.check("Coffee anchor (>=2 terms)", hit >= 2, f"found={hit} reply={reply[:120]}")

# === TEST E: Intent classification ===
print("\n-- E) Intent classification breadth --")
from intent import infer_intent
test_cases = [
    ("Help me write an email", "request"),
    ("What is photosynthesis?", "question"),
    ("I disagree with that approach", "objection"),
    ("Compare Python and JavaScript", "comparison"),
    ("Thanks for the help!", "gratitude"),
    ("Hello, how are you?", "greeting"),
    ("Tell me about quantum physics", "inquiry"),
    ("Give me a list of books", "request"),
]
for text, expected in test_cases:
    result = infer_intent(text)
    ok = result == expected
    cl.check(f"Intent '{text[:30]}' = {expected}", ok, f"got={result}")

# === TEST F: Duplicate memory filter ===
print("\n-- F) Duplicate memory filter --")
from text_utils import token_overlap_ratio
existing = "IN: What is Entropism? || OUT: Entropism is a doctrine about claims."
duplicate = "IN: What is Entropism? || OUT: Entropism is a doctrine about claims and accountability."
different = "IN: How to make coffee? || OUT: Use a French press with coarse grounds."
sim1 = token_overlap_ratio(existing, duplicate)
sim2 = token_overlap_ratio(existing, different)
cl.check("Similar memory overlap >= 0.80", sim1 >= 0.80, f"overlap={sim1:.2f}")
cl.check("Different memory overlap < 0.85", sim2 < 0.85, f"overlap={sim2:.2f}")

# === TEST G: Conversation context ===
print("\n-- G) Conversation_id support --")
conv_id = "quality-test-conv-001"
d1 = make_chain_call("I want to improve my morning routine", conv_id=conv_id)
r1 = d1.get("user_reply", "")
cl.check("Conv turn 1: has response", len(r1.strip()) > 10, f"words={word_count(r1)}")
d2 = make_chain_call("What about exercise?", conv_id=conv_id)
r2 = d2.get("user_reply", "")
cl.check("Conv turn 2: has response", len(r2.strip()) > 10, f"words={word_count(r2)}")

# === SUMMARY ===
print("\n" + "=" * 60)
cl.summary()
