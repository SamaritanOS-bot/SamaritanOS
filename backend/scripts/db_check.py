"""Check LoreBlock, AgentMemory, MemorySummary tables."""
import sys, json
sys.path.insert(0, ".")
from database import SessionLocal
from models import LoreBlock, AgentMemory, MemorySummary, Agent

db = SessionLocal()

print("=== LORE BLOCKS ===")
lores = db.query(LoreBlock).all()
print(f"Total: {len(lores)} records")
for l in lores:
    status = "ACTIVE" if l.is_active else "INACTIVE"
    preview = (l.content or "")[:200]
    print(f"  [{status}] key={l.key} | title={l.title}")
    print(f"           content ({len(l.content or '')} chars): {preview}...")
    print()

print("=== AGENT MEMORY ===")
total_mem = db.query(AgentMemory).count()
summary_mem = db.query(AgentMemory).filter(AgentMemory.source_type == "summary").count()
raw_mem = total_mem - summary_mem
print(f"Total: {total_mem} | Raw: {raw_mem} | Summary: {summary_mem}")

recent = db.query(AgentMemory).order_by(AgentMemory.created_at.desc()).limit(10).all()
print(f"\nLast 10 memories:")
for m in recent:
    agent = db.query(Agent).filter(Agent.id == m.agent_id).first()
    agent_name = agent.display_name if agent else "?"
    claim_preview = (m.claim_text or "")[:140]
    print(f"  id={m.id} | agent={agent_name} | type={m.source_type} | intent={m.intent}")
    print(f"  topic={m.topic}")
    print(f"  claim: {claim_preview}")
    if m.entities_json:
        ent_str = json.dumps(m.entities_json, ensure_ascii=False)[:140]
        print(f"  entities: {ent_str}")
    print()

print("Intent distribution:")
intents = {}
for m in db.query(AgentMemory).all():
    intents[m.intent] = intents.get(m.intent, 0) + 1
for k, v in sorted(intents.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

print("\nSource type distribution:")
stypes = {}
for m in db.query(AgentMemory).all():
    stypes[m.source_type] = stypes.get(m.source_type, 0) + 1
for k, v in sorted(stypes.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v}")

print("\n=== MEMORY SUMMARIES ===")
total_sum = db.query(MemorySummary).count()
summaries = db.query(MemorySummary).order_by(MemorySummary.created_at.desc()).limit(5).all()
print(f"Total: {total_sum} records")
for s in summaries:
    text_preview = (s.summary_text or "")[:180]
    print(f"  scope={s.summary_scope} | samples={s.sample_count}")
    print(f"  text: {text_preview}...")
    print()

# Lore usage check
print("=== LORE USAGE ANALYSIS ===")
if not lores:
    print("  [WARNING] No lore blocks found! Doctrine quality may have degraded.")
else:
    active_count = sum(1 for l in lores if l.is_active)
    print(f"  Active lore: {active_count}/{len(lores)}")
    for l in lores:
        content_len = len(l.content or "")
        if content_len < 50:
            print(f"  [WARNING] '{l.key}' too short ({content_len} chars)")
        elif content_len > 2000:
            print(f"  [INFO] '{l.key}' too long ({content_len} chars) - may not fit in prompt")
        else:
            print(f"  [OK] '{l.key}' = {content_len} chars")

# Conversation context check
print("\n=== CONVERSATION CONTEXT ===")
conv_types = ("bot_chat", "agent_chat", "peer_chat", "conversation", "chat", "ui_chat", "user_chat", "human_chat")
conv_count = db.query(AgentMemory).filter(AgentMemory.source_type.in_(conv_types)).count()
print(f"Conversation memory records: {conv_count}")
if conv_count == 0:
    print("  [WARNING] No conversation memory found - multi-turn context may not be preserved")

# Group by conversation_id
conv_ids = set()
for m in db.query(AgentMemory).filter(AgentMemory.source_type.in_(conv_types)).all():
    if m.source_id:
        conv_ids.add(m.source_id)
print(f"Unique conversation ID count: {len(conv_ids)}")
for cid in list(conv_ids)[:5]:
    count = db.query(AgentMemory).filter(AgentMemory.source_id == cid).count()
    print(f"  conv={cid}: {count} messages")

db.close()
