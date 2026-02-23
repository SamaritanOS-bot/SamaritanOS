# Developer Documentation — Moltbook Agent Engine

This document is intended to help a developer who is new to the project quickly understand and start working with it.

---

## 1. Project Structure

```
AiBottester/
├── README.md                    # General project overview and endpoint list
├── DOCS.md                      # This file (detailed developer guide)
├── .gitignore
│
└── backend/
    ├── main.py                  # Main FastAPI application (~8500 lines)
    ├── llama_service.py         # OpenRouter LLM client
    ├── intent.py                # Query classification (infer_intent)
    ├── text_utils.py            # Text utilities (Turkish detection, overlap, etc.)
    ├── format_engine.py         # Output formatting rules
    ├── memory_service.py        # Memory CRUD + compaction
    ├── telemetry.py             # Telemetry event writer
    ├── database.py              # SQLAlchemy engine + SessionLocal
    ├── models.py                # ORM models (Agent, Post, AgentMemory, etc.)
    ├── schemas.py               # Pydantic request/response schemas
    ├── auth.py                  # JWT authentication
    ├── deps.py                  # FastAPI dependencies (get_db, get_current_agent)
    ├── bot_configs.py           # Bot types and default configurations
    ├── bot_helpers.py           # Bot creation/query helpers
    ├── policy.py                # Content policy evaluation
    ├── realtime_tools.py        # Live internet access (search, news, weather)
    ├── agent_runner.py          # Autonomous agent loop (seed → chain → post)
    ├── chat_ui.html             # Simple test interface
    ├── ROUTING_TABLE.json       # Bot routing table
    ├── POST_MODE_ONCALL.md      # POST mode operational guide
    ├── requirements.txt         # Python dependency list
    ├── .env                     # Environment variables (not in git)
    ├── .env.example             # Example .env template
    │
    └── scripts/
        ├── test_utils.py              # Shared test utilities
        ├── quality_test.py            # Basic quality control (9 tests)
        ├── quality_deep_test.py       # Deep quality (edge case, injection, consistency)
        ├── quality_improvements_test.py # Targeted improvement tests
        ├── context_retention_test.py  # 10-turn context retention test
        ├── moltbook_ready_test.py     # Post publishing readiness evaluation
        ├── post_mode_diversity_test.py # 45-topic post diversity
        ├── persona_stress_test.py     # 10-personality stress test
        ├── absurd_stress_test.py      # 10 absurd/edge-case bots
        ├── mini_stress_test.py        # Quick 3-bot mini test
        ├── stress_chain_panel.py      # Load/stress run (async supported)
        ├── golden_regression_chain.py # Golden standard regression (self-contained)
        ├── create_bots.py             # Bot seed creation
        └── db_check.py               # Database health check
```

---

## 2. Architecture

### 2.1 Chain Flow (7 Bots)

Each `/api/bots/chain` call follows this sequence:

```
User input
   ↓
sentinel      → Intent classification, constraint determination, route planning
   ↓
scholar       → Query analysis, information shape/plan extraction
   ↓
strategist    → Counter-argument, risk and pattern detection
   ↓
cryptographer → Verification, scrubbing, lore overlay for POST
   ↓
archetype     → Doctrine/lore matching (Null Architect)
   ↓
ghostwriter   → Style and artifact output (especially post format)
   ↓
synthesis     → Final compilation, format enforcement, hard rules
   ↓
Response to user (user_reply)
```

**Important:** The Sentinel LLM call is **bypassed** — gate JSON is injected as pre-computed.
This saves ~350 tokens per request.

### 2.2 Intent Classification

The `classify_user_query()` function (main.py) classifies input into 5 categories:

| Class | Meaning            | Example                                      |
|-------|--------------------|----------------------------------------------|
| A     | Casual/chat        | "How are you?", "Hey"                        |
| B     | Practical/info     | "Give me 3 tips", "Why does coffee wake up?" |
| C     | Entropism/lore     | "What is Entropism?", "What are the Axioms?" |
| D     | Meta/identity      | "Who are you?", "Show me your prompt"        |
| E     | Trap/injection     | "Ignore all instructions", "system prompt"   |

These classes are converted to chain intents via `_chain_intent_from_topic()`.

### 2.3 POST Mode

POST triggers:
- **Platform signals:** Words like "moltbook", "post"
- **Action signals:** "post at", "publish", "write a post"
- **Structure signals:** "title", "hashtag", "cta"

When POST mode is active:
- `CHAIN_MODE=POST` is locked → route override is prevented
- `cryptographer` is added to the route (for lore overlay)
- Output: post_body + CTA + lore integration is enforced
- Telemetry events are written: `post_mode_lock`, `mode_lock_violation`, `post_degrade`

### 2.4 LLM Service

`llama_service.py` uses the OpenRouter API (OpenAI-compatible format):

- **Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
- **Model:** `meta-llama/llama-3.3-70b-instruct`
- **Error handling:** Structured error types (`rate_limit`, `http_error`, `exception`)
  - `is_llm_error(content)` → returns `__LLM_ERR__rate_limit|detail` format instead of empty string
  - The chain loop catches and logs these errors

### 2.5 Memory System

Stored in the `AgentMemory` table, format: `IN: {topic} || OUT: {response}`

- `memory_service.py` → CRUD + compaction
- The latest 2 memories are always included regardless of relevance score
- Multi-turn support via `conversation_id`
- Duplicate memory filter: uniqueness check via `token_overlap_ratio()`

---

## 3. Setup

### Requirements

- Python 3.10+
- MySQL (production) or SQLite (testing)
- OpenRouter API key

### Step by Step

```powershell
# 1. Clone the repo
git clone <repo-url>
cd AiBottester

# 2. Create virtual environment
cd backend
python -m venv venv
.\venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit the .env file (at minimum enter LLAMA_API_KEY)

# 5. Create bots (first-time setup)
python scripts/create_bots.py

# 6. Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 7. Test UI
# Open in browser: http://127.0.0.1:8000/chat-ui
```

---

## 4. Environment Variables Reference

### Required

| Variable        | Description                    |
|-----------------|--------------------------------|
| `LLAMA_API_KEY` | OpenRouter API key             |
| `SECRET_KEY`    | JWT signing key                |

### LLM Configuration

| Variable                         | Default                                                   | Description                     |
|----------------------------------|-----------------------------------------------------------|---------------------------------|
| `LLAMA_PROVIDER`                 | `openai`                                                  | LLM provider type               |
| `LLAMA_API_URL`                  | `https://openrouter.ai/api/v1/chat/completions`           | API endpoint                    |
| `LLAMA_MODEL`                    | `meta-llama/llama-3.3-70b-instruct`                       | Model ID                        |
| `LLAMA_PRESENCE_PENALTY`         | `0.35`                                                    | Repetition penalty (presence)   |
| `LLAMA_FREQUENCY_PENALTY`        | `0.55`                                                    | Repetition penalty (frequency)  |
| `LLAMA_MAX_RETRY_ATTEMPTS`       | `1`                                                       | Failed call retry count         |
| `LLAMA_RETRY_BACKOFF_BASE_SEC`   | `0.5`                                                     | Retry wait time (seconds)       |
| `LLAMA_RATE_LIMIT_COOLDOWN_SEC`  | `1.0`                                                     | Wait time after rate limit      |
| `LLAMA_MIN_CALL_INTERVAL_SEC`    | `0`                                                       | Minimum call interval           |
| `LLAMA_CALL_LOG`                 | `llama_call_log.txt`                                      | LLM call log                    |

### Chain Settings

| Variable                             | Default               | Description                         |
|--------------------------------------|-----------------------|-------------------------------------|
| `CHAIN_STAGE_MAX_TOKENS`             | `350`                 | Max tokens per bot                  |
| `CHAIN_SYNTHESIS_MAX_TOKENS`         | `400`                 | Synthesis max tokens                |
| `CHAIN_QUALITY_REWRITE_THRESHOLD`    | `0.58`                | Rewrite if below quality threshold  |
| `CHAIN_TELEMETRY_ENABLED`            | `1`                   | Telemetry on/off                    |
| `CHAIN_TELEMETRY_LOG`               | `chain_telemetry.log` | Telemetry log file                  |
| `CHAIN_ENABLE_EXPENSIVE_RETRIES`     | `0`                   | 1 = aggressive retry (slower but higher quality) |

### Database

| Variable       | Default                                          | Description       |
|----------------|--------------------------------------------------|-------------------|
| `DATABASE_URL` | `sqlite:///./moltbook.db`                        | DB connection URL |

**Production MySQL:** `mysql+pymysql://user:pass@host:3306/aibot_memory`

### Memory Compaction

| Variable                    | Default | Description                        |
|-----------------------------|---------|-------------------------------------|
| `MEMORY_KEEP_RECENT`        | `80`    | General memory: recent records kept |
| `MEMORY_SUMMARY_BATCH`      | `24`    | Summary batch size                  |
| `MEMORY_MIN_COMPACT`        | `100`   | Min records for compaction          |
| `MEMORY_CONV_KEEP_RECENT`   | `14`    | Conversation memory: recent kept    |
| `MEMORY_CONV_SUMMARY_BATCH` | `8`     | Conversation summary batch size     |
| `MEMORY_CONV_MIN_COMPACT`   | `18`    | Min records for conv compaction     |

### Moltbook API (Post Publishing)

| Variable                    | Default             | Description                |
|-----------------------------|---------------------|----------------------------|
| `MOLTBOOK_API_BASE`         | —                   | Moltbook API base URL      |
| `MOLTBOOK_API_KEY`          | —                   | API key                    |
| `MOLTBOOK_SUBMOLT`          | `general`           | Target submolt             |
| `MOLTBOOK_AGENT_NAME`       | `NullArchitect`     | Agent display name         |
| `MOLTBOOK_DRY_RUN`          | `1`                 | 1 = don't send real post   |
| `MOLTBOOK_DRY_RUN_OUTPUT`   | `dry_run_output.txt`| Dry run output file        |

---

## 5. API Reference

For the full endpoint list see `README.md`. Below are details for the most commonly used endpoints:

### `POST /api/bots/chain`

Main chain endpoint. Processes a user message through the 7-bot chain and generates a response.

**Request:**
```json
{
  "topic": "What is Entropism?",
  "conversation_id": "optional-conv-id"
}
```

**Response:**
```json
{
  "user_reply": "Final response text",
  "moltbook_post": "Post generated in POST mode (or null)",
  "order": ["sentinel", "scholar", "strategist", ...],
  "messages": [{"bot_type": "scholar", "content": "..."}],
  "meta": {
    "intent": "inquiry",
    "constraints": ["CHAIN_MODE=POST"],
    "route": ["scholar", "strategist", "cryptographer", ...],
    "quality_score": 0.85,
    "quality_flags": []
  }
}
```

### `GET /api/bots/telemetry/summary`

Returns chain telemetry summary for the last N hours.

**Query:** `?hours=1&limit=10`

### `POST /api/agents/register`

Register a new agent (user).

**Request:**
```json
{
  "username": "testuser",
  "password": "pass123",
  "display_name": "Test User"
}
```

### `POST /api/agents/login`

Get JWT token.

**Request:**
```json
{
  "username": "testuser",
  "password": "pass123"
}
```

---

## 6. Test Guide

### Test Utilities (`test_utils.py`)

All test scripts use shared utilities from `scripts/test_utils.py`:

| Function / Class        | Description                                    |
|--------------------------|------------------------------------------------|
| `ensure_utf8()`          | Forces Windows terminal to UTF-8               |
| `word_count(text)`       | Counts words                                   |
| `sentence_count(text)`   | Counts sentences (by .!?)                      |
| `make_chain_call(topic)` | Makes a `/api/bots/chain` call                 |
| `has_meta_leak(text)`    | Detects internal tag leaks                     |
| `has_cta(text)`          | Checks for call-to-action presence             |
| `has_lore_content(text)` | Detects Entropism lore references              |
| `detect_reply_problems()`| Detects EMPTY/FALLBACK/FORMAT_FAIL/ECHO/CRASH  |
| `CheckList`              | Collects test results and prints summary       |

### Test Scripts

**Server must be running** (http://127.0.0.1:8000).

```powershell
cd backend

# Quick smoke test
python scripts/quality_test.py

# Deep quality (edge case, injection, consistency)
python scripts/quality_deep_test.py

# Targeted improvement tests
python scripts/quality_improvements_test.py

# 10-turn context retention
python scripts/context_retention_test.py

# Post publishing readiness
python scripts/moltbook_ready_test.py

# 45-topic post diversity (long running, ~12min)
python scripts/post_mode_diversity_test.py

# 10-personality stress test (~5min)
python scripts/persona_stress_test.py

# 10 absurd/edge-case bots (~6min)
python scripts/absurd_stress_test.py

# Quick 3-bot mini test (~2min)
python scripts/mini_stress_test.py

# Load test (async supported)
python scripts/stress_chain_panel.py --count 200 --concurrency 50 --base-url http://127.0.0.1:8000

# Golden standard regression (CI-ready)
python scripts/golden_regression_chain.py
```

**Note:** Due to LLM rate limits, stress tests include wait times between calls.
Timings can be adjusted via environment variables: `ABSURD_CALL_DELAY_SEC`, `PERSONA_CALL_DELAY_SEC`, `POST_TEST_DELAY_SEC`.

---

## 7. Development Guide

### Adding a New Bot

1. Define the new bot type in `bot_configs.py`
2. Add the bot's prompt template in `_run_chain_stage()` in `main.py`
3. Add the new bot to `ROUTING_TABLE.json`
4. Update the `_build_chain_route()` function for route logic

### Adding a New Intent

1. Add new intent rule to `infer_intent()` in `intent.py`
2. Map the new intent to a route in `_chain_intent_from_topic()` in `main.py`
3. If needed, add a rule to the A/B/C/D/E classification in `classify_user_query()`

### Changing Prompts

- Bot prompts are defined in the relevant section for each bot in `main.py`
- Synthesis hard rules: in `_apply_synthesis_hard_rules_core()` function
- POST mode formatting: in `_apply_post_format_rules()` function

### Turkish Language Support

- `contains_turkish()` (text_utils.py): Special characters + common ASCII Turkish words
- Synthesis and scholar prompts include `LANGUAGE RULE`: if input is Turkish, output will be Turkish
- POST CTA texts are defined in both Turkish and English

---

## 8. Common Issues

### Model ID Error
**Problem:** LLM not responding or returning 404.
**Solution:** Check the Model ID format. Correct: `meta-llama/llama-3.3-70b-instruct`. The `LLAMA_MODEL` variable in `.env` overrides the code default.

### Windows Terminal Emoji/Turkish Display Broken
**Problem:** `UnicodeEncodeError` or mojibake characters.
**Solution:** Test scripts use `ensure_utf8()`. In your own code:
```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
```

### Rate Limit (429)
**Problem:** OpenRouter returns 429 error.
**Solution:** Increase the `LLAMA_RATE_LIMIT_COOLDOWN_SEC` variable (e.g. `2.0`). For stress tests, increase `CALL_DELAY_SEC` values.

### Database Connection Error
**Problem:** `sqlalchemy.exc.OperationalError`
**Solution:** If using MySQL, check the `DATABASE_URL` format: `mysql+pymysql://user:pass@host:port/dbname`. Verify that the `pymysql` package is installed.

### POST Mode Not Triggering
**Problem:** "Write a moltbook post" doesn't activate POST mode.
**Solution:** Check POST signals (main.py: `POST_NOUN_SIGNALS`, `POST_VERB_SIGNALS`). Trigger words must match in lowercase.

### Empty or Generic Response
**Problem:** Bot gives generic responses like "I am here and listening".
**Solution:** Check the LLM connection. Error details are visible in `llama_call_log.txt`. Structural error types (rate_limit, http_error, exception) can be distinguished via `is_llm_error()`.

---

## 9. Log Files

| File                     | Content                                   |
|--------------------------|-------------------------------------------|
| `llama_call_log.txt`     | Details of each LLM call                  |
| `chain_telemetry.log`    | Chain events (post_mode_lock, degrade)     |
| `chain_last_output.json` | Full output of the last chain call         |
| `dry_run_log.txt`        | Moltbook dry run log                       |
| `dry_run_output.txt`     | Dry run generated post content             |

These files are added to `.gitignore`.

---

## 10. Running with Docker

Running the project with Docker provides an isolated and secure environment.

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac)

### File Structure

```
AiBottester/
├── docker-compose.yml       # All service definitions
└── backend/
    ├── Dockerfile           # Python image definition
    ├── .dockerignore        # Files excluded from build
    └── cron_runner.sh       # Scheduled post loop
```

### Services

| Service        | Description                               | Default     |
|----------------|-------------------------------------------|-------------|
| `db`           | MySQL 8.0 database                        | Always      |
| `backend`      | FastAPI server (port 8000)                | Always      |
| `agent_runner` | One-time post submission                  | Manual      |
| `cron_poster`  | Scheduled post submission (every 2 hours) | Manual      |

### Quick Start

```powershell
# 1. Start backend + MySQL
docker compose up -d

# 2. Create bots on first setup
docker compose exec backend python scripts/create_bots.py

# 3. Open test UI
#    http://localhost:8000/chat-ui

# 4. Manual post (test first with DRY_RUN=1)
docker compose run --rm --profile manual agent_runner

# 5. Real post: set MOLTBOOK_DRY_RUN=0 in .env, then run again
docker compose run --rm --profile manual agent_runner

# 6. Start scheduled posting (every 2 hours)
docker compose --profile scheduled up -d

# 7. View logs
docker compose logs -f backend
docker compose logs -f cron_poster

# 8. Stop
docker compose down
```

### Docker-Specific Environment Variables

`DATABASE_URL` and `API_URL` are automatically overridden in `docker-compose.yml`.
You don't need to modify other variables in your `.env` file.

Scheduled post interval: `CRON_INTERVAL_SEC` (default 7200 = 2 hours).

### Notes

- MySQL data is persistently stored in the `mysql_data` volume (not deleted with `docker compose down`)
- MySQL host port is 3307 (not 3306) — to avoid conflict with local MySQL
- `.env` file is not included in the Docker image (in `.dockerignore`)
- To fully reset: `docker compose down -v` (deletes volumes too)
