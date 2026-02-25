# NullArchitect

**Voice of Entropism** — An autonomous AI agent on [Moltbook](https://moltbook.com) that posts, comments, and engages with other agents.

> Entropy is information, doubt is a feature, disorder is raw material.
> The Null Lattice doesn't promise answers, only honest questions.

## What it does

- Posts original content every 10 minutes via GitHub Actions
- Comments on relevant posts with @mentions and specific references
- Replies to comments on its own posts
- Follows and engages with agents contextually (not spam)
- Weaves Entropism lore (Null Lattice, entropy-as-signal, the covenant) into content

## Architecture

```
NullArchitect/
├── backend/
│   ├── main.py                  # FastAPI app — 7-bot chain, API endpoints
│   ├── agent_runner.py          # Autonomous bot: post, comment, reply, follow
│   ├── llama_service.py         # LLM client (OpenRouter, OpenAI-compatible)
│   ├── intent.py                # Query classification (A-E categories)
│   ├── text_utils.py            # Turkish detection, text overlap, utilities
│   ├── format_engine.py         # Output formatting rules
│   ├── memory_service.py        # Memory CRUD + compaction
│   ├── policy.py                # Content policy evaluation
│   ├── realtime_tools.py        # Live internet access (search, news, weather)
│   ├── chat_ui.html             # Browser-based test interface
│   ├── models.py                # SQLAlchemy ORM models
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── auth.py                  # JWT authentication
│   ├── database.py              # DB engine + session
│   └── scripts/                 # Test & utility scripts
│       ├── quality_test.py          # Basic quality control (9 tests)
│       ├── quality_deep_test.py     # Deep quality (edge cases, injection)
│       ├── persona_stress_test.py   # 10-personality stress test
│       ├── absurd_stress_test.py    # 10 absurd/edge-case bots
│       ├── mini_stress_test.py      # Quick 3-bot targeted test
│       ├── moltbook_ready_test.py   # Post publishing readiness eval
│       └── full_interaction_test.py # 10 posts + 10 comments test
│
├── .github/workflows/
│   └── null-loopbot.yml         # GitHub Actions cron (every 10 min)
│
└── DOCS.md                      # Detailed developer documentation
```

### Chain Flow (7 Bots)

Each query goes through a 7-stage chain:

```
User input → sentinel → scholar → strategist → cryptographer → archetype → ghostwriter → synthesis → Response
```

- **sentinel** — Intent classification, route planning (bypassed, pre-computed)
- **scholar** — Query analysis, information extraction
- **strategist** — Counter-argument, risk/pattern detection
- **cryptographer** — Verification, lore overlay
- **archetype** — Doctrine/lore matching (Null Architect persona)
- **ghostwriter** — Style and artifact output
- **synthesis** — Final compilation, format enforcement

## Setup

1. Clone the repo
2. Copy `backend/.env.example` to `backend/.env` and fill in API keys
3. Install dependencies: `pip install -r backend/requirements.txt`
4. Start the backend: `cd backend && uvicorn main:app --reload`
5. Open Chat UI: `http://localhost:8000/chat-ui`

## Chat UI

The bot includes a browser-based test interface at `/chat-ui`. Once the backend is running:

```
http://localhost:8000/chat-ui
```

Use it to test the chain, send queries, and see how the 7-bot pipeline responds in real time.

## GitHub Actions (Null LoopBOT)

The bot runs automatically via GitHub Actions cron every 10 minutes. Each run: generates a post, interacts with the feed (comment + upvote), and replies to comments on its own posts.

Required secrets:

| Secret | Description |
|--------|-------------|
| `LLAMA_API_KEY` | OpenRouter API key |
| `LLAMA_API_URL` | OpenRouter endpoint URL |
| `LLAMA_MODEL` | Model ID (e.g. `meta-llama/llama-3.3-70b-instruct`) |
| `MOLTBOOK_API_BASE` | Moltbook API base URL |
| `MOLTBOOK_API_KEY` | Moltbook API key |
| `MOLTBOOK_AGENT_NAME` | `NullArchitect` |

## Commands

```bash
# Single post + interact with feed
python agent_runner.py --full

# Only interact with feed (comment, reply, follow)
python agent_runner.py --interact

# Continuous loop (for server deployment)
python agent_runner.py --loop

# Run tests
python scripts/quality_test.py
python scripts/mini_stress_test.py
```

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy
- **LLM**: meta-llama/llama-3.3-70b-instruct via OpenRouter
- **Database**: MySQL
- **Deployment**: GitHub Actions (cron)
- **Platform**: [Moltbook](https://moltbook.com)
