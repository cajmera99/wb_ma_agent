# M&A Acquirer Identification Engine

A production-grade agentic system that identifies the 10 most likely acquirers for a target company using 500 historical M&A transactions and LLM synthesis.

Built for the William Blair AI Innovation Team take-home assessment.

---

## Architecture

```
frontend (React/Vite)
    └── POST /api/analyze          → starts a run, returns run_id
    └── GET  /api/runs/{id}/stream → SSE event stream (live progress)
    └── GET  /api/runs/{id}/result → final rationales JSON
    └── GET  /api/runs/{id}/pdf    → download 10-page PDF

backend (FastAPI)
    └── Lifespan startup
        ├── Load CSV once (feature engineering pre-computed)
        ├── Build acquirer profiles (groupby aggregations)
        ├── Instantiate ChatOpenAI (GPT-4o)
        └── Build LangChain tools (closed over static data)

    └── LangGraph Agent (per request, runs in BackgroundTask)
        ├── Node 1: score_and_rank       — pure Python multi-factor scoring
        ├── Node 2: evaluate_coverage    — deterministic routing decision
        ├── Node 3: expand_candidate_pool (conditional) — widens pool if thin
        ├── Node 4: llm_rerank           — LLM with tool-calling selects final 10
        └── Node 5: generate_rationales  — 10 concurrent LLM calls → Pydantic validated
```

## Agent Graph (LangGraph StateGraph)

```
score_and_rank → evaluate_coverage ──┬──(sufficient)──→ llm_rerank → generate_rationales
                                     └──(thin)──→ expand_candidate_pool ──→ llm_rerank
```

The conditional edge is driven by a deterministic Python function (`route_after_coverage`), not the LLM. Routing decisions that can be made with code should not consume LLM tokens.

## Scoring Model (6 Dimensions)

| Dimension | Weight | Method |
|-----------|--------|--------|
| Sector affinity | 35% | Primary (1.0) / Adjacent (0.7) / Secondary (0.3) |
| Deal size match | 20% | Gaussian decay around target EV ±60% |
| Rationale tag alignment | 20% | Fraction of HIGH_RELEVANCE_TAGS matched |
| Recency | 10% | Stale penalty + recent deal count bonus |
| Outcome quality | 10% | Closed / total deals ratio |
| Ownership match | 5% | Private + PE-backed / total |

Composite score = weighted sum × 100, range 0–100.

## LangChain Tools

| Tool | Purpose |
|------|---------|
| `search_transactions` | Filter CSV by sector/size/outcome — evidence gathering |
| `get_acquirer_profile` | Full profile + stats for a named acquirer |
| `get_acquirer_precedent_deals` | Up to N deals for one acquirer, sorted by relevance |
| `get_valuation_comps` | EV/EBITDA and EV/Revenue stats from closed comps |

Tools are built at startup via factory functions closed over the pre-loaded DataFrame. They are given to the LLM via `llm.bind_tools()` in `llm_rerank` — the LLM decides which to call and with what arguments.

## Observability

Every node in the graph emits typed events to a `RunStore` via an `EventEmitter`. Events are:
- Written to an in-memory log (persists for the server lifetime)
- Pushed to a per-run `asyncio.Queue` for live SSE delivery

The frontend subscribes to `GET /api/runs/{id}/stream` immediately after run creation. Events arrive in real-time as the agent progresses through nodes.

## Structured Output + Repair Loop

Rationale generation uses `llm.with_structured_output(AcquirerRationale)` — OpenAI function-calling constrained to a Pydantic schema. If validation fails:
1. The error is emitted as `validation.failed`
2. A repair message with the exact validation error is appended to the conversation
3. The LLM retries once
4. If the repair also fails, a stub page is inserted so the PDF still has 10 pages

## Concurrency

All 10 rationale LLM calls are fired simultaneously with `asyncio.gather`. Total latency is ~1 LLM call duration, not 10× sequential.

Per-run isolation is guaranteed: each run gets its own `asyncio.Queue`. Events from different concurrent runs cannot bleed into each other.

Approximate capacity on OpenAI Tier 1 (TPM limits):
- ~2–3 concurrent banker requests before rate-limit latency appears
- **Production path**: add `asyncio.Semaphore(3)` around the graph invocation in `_run_agent`

## Known Trade-offs and Production Upgrade Path

### In-memory state (current)
`RunStore` stores all runs and events in Python dicts. History persists for the lifetime of the server process but is lost on restart.

The frontend mitigates this at the browser level: on page load it fetches `/api/runs`, finds the most recent completed run, and restores the result automatically. This makes browser refreshes seamless as long as the server has not restarted.

**Recommended next step — SQLite persistence:**

Replace the in-memory dicts in `backend/services/run_store.py` with a local SQLite database using Python's built-in `sqlite3` module (no new dependencies). Schema:

```sql
CREATE TABLE runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,          -- 'running' | 'completed' | 'failed'
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    target       TEXT NOT NULL           -- JSON blob of TargetProfile
);
CREATE TABLE events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    node         TEXT,
    data         TEXT NOT NULL,          -- JSON blob
    timestamp    TEXT NOT NULL
);
```

Key implementation notes:
- `create_run()` → `INSERT INTO runs`
- `add_event()` → `INSERT INTO events` + `UPDATE runs SET status=...` on terminal events + `queue.put_nowait()` for live SSE (queue stays in-memory)
- `list_runs()` / `get_events()` → standard `SELECT` queries
- Use `check_same_thread=False` on the connection and a `threading.Lock` for writes since FastAPI sync routes run in a thread pool
- Mark any runs still `status='running'` as `'failed'` on startup (`UPDATE runs SET status='failed' WHERE status='running'`) — they cannot complete after a restart
- PDF files already persist on disk in `backend/output/` so no changes needed there

This upgrade makes run history and all event logs survive server restarts indefinitely with zero infrastructure dependencies.

**Further production upgrade**: replace SQLite with Postgres tables and `asyncio.Queue` with Redis pub/sub channels for horizontal scaling across multiple server instances.

### LLM call count (current: ~12 per run)
- 1 call: `llm_rerank` (may trigger up to 3 tool rounds)
- 10 calls: `generate_rationales` (concurrent)
- 1–2 extra: tool call rounds in rerank

**Optimisation options**:
1. Combine rerank + top-3 rationale into one call (reduces calls by 3)
2. Use `gpt-4o-mini` for rerank (cheaper, still adequate for ranking)
3. Cache valuation comps across a run (they're identical for all 10 acquirers)

### Adjacent sector broadening
Healthcare Services has ~46 transactions in the dataset; ~12 in the $100–400M range. The scoring model weights adjacent sectors (Behavioral Health, Physician Groups, Home Health/Hospice) at 0.7× to include relevant buyers without polluting the Primary sector signal.

## Setup

### Prerequisites
- Python 3.11+
- Node 18+
- OpenAI API key

### Backend

```bash
cd William_Blair_IB_Project
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Create .env with your key
echo OPENAI_API_KEY=sk-... > .env
echo OPENAI_MODEL=gpt-4o >> .env

# Place CSV at: data/ma_transactions_500.csv

uvicorn backend.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","acquirers_loaded":85,"transactions_loaded":500}
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Start a run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Run metadata |
| GET | `/api/runs/{id}/events` | Full event log |
| GET | `/api/runs/{id}/stream` | Live SSE stream |
| GET | `/api/runs/{id}/result` | Final rationales JSON |
| GET | `/api/runs/{id}/pdf` | Download PDF report |
| GET | `/health` | Server health |

## Sample Request

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sector": "Healthcare Services",
    "deal_size_mm": 200,
    "geography": "Midwest",
    "ebitda_margin_pct": 18,
    "revenue_growth_pct": 12,
    "ownership": "Private"
  }'
# {"run_id":"abc123...","stream_url":"/api/runs/abc123.../stream","events_url":"..."}
```
