# CLAUDE.md — M&A Acquirer Identification Engine

Developer guide for AI assistants and new contributors. Documents architecture decisions, data flow, and non-obvious implementation choices.

---

## Project Overview

William Blair AI Innovation Team take-home assessment. Given a target company profile and 500 historical M&A transactions, identify the 10 most likely acquirers and generate a one-page rationale for each. Output is a 10-page PDF.

Graded on: agentic system design, LLM prompt quality, structured output validation, production-grade patterns (tool use, dynamic routing, observability), and code quality.

---

## How to Run

```bash
# Backend — restrict reload watcher to backend/ so frontend file saves don't
# restart the server and wipe the in-memory RunStore
conda activate wb_ib_env        # Python 3.11, created for this project
cd William_Blair_IB_Project
uvicorn backend.main:app --reload --reload-dir backend --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev                     # → http://localhost:5173
```

> **Windows / Chrome note:** Vite's dev proxy is pointed at `http://127.0.0.1:8000`
> (not `localhost`). On Windows, `localhost` can resolve to `::1` (IPv6) while
> uvicorn only binds to `127.0.0.1` (IPv4), causing silent proxy failures.
> Never change the proxy target back to `localhost`.

**Required:** `.env` file in project root:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

**Required:** `data/ma_transactions_500.csv` (the 500-row dataset provided in the assessment).

---

## Repository Structure

```
William_Blair_IB_Project/
├── backend/
│   ├── main.py                     # FastAPI app, lifespan startup
│   ├── core/
│   │   ├── config.py               # Pydantic settings, scoring weights
│   │   ├── loader.py               # CSV load + validation (once at startup)
│   │   ├── profiler.py             # Acquirer profile aggregation
│   │   └── scorer.py               # 6-dimension scoring model
│   ├── models/
│   │   ├── target.py               # TargetProfile (request input)
│   │   ├── rationale.py            # AcquirerRationale + sub-models (Pydantic)
│   │   └── events.py               # EventType enum + RunEvent
│   ├── agent/
│   │   ├── graph.py                # LangGraph StateGraph assembly
│   │   ├── state.py                # AgentState TypedDict
│   │   ├── prompts.py              # All LLM prompts (centralised)
│   │   ├── tools/
│   │   │   ├── __init__.py         # build_tools() factory
│   │   │   ├── search_transactions.py
│   │   │   ├── get_acquirer_profile.py
│   │   │   ├── get_acquirer_precedent_deals.py
│   │   │   └── get_valuation_comps.py
│   │   └── nodes/
│   │       ├── score_and_rank.py
│   │       ├── evaluate_coverage.py
│   │       ├── expand_candidate_pool.py
│   │       ├── llm_rerank.py
│   │       └── generate_rationales.py
│   ├── api/
│   │   ├── deps.py                 # FastAPI dependency injection
│   │   └── routes/
│   │       ├── analyze.py          # POST /api/analyze
│   │       └── runs.py             # GET /api/runs/...
│   ├── services/
│   │   ├── app_state.py            # AppState dataclass (single instance)
│   │   ├── run_store.py            # In-memory run + event storage
│   │   ├── event_emitter.py        # Thin wrapper: emit → RunStore
│   │   └── pdf_generator.py        # reportlab 10-page PDF
│   └── observability/
│       └── logging.py              # structlog configuration
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Root layout + state
│   │   └── components/
│   │       ├── TargetForm.jsx      # Input form (defaults pre-filled)
│   │       ├── RunProgress.jsx     # Live SSE event log
│   │       ├── AcquirerCard.jsx    # Collapsible rationale card
│   │       └── RunHistory.jsx      # Sidebar, polls /api/runs every 5s
│   ├── vite.config.js              # Proxies /api → localhost:8000
│   └── package.json
├── data/
│   └── ma_transactions_500.csv     # Assessment dataset (committed for Railway)
├── backend/output/                 # Generated PDFs (created at startup)
├── Dockerfile                      # Multi-stage build: Node → React dist, Python → FastAPI
├── railway.json                    # Forces Railway to use Dockerfile builder (not Railpack)
├── .dockerignore                   # Excludes __pycache__, .env, node_modules, .git
├── .env                            # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Agent Graph (LangGraph StateGraph)

```
score_and_rank
      │
evaluate_coverage ──── route_after_coverage() ────┐
      │                                             │
   (sufficient)                              (thin coverage)
      │                                             │
  llm_rerank  ◄──────── expand_candidate_pool ◄───┘
      │
generate_rationales
      │
     END
```

All five nodes are in `backend/agent/nodes/`. Each receives `state` (AgentState TypedDict) and `config` (RunnableConfig). Shared dependencies (`emitter`, `app_state`) travel via `config["configurable"]` — not as globals or module-level imports.

### Why this graph shape?

The original design had `search_transactions` called twice by the LLM to "discover" thin coverage. That was rejected as theatrical: the dataset is static, so the LLM would always find the same 12 Healthcare Services records. The routing decision was moved to a deterministic Python function (`route_after_coverage`). The LLM is involved only in decisions that require qualitative judgment — not ones computable with a comparison operator.

---

## Data Flow (per request)

1. `POST /api/analyze` → creates a `run_id`, calls `RunStore.create_run()`, launches `_run_agent()` as a FastAPI BackgroundTask, returns `run_id` immediately.

2. `_run_agent()` emits `RUN_STARTED`, builds the initial AgentState, invokes `compiled_graph.ainvoke()`.

3. **score_and_rank**: calls `rank_acquirers()` — pure Python, no LLM. Scores all 107 acquirers in ~5ms and returns them sorted.

4. **evaluate_coverage**: checks how many scored above 30.0. Routes to `expand_candidate_pool` if fewer than 15 pass. Otherwise goes straight to `llm_rerank`. Always sets `top_candidates` to top-20 from the scored list.

5. **expand_candidate_pool** (conditional): takes top-25 instead of top-20. Lets the LLM see a wider pool when Healthcare Services data is thin.

6. **llm_rerank**: LLM receives the top-N candidate summaries and may call `get_acquirer_profile` or `search_transactions` to dig deeper before committing to a ranking. Tool-calling loop runs up to 3 rounds. Returns a ranked list of 10 names.

7. **generate_rationales**: fires 10 LLM calls concurrently via `asyncio.gather`, throttled to 5 at a time by `asyncio.Semaphore(5)`. Each call uses `gpt-4o-mini` (`llm_fast`) — rationale generation is pure synthesis from pre-assembled data, not tool-use judgment. Each call:
   - **Pre-computes citation-anchor fields** from the acquirer profile before building the prompt. These give the LLM the exact numbers needed for the grader-expected sentence pattern ("This acquirer has completed X deals in [sector] in the $Y–$Z range at Mx EV/EBITDA"): `primary_sector_deal_count` (exact count of deals in the target's sector, from `sector_counts[target.sector]`), `deals_near_target` (deals within 0.5×–2.0× of target EV, computed from the `deal_sizes_mm` array stored in the profile), `target_size_band` ("$100M–$400M" for a $200M target), `deal_size_range` (min–max string), and the full `sector_counts` dict. The Section 1 prompt instruction renders these as actual numbers so the LLM's example is already acquirer-specific before any generation occurs.
   - **Pre-computes 7 Python anomaly signals** before touching the LLM. These are injected into the prompt as `⚠`/`✓` attention markers. Signals: deal size (4-branch: GENUINE STRETCH / AT-SIZE RANGE COVERS TARGET / BELOW MEDIAN / AT-SIZE), completion rate (⚠ if outcome_score < 70; ✓ block if ≥ 70), ownership mismatch (⚠ if ownership_score < 25), valuation posture (ABOVE-MARKET with `turns_diff` = additive turn difference and `gap_pct` = % above market; BELOW-MARKET with `stretch_pct` = % above acquirer's own median — **not** market median; AT-MARKET), oversized precedent deals (⚠ listing each deal >3× target EV by name with its ratio, mandatory disclosure when cited), and an unconditional ⚠ blocking "target's EBITDA margins complement…" in all sections.
   - **Valuation math rules**: For below-market acquirers, the stretch % uses the acquirer's own historical median as the denominator ("must bid X% above historical comfort") — not the market median. For above-market acquirers, risk label uses "+N turns above market" (additive turn count) not "Nx Premium" (which implies N times the market price).
   - **Canonical name lookup**: uses `candidate.get("acquirer_name")` (from the scored profile, exact CSV string) for all tool invocations — not the LLM's rerank-output name, which may be subtly altered.
   - **Two-pass precedent deal fetch**: first fetches up to 5 deals filtered to the target's primary sector, then fetches up to 10 deals with no sector filter and fills remaining slots with non-duplicates. Combined list capped at 5. Sector-relevant deals appear first.
   - Fetches valuation comps via `get_valuation_comps` tool (Python, not LLM).
   - Builds the full evidence packet and calls `llm_fast.with_structured_output(AcquirerRationale)`.
   - **Conviction level enforced in Python** post-generation: `result["conviction_level"] = conviction_baseline` (composite > 80 → High; 50–79 → Medium; < 50 → Low). The LLM writes rationale text calibrated to the level but never controls the label.
   - **Post-generation EBITDA content scan**: after the LLM returns, a regex scans all text fields (`acquirer_overview`, `strategic_fit_thesis`, `conviction_rationale`, risk flag descriptions) for "the target's EBITDA margins" or variants. If found, emits `VALIDATION_FAILED` and fires a targeted repair call quoting the exact violation. Three prompt layers already forbid the phrase — this scan catches what instruction-following alone misses.
   - On Pydantic validation failure, runs one repair loop with the error message.
   - On repair failure, inserts a stub entry so the PDF still has 10 pages.

8. `_run_agent()` **sorts rationales by `composite_score` descending** and reassigns `rank` numbers before calling `generate_pdf()`. This ensures conviction level (Python-derived from composite score) always matches position order in both the PDF and UI — the LLM rerank selects *which* 10 make the shortlist but does not control the final ordering. Calls `generate_pdf()` via `asyncio.get_running_loop().run_in_executor()` (not `get_event_loop()` — the latter is deprecated inside a running async function), then emits `RUN_COMPLETED` with rationales + PDF URL.

---

## Startup: What Happens Once vs Per Request

**Once at server startup (lifespan):**
- `load_transactions()` — reads CSV, validates columns, parses rationale tags, flags adjacent-sector rows
- `build_acquirer_profiles()` — groupby aggregations across all 107 acquirers; result is a `dict[str, dict]` held in memory
- `ChatOpenAI(...)` — single LLM client instance, reused on every request
- `build_tools(df, profiles)` — 4 LangChain tools built via factory functions closed over the pre-loaded data
- All of the above stored in `AppState` (a dataclass on `app.state`)

**Per request:**
- `run_id` generated (UUID)
- `asyncio.Queue` created in RunStore for that run's SSE stream
- LangGraph invoked with the above shared state passed through `config["configurable"]`

The graph itself (`compiled_graph`) is also built once at module import time and shared. LangGraph's compiled graph holds no per-run state — it's safe to share.

---

## Scoring Model

Six dimensions, each returns 0.0–1.0, multiplied by weight and summed to a 0–100 composite.

| Dimension | Weight | Logic |
|-----------|--------|-------|
| Sector affinity | 35% | Primary sector = 1.0, Adjacent = 0.7, Secondary = 0.3, Other = 0. Normalised by total deals. |
| Deal size match | 20% | Gaussian decay: `exp(-0.5 * ((median - target) / (target * 0.6))^2)`. An acquirer whose median deal exactly matches the target scores 1.0. |
| Rationale tags | 20% | Fraction of the acquirer's top tags that appear in `HIGH_RELEVANCE_TAGS` (Platform Build, Geographic Expansion, High Growth, Margin Improvement, Vertical Integration, Bolt-on, Cost Synergies). |
| Recency | 10% | 50% from decay since last deal (−0.15/year) + 50% from recent deal count (3+ = full score). |
| Outcome quality | 10% | Closed deals / total deals. |
| Ownership match | 5% | Fraction of past targets with matching ownership type. Private and PE-Backed are treated as equivalent ("private-side"). |

Sub-scores are stored per-acquirer in the `scored_candidates` list at 0–100 scale. The PDF score bars and rationale prompts receive these directly — do **not** multiply by 100 again anywhere downstream.

**Adjacent sector definition:** Healthcare Services, Behavioral Health, Physician Groups, Home Health/Hospice all count as primary or adjacent. The dataset has ~46 Healthcare Services transactions, ~12 in the $100–400M size band. Weighting adjacents at 0.7 is what makes the model produce meaningful differentiation despite thin primary-sector data.

---

## LangChain Tools

Tools are built via factory functions in `backend/agent/tools/` and registered in `build_tools()`. The factory pattern closes over the pre-loaded `df` and `profiles` so the tools don't need to import globals or receive data at call time.

| Tool | Used in | Purpose |
|------|---------|---------|
| `search_transactions` | `llm_rerank` (LLM-driven) | Filter CSV by sectors, deal size range, outcomes |
| `get_acquirer_profile` | `llm_rerank` (LLM-driven) | Full aggregated profile for one acquirer |
| `get_acquirer_precedent_deals` | `generate_rationales` (Python-driven) | Top 5 deals for one acquirer, Closed-first |
| `get_valuation_comps` | `generate_rationales` (Python-driven) | EV/EBITDA + EV/Revenue stats from closed comps |

**Important distinction:** In `llm_rerank`, the LLM decides whether to call tools and with what arguments — real agentic tool use. In `generate_rationales`, tools are called directly in Python to assemble the evidence packet before the LLM is invoked — the LLM synthesises, not retrieves. This separation keeps latency down and avoids the LLM fetching data it won't use.

---

## Structured Output + Repair Loop

`AcquirerRationale` is a Pydantic v2 model with enforced constraints: `min_length=1` on `precedent_deals`, `min_length=2` on `risk_flags`, `Literal` types on `acquirer_type` and `conviction_level`.

Generation uses `llm.with_structured_output(AcquirerRationale)` which binds the Pydantic schema as an OpenAI function call — the API rejects responses that don't conform to the shape. There are two distinct repair triggers:

**Trigger 1 — Pydantic schema violation** (shape/type error):
1. Emit `VALIDATION_FAILED` event
2. Append the exact Pydantic error to the conversation as a repair prompt
3. Call `llm_structured.ainvoke()` once more
4. On second failure: log error, insert a stub dict so the PDF still renders all 10 pages

**Trigger 2 — EBITDA content policy violation** (phrase detected in output text):
1. Post-generation regex scan checks all text fields for "the target's EBITDA margins" or variants
2. If matched: emit `VALIDATION_FAILED`, send a targeted repair message quoting the exact violation and explaining why it is unfounded (target has no disclosed margin %)
3. On repair success: emit `VALIDATION_REPAIRED`, re-apply Python overrides (rank, scores, conviction)
4. On repair failure: log error, keep original output (better than a stub)

---

## Observability (SSE Events)

Every meaningful state change emits a typed `RunEvent` via `EventEmitter.emit()`. Events are:
- Appended to `RunStore._events[run_id]` (the persistent log)
- `put_nowait`'d into `RunStore._queues[run_id]` (the live SSE queue)

The SSE stream endpoint (`GET /api/runs/{id}/stream`) first replays all past events (for late-connecting clients), then tails the queue until a terminal event (`run.completed` or `run.failed`) arrives.

**Event types defined in `EventType` enum:**
`run.started`, `node.started`, `node.completed`, `node.error`, `routing.decision`, `tool.called`, `tool.result`, `rationale.generated`, `validation.failed`, `validation.repaired`, `llm.tokens_used`, `run.completed`, `run.failed`

Run history survives page refreshes because it's stored in the RunStore for the server's lifetime. It does **not** survive server restarts (in-memory). See production upgrade path below.

**SSE stream vs event log:** `/stream` is a live tail that closes after the terminal event; `/events` is a static dump of all stored events and works at any time after the run starts. Once a run completes both return the same data.

**`--reload` kills history:** Every Python file save triggers a uvicorn restart which wipes RunStore. Use `--reload-dir backend` to limit restarts to backend changes, and never save Python files mid-run.

---

## Dependency Injection Pattern

`app.state.app_state` and `app.state.run_store` are set during lifespan startup. Route handlers access them via:

```python
def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state
```

Agent nodes access them via `config["configurable"]["app_state"]`.

Neither pattern requires importing from `main.py`, which avoids a circular import (routes imported by main → would import main → circular). If you need to add a new shared resource, add it to `AppState`, set it in `lifespan`, and read it via `deps.py`.

---

## Concurrency Model

- Each `POST /api/analyze` runs in a FastAPI `BackgroundTask` (asyncio, not a thread)
- 10 rationale LLM calls per run fire simultaneously via `asyncio.gather`
- Per-run `asyncio.Queue` ensures event streams from different runs don't bleed
- OpenAI Tier 1 limit: roughly 2–3 concurrent banker sessions before rate-limit latency
- Production fix: `asyncio.Semaphore(N)` wrapping the `compiled_graph.ainvoke()` call in `_run_agent()`

---

## PDF Generation

`generate_pdf()` in `backend/services/pdf_generator.py` uses reportlab Platypus (flow-based layout). Called synchronously from async context via `asyncio.get_running_loop().run_in_executor(None, generate_pdf, ...)` to avoid blocking the event loop. Use `get_running_loop()` not `get_event_loop()` — the latter is deprecated when called inside a running async function and raises a DeprecationWarning in Python 3.10+.

Output: `backend/output/{run_id}.pdf`. The directory is created at startup. Served via `GET /api/runs/{run_id}/pdf` which returns a `FileResponse`.

**Download filename** is constructed dynamically from the run's stored target: `{Sector_With_Underscores}_Acquirer_Analysis_{MMDDYYYY}.pdf` (e.g. `Healthcare_Services_Acquirer_Analysis_06142026.pdf`). The run_id is looked up in RunStore at download time to retrieve the sector and date.

**Layout constants:**
- `PAGE_W = 7.3 * inch` — usable content width (letter 8.5" minus 0.6" margins each side). Every table uses `PAGE_W` so changing the margin requires only updating this one constant.
- Margins: 0.6" left/right, 0.6" top/bottom.

**Cover page** contains:
1. Blue header banner (full `PAGE_W`)
2. Target Company Profile table
3. Acquirer Scoring Methodology table — all cells are `Paragraph` objects (not plain strings) so text wraps within column bounds rather than clipping. Plain strings in reportlab table cells do not wrap.
4. Color legend for score thresholds
5. Confidentiality footer (drawn via canvas callback, not a flowable)

**Per-acquirer page structure:**
1. Blue header banner — rank + name (44% width), acquirer type + score + conviction (56% width, 8pt font). Right cell is wider because "Financial Sponsor | Score: 100.0/100 | ■ Medium Conviction" needs ~3.9" at 8pt.
2. Score grid — 3-column × 2-row table. Each cell is a nested `Table` containing a label `Paragraph` and a color-coded score `Paragraph`. Color thresholds: green ≥ 70, amber ≥ 40, red < 40. Uses hardcoded hex strings for colors (not `HexColor.__int__()` which is unreliable across reportlab versions).
3. Sections 1–6: Overview, Thesis, Precedent Activity (table, max 5 deals), Valuation Context, Risk Flags (severity-tagged), Conviction.

**Footer:** `_draw_page_footer()` is a canvas-level callback passed to `doc.build(onFirstPage=..., onLaterPages=...)`. It draws "Prepared: [date] | Confidential — For Internal Use Only" in light gray at 0.3" from the bottom on every page. Canvas callbacks are immune to flowable overflow — they always render at the fixed position regardless of page content length.

**Precedent deals table** column widths sum to `PAGE_W`: Transaction 2.5", Size 0.85", Type 1.5", EV/EBITDA 1.0", Outcome 1.45". Type column was widened from 1.0" to 1.5" to accommodate "Platform Acquisition" etc. without truncation.

**Do not use ASCII bar characters** (█ ░) for score visualisation — they wrap unpredictably in reportlab. The numeric score grid (label + colored number) replaced them.

---

## Prompt Design Decisions

All prompts live in `backend/agent/prompts.py`. Key choices:

**SYSTEM_PROMPT** — establishes the banker persona and two non-negotiable rules applied to every LLM call:
- Every claim must cite a specific number from the provided data
- Strategic vs Financial Sponsor theses must be framed differently (one is synergy-driven, the other is return-on-capital-driven)

**RERANK_PROMPT_TEMPLATE** — gives the LLM the quantitative scores and asks for qualitative override. Explicitly instructs: prefer diversity of thesis over clustering similar buyers, consider bolt-on vs platform logic for PE sponsors.

**RATIONALE_PROMPT_TEMPLATE** — the most critical prompt. The LLM receives:
- The full acquirer M&A profile (deal counts, multiples, tags, geography)
- Score breakdown per dimension
- Precedent deals JSON (from `get_acquirer_precedent_deals`) — note `acquired_co_ebitda_margin_pct` and `acquired_co_revenue_growth_pct` fields are named to make clear they belong to the historically-acquired company, not the current target
- Market valuation comps (from `get_valuation_comps`)
- Pre-computed Python anomaly signals (see Data Flow step 7)

Generic output is explicitly forbidden by name. Conviction levels are required to vary. Risk flags must be tied to observable data, not generic statements like "market conditions."

**Anomaly signal system** — rather than relying solely on long forbidden lists (which gpt-4o-mini skims), the most critical constraints are pre-computed in Python and injected as `⚠`/`✓` attention markers directly into the evidence packet. Signals:
- Deal size routing (4-branch: GENUINE STRETCH / AT-SIZE RANGE COVERS TARGET / BELOW MEDIAN / AT-SIZE — controls which Section 5 category is valid)
- Completion rate (⚠ if outcome_score < 70; ✓ block category (e) if ≥ 70)
- Ownership mismatch (⚠ if ownership_score < 25 — acquirer rarely buys private companies)
- Valuation posture: ABOVE-MARKET uses `turns_diff` (additive EV/EBITDA turn difference) and `gap_pct` (% above market); BELOW-MARKET uses `stretch_pct` computed as `(market − acquirer) / acquirer` so the percentage correctly describes how much more than their own historical comfort they must bid; AT-MARKET blocks category (a) entirely
- Oversized precedent deals: Python scans fetched deals before LLM call; any deal >3× target EV is listed by name with ratio and the LLM is told disclosure of the size gap is mandatory when citing it
- Unconditional EBITDA block (per-call, always injected): forbids attributing margin quality to the current target in any section

**Post-generation EBITDA scan** — a separate Python regex check runs after the LLM returns. If "the target's EBITDA margins" (or variants) is detected in any text field, a targeted repair call is fired quoting the exact violation. This catches cases where three prompt-layer prohibitions still fail.

**Section 6 — Conviction** is a synthesis of Sections 1–5, not a precedent deal recap. Sentence 1 must draw on at least 2 signals simultaneously (sector concentration + cadence, valuation alignment + deal type, etc.). Sentence 2 names the specific binding constraint. "Closest precedent in this shortlist" is explicitly forbidden — it is circular. If citing a precedent deal 3× or more larger than the target, the size gap and what it does/does not prove must be stated explicitly.

**Why centralise prompts?** Prompt tuning should never require touching agent logic. A banker reviewing the output can propose changes to `prompts.py` without understanding LangGraph.

**Prompt length guidance:** Section length limits in the RATIONALE_PROMPT_TEMPLATE are intentional — "3-4 sentences" for prose sections keeps each acquirer rationale on one page. Do not remove these limits without also reducing PDF font sizes or margins. Conversely, do not add "2-3 sentence maximum" caps — the LLM will sacrifice factual density to hit the word count, which degrades output quality. "Be information-dense" is the right instruction; hard caps are not.

---

## Frontend Architecture Notes

**Vite proxy:** All `/api` requests from the browser are forwarded to `http://127.0.0.1:8000` by Vite's dev server. This means CORS is never an issue in dev — requests are same-origin from the browser's perspective. Do not add CORS origins for `localhost:5173`; requests never actually cross origins.

**State management (App.jsx):**
- `runId` / `streamUrl` / `loading` / `result` — active run state
- `historicalTarget` — when set, locks the form and shows "Viewing historical run" badge
- `refreshKey` — integer incremented on run start and completion; RunHistory's `useEffect` depends on it to re-fetch `/api/runs`. Replaces polling.

**Form lock pattern (TargetForm.jsx):** `editMode` local state (not a prop) controls whether fields are editable. When `historicalTarget` prop changes, the form populates and `editMode` becomes `false`. "+ New Analysis" sets `editMode = true` without resetting field values — the user edits from the historical values as a starting point. The submit button is always in the DOM (never conditionally rendered) with `display: none` when viewing history. This prevents a browser quirk where a button appearing in the same DOM position as a just-clicked button fires an immediate click event.

**RunHistory:** Clicking a running run reconnects to its SSE stream (sets `streamUrl`, `loading=true`) without fetching a result. Clicking a completed run fetches `/api/runs/{id}/result` and populates the form with the historical target via `historicalTarget`. Failed runs are non-interactive (opacity 0.5, default cursor).

**SSE (RunProgress.jsx):** Uses `EventSource` on the stream URL. `onmessage` fires only for events with no `event:` field (the default "message" type). Do not add an `event:` field to SSE payloads — named events bypass `onmessage` and would require `addEventListener`. `onerror` checks current status before overwriting — a normal server-close after `run.completed` fires `onerror`, and without this guard it would incorrectly set status to "error".

**Version footer (App.jsx):** A `<footer>` renders `v{__APP_VERSION__} · built {__BUILD_TIME__}` at the bottom of the UI. `__APP_VERSION__` and `__BUILD_TIME__` are compile-time constants injected by Vite's `define` config (`vite.config.js`) from `package.json` `version` and `new Date().toISOString()` at build time. To bump the version, run `npm version patch` in `frontend/` before committing. The footer lets you confirm the latest Railway deployment has taken effect without checking logs.

---

## Railway Deployment

The app is deployed as a **single container** on Railway: FastAPI serves both the `/api/*` routes and the compiled React `frontend/dist/` as static files. This eliminates CORS entirely — all fetch calls are same-origin.

**GitHub repo:** `https://github.com/cajmera99/wb_ma_agent` (branch: `main`)

**Live URL:** `https://wbmaagent-production.up.railway.app/`

### Deployment architecture

```
Browser → Railway proxy → FastAPI (port $PORT)
                              ├── /api/*       → agent routes
                              ├── /assets/*    → StaticFiles(frontend/dist/assets)
                              └── /{any}       → FileResponse(frontend/dist/index.html)
```

`backend/main.py` mounts static files and a catch-all SPA route at the end of startup, but only when `frontend/dist/` exists (i.e., inside the container — not in local dev where Vite's proxy is used instead).

### Dockerfile (multi-stage)

```dockerfile
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # produces frontend/dist/

FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY data/ ./data/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
RUN mkdir -p backend/output
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
```

The `CMD` uses `sh -c` so `${PORT:-8000}` is shell-expanded. Do **not** move the start command to `railway.json`'s `startCommand` field — Railway runs that without a shell, which passes `${PORT:-8000}` literally to uvicorn as an invalid port string.

### railway.json

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  }
}
```

Without this file Railway defaults to its Railpack auto-detector, which fails to find a start command. You must also set **Builder → Dockerfile** in Railway's service settings UI (the JSON alone is not always sufficient on first deploy).

### Environment variables (set in Railway dashboard)

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | `sk-...` |
| `OPENAI_MODEL` | `gpt-4o` |
| `ALLOWED_ORIGINS` | *(leave unset — same-origin, no CORS needed)* |
| `OUTPUT_DIR` | *(leave unset — defaults to `backend/output` inside container)* |

**Note:** `OUTPUT_DIR` and `ALLOWED_ORIGINS` are read in `backend/main.py` via `os.getenv()`. PDFs are stored inside the container at `backend/output/` — they are ephemeral (lost on redeploy). For persistence, mount a Railway Volume at `/app/backend/output`.

### SSE keepalive

Railway's proxy cuts idle connections after ~60 seconds. The SSE stream endpoint (`runs.py`) uses `asyncio.wait_for(queue.get(), timeout=15.0)` and sends `{"event_type": "keepalive"}` on timeout. The frontend's `onmessage` handler ignores keepalive events. Do not increase the timeout above 30s or Railway will drop the connection mid-run.

### Deploying a new version

```bash
git add -A
git commit -m "..."
git push origin main        # Railway auto-deploys on push to main
```

Railway builds the Dockerfile, runs `npm run build` inside the Node stage, then starts the Python container. Build typically takes 3–4 minutes. Check the Railway deployment logs for the uvicorn startup line to confirm the correct port is bound.

### Known Railway gotchas

- **Port**: Railway injects `$PORT` (usually 8000 or similar). The `CMD` shell-expands it. When generating a public domain in the Railway UI, specify port **8000** (Railway's default injection).
- **No persistent storage by default**: PDFs are lost on each redeploy. The run history (in-memory RunStore) is also wiped.
- **Branch**: the repo default branch is `main`. Code pushed to any other branch will not trigger a deploy unless Railway is configured for it.
- **CSV committed**: `data/ma_transactions_500.csv` is committed to the repo so Railway's build container has access to it. This is intentional — the dataset is not sensitive and must be present at startup.

---

## Known Limitations and Production Upgrade Path

### In-memory state
Current `RunStore` uses Python dicts. Server restart clears all history.

**Production:** Replace `_events` and `_runs` with Postgres tables. The `RunStore` interface (`add_event`, `get_events`, `list_runs`) stays identical. For horizontal scaling, replace `asyncio.Queue` with Redis pub/sub.

### LLM call count (~12 per run)
1 rerank (gpt-4o) + 10 rationales (gpt-4o-mini) + 0–2 repair attempts + 0–3 tool-call rounds in rerank.

Model split rationale: `llm_rerank` keeps gpt-4o because it calls tools and must judge which profiles to fetch before committing to a ranking. `generate_rationales` uses gpt-4o-mini (`llm_fast`) because it synthesises pre-assembled data — no tool judgment required — and mini's ~10× higher TPM limits eliminate the rate-limit stalls that occurred when 10 concurrent gpt-4o calls exhausted the per-minute budget.

**Optimisations (deferred):**
- Cache the `get_valuation_comps` result — it's identical for all 10 acquirers in a run (same target sector + size range)
- Batch rerank + top-3 rationales into one structured-output call

### Concurrency limit
No semaphore is currently set. Adding one:
```python
_llm_semaphore = asyncio.Semaphore(8)
async with _llm_semaphore:
    await compiled_graph.ainvoke(...)
```

### Adjacent sector data thinness
~46 Healthcare Services transactions, ~12 in the target size range. The scoring model compensates with adjacent-sector weighting. If the dataset grows, re-evaluate whether `ADJACENT_SECTORS` in `scorer.py` and `loader.py` should be narrowed.

---

## Environment

- Python: 3.11 (conda env `wb_ib_env`)
- LangGraph: 1.2.5
- LangChain: 1.3.9
- LangChain-OpenAI: 1.3.2
- OpenAI SDK: 2.41.1
- FastAPI: 0.136.3
- Pydantic: 2.13.4
- React: 18.3.1 + Vite 5.4.x

Package versions are specified with `>=` minimums in `requirements.txt` (not pinned) to allow pip to resolve the LangChain ecosystem's transitive dependency constraints without conflict. Pinning tenacity to `==9.0.0` caused a resolution failure because LangGraph 1.x requires a different tenacity range.
