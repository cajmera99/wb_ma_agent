# M&A Acquirer Identification Engine

A production-grade agentic system that identifies the 10 most likely acquirers for a target company using 500 historical M&A transactions and LLM synthesis. Output is a 10-page PDF — one page per acquirer — with data-backed rationale, valuation context, risk flags, and conviction levels.

Built for the William Blair AI Innovation Team take-home assessment.

---

## Live Demo

**Deployed on Railway: [https://wbmaagent-production.up.railway.app/](https://wbmaagent-production.up.railway.app/)**

The full stack (FastAPI + React) runs as a single container. Open the URL, fill in a target company profile, and click **Run Analysis**. No setup required.

API docs (Swagger): **[https://wbmaagent-production.up.railway.app/docs](https://wbmaagent-production.up.railway.app/docs)**

**If for some reason the website and/or the ability to download a file is being blocked by the intranet firewall, I have attached a sample pdf file in the root of the repo.**

---

## Quick Start (Local)

### Option 1 — Docker (single command)

```bash
docker build -t wb-ma-engine .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-4o \
  wb-ma-engine
```

Open `http://localhost:8000` — FastAPI serves both the API and the compiled React frontend.

### Option 2 — Manual (dev mode, hot reload)

**Prerequisites:** Python 3.11+, Node 18+, OpenAI API key, `data/ma_transactions_500.csv`

```bash
# Recommended: conda env (tested)
conda create -n wb_ib_env python=3.11 -y
conda activate wb_ib_env
pip install -r requirements.txt

# Or: standard venv
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

```bash
# Create .env in project root
echo OPENAI_API_KEY=sk-... > .env
echo OPENAI_MODEL=gpt-4o >> .env
```

```bash
# Terminal 1 — Backend
# --reload-dir backend prevents frontend file saves from restarting the server
# and wiping the in-memory run history
uvicorn backend.main:app --reload --reload-dir backend --port 8000

# Terminal 2 — Frontend (dev server with hot reload)
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

> **Windows note:** Vite's dev proxy targets `http://127.0.0.1:8000` (not `localhost`). On Windows,
> `localhost` can resolve to `::1` (IPv6) while uvicorn binds to `127.0.0.1` (IPv4), causing
> silent proxy failures. Do not change the proxy target.

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","acquirers_loaded":107,"transactions_loaded":500}
```

---

## Architecture Decisions

### Agent Graph (LangGraph StateGraph)

```
score_and_rank → evaluate_coverage ──┬──(sufficient)──→ llm_rerank → generate_rationales
                                     └──(thin data)──→ expand_candidate_pool ──→ llm_rerank
                                                                                        ↓
                                                                               quality_gate
                                                                              ↙            ↘
                                                                          END     targeted_regeneration → END
```

The graph has two conditional edges with different routing philosophies:

- **`route_after_coverage`** — deterministic Python threshold. If fewer than 15 acquirers score above 30/100, the pool widens from top-20 to top-25 before reranking.
- **`route_after_quality_gate`** — LLM-driven. After all 10 rationales are generated, compact summaries are sent to GPT-4o-mini in a single call. The LLM identifies 0–3 weak rationales (low citation density, template recycling across acquirers, thin conviction, bare risk flag labels) and routes to `targeted_regeneration` if any are found. This cross-acquirer qualitative comparison cannot be reduced to a Python threshold — it is the primary genuinely agentic routing decision in the graph.

```
Backend startup (once)
├── Load CSV → pandas DataFrame (feature engineering pre-computed)
├── Build acquirer profiles → dict[str, dict] (groupby aggregations for 107 acquirers)
├── Instantiate ChatOpenAI (GPT-4o + GPT-4o-mini)
└── Build LangChain tools (factory functions closed over the static data)

Per-request agent (BackgroundTask)
├── Node 1: score_and_rank          — pure Python, scores all 107 acquirers in ~5ms
├── Node 2: evaluate_coverage       — deterministic routing
├── Node 3: expand_candidate_pool   (conditional) — widens pool if sector data is thin
├── Node 4: llm_rerank              — GPT-4o with tool-calling selects final 10
├── Node 5: generate_rationales     — 10 concurrent GPT-4o-mini calls → Pydantic validated
│                                     (valuation comps pre-fetched once, shared across all 10)
├── Node 6: quality_gate            — LLM-driven cross-acquirer quality check → routing
└── Node 7: targeted_regeneration   (conditional) — re-runs 1–3 weak rationales
```

### How the CSV Is Used

The 500-row CSV is loaded **once at server startup** — never re-read during a request.

**Startup feature engineering:**
- Parse `rationale_tags` from semicolon-delimited strings into Python lists
- Flag each row as `is_adjacent_sector` (Healthcare Services, Behavioral Health, Physician Groups, Home Health/Hospice)
- Build per-acquirer profiles via `groupby`: sector counts, sub-sector counts, deal sizes (full sorted list, min, max, median), EV multiples, rationale tag frequencies, deal type breakdown, geography mix, ownership type counts, platform cadence, recency signals

**Per-request scoring:** The pre-built profiles are scored against the target's sector, deal size, geography, and ownership without touching the CSV.

**Per-rationale evidence assembly:** Two tools (`get_acquirer_precedent_deals`, `get_valuation_comps`) filter the in-memory DataFrame to build a 5-deal precedent table and market comp statistics. This is done directly in Python — the LLM does not decide what to fetch at this stage.

### LLM Prompt Structure

Four prompt templates live in `backend/agent/prompts.py`:

**1. `SYSTEM_PROMPT`** — Applied to every LLM call. Establishes the senior M&A analyst persona and sets non-negotiable rules: all claims must cite specific numbers, forbidden phrases are listed explicitly (e.g. "demonstrates their capability", "track record of acquisitions", attributing EBITDA margins to the current target), and Strategic vs. Financial Sponsor theses must be framed differently.

**2. `RERANK_PROMPT_TEMPLATE`** — Used by GPT-4o in `llm_rerank`. Provides the quantitative scores for top-20/25 candidates and asks for qualitative override: thesis diversity over score clustering, bolt-on vs. platform logic for PE sponsors, geographic appetite. The LLM may call `get_acquirer_profile` or `search_transactions` tools before committing to a ranking.

**3. `RATIONALE_PROMPT_TEMPLATE`** — The most critical prompt. Used by GPT-4o-mini in `generate_rationales`. Each acquirer receives a fully assembled evidence packet:
- Acquirer profile with **pre-computed citation anchors**: exact deal count in the target sector, count of deals in the comparable size band (0.5×–2.0× target EV), full deal size range, complete sector breakdown — enabling the sentence pattern "completed 6 Healthcare Services deals in the $100–$400M range at a median 12.5× EV/EBITDA"
- 5 precedent deals (sector-relevant first via two-pass fetch)
- Market EV/EBITDA and EV/Revenue comps
- **Python-computed anomaly signals** injected as `⚠`/`✓` markers: deal size routing, completion rate gate, ownership mismatch, valuation posture (above/below/at-market with pre-computed turn differences and correct percentage denominators), oversized precedent deal disclosures, and an acquirer-type EBITDA differentiation signal (PE sponsors receive IRR/return-on-capital framing using the acquirer's historical acquired-company margins; strategics receive a margin comparison frame)

The anomaly signal design is intentional: long instruction-only forbidden lists get skimmed by smaller models. Injecting pre-computed values as attention markers (`⚠ ABOVE-MARKET PAYER: ... use EXACTLY: 'Above-Market Payer — 16.2x historical median vs 11.7x market median (+4.5 turns, +38% above market)'`) eliminates the class of errors where the LLM invents its own numbers.

A **post-generation regex scan** checks all text fields for generic EBITDA boilerplate (e.g. "the target's EBITDA margins complement / align with / support…"). If detected, a targeted repair call is fired redirecting toward acquirer-specific framing — more reliable than preemptive instructions alone.

**4. `QUALITY_GATE_PROMPT_TEMPLATE`** — Used by GPT-4o-mini in `quality_gate`. Receives compact summaries of all 10 rationales and identifies 0–3 weak ones across four criteria: citation density, template recycling across acquirers, thin conviction, and bare risk flag labels with no embedded numbers. Returns a JSON routing decision. Flagging at most 3 is enforced in Python regardless of LLM output.

### Scoring Model (6 Dimensions)

| Dimension | Weight | Method |
|-----------|--------|--------|
| Sector affinity | 35% | Primary sector (1.0) / Adjacent healthcare (0.7) / Secondary (0.3) |
| Deal size match | 20% | Gaussian decay around target EV, σ = 60% of target |
| Rationale tag alignment | 20% | Fraction of HIGH_RELEVANCE_TAGS matched (Platform Build, Geographic Expansion, etc.) |
| Recency | 10% | Stale penalty (−0.15/yr since last deal) + recent deal count bonus |
| Outcome quality | 10% | Closed / total deals ratio |
| Ownership match | 5% | Private + PE-backed / total deals |

Composite score = weighted sum × 100 (0–100). **Conviction level is Python-enforced:** composite > 80 → High, 50–79 → Medium, < 50 → Low. The LLM writes text calibrated to the level but never controls the label.

**Rationale ordering** is deterministic: the final 10 rationales are sorted by composite score descending before PDF generation. The LLM rerank selects *which* 10 make the shortlist, not the order within it.

### LangChain Tools

| Tool | Used in | Caller |
|------|---------|--------|
| `search_transactions` | `llm_rerank` | LLM (agentic) |
| `get_acquirer_profile` | `llm_rerank` | LLM (agentic) |
| `get_acquirer_precedent_deals` | `generate_rationales` | Python (per-acquirer) |
| `get_valuation_comps` | `generate_rationales`, `targeted_regeneration` | Python (once per node, cached) |

In `llm_rerank`, the LLM decides whether to call tools and with what arguments — real agentic tool use. In `generate_rationales`, tools are called directly in Python to build the evidence packet before the LLM is invoked. `get_valuation_comps` is called once per node invocation and its result is shared across all 10 (or 1–3) `_generate_one` calls — not once per acquirer.

### Structured Output + Repair Loop

Rationale generation uses `llm.with_structured_output(AcquirerRationale)` — OpenAI function-calling constrained to a Pydantic v2 schema with enforced constraints (`min_length=1` on precedent deals, `min_length=2` on risk flags, `Literal` types on conviction and acquirer type).

Two repair triggers:
1. **Schema failure**: Pydantic raises → send exact error back as repair message → retry once → stub page on second failure
2. **Content violation**: post-generation regex detects forbidden phrase → send targeted correction with quoted violation → retry once → keep original on failure (better than a stub)

### Observability

Every node transition emits a typed `RunEvent` (13 event types: `run.started`, `node.started`, `node.completed`, `node.error`, `routing.decision`, `tool.called`, `tool.result`, `rationale.generated`, `validation.failed`, `validation.repaired`, `llm.tokens_used`, `run.completed`, `run.failed`).

Events are appended to an in-memory log and pushed to a per-run `asyncio.Queue`. The SSE endpoint (`GET /api/runs/{id}/stream`) replays all past events for late-connecting clients, then tails the queue until a terminal event closes the stream.

---

## Assumptions

**About the target company:**
- The `deal_size_mm` field represents enterprise value (not equity value) — consistent with the CSV's `deal_size_mm` column
- "Strong EBITDA margins" in the profile description is taken as a qualitative descriptor only; no specific margin percentage is assumed or invented in any rationale
- The target is assumed to be in an active or imminent sale process (not hypothetical exploration)
- "Midwest" geography is used to assess geographic fit but is not used to hard-filter acquirers — cross-regional acquirers are included where other signals are strong

**About the dataset:**
- Deal sizes in the CSV are enterprise values paid at close (not LOI values or equity checks)
- Outcomes of "Withdrawn" or "Terminated" represent pre-close failures (regulatory, price disagreement, due diligence) — not post-close integration failures. They reduce completion score but are not treated as the same as never-announced deals
- PE-Backed and Private targets are treated as equivalent ("private-side") for ownership scoring — both require private-company diligence processes
- Adjacent sectors (Behavioral Health, Physician Groups, Home Health/Hospice) are assumed to indicate meaningful transferable M&A capability for a Healthcare Services target
- The 500 transactions are assumed to represent the universe of relevant acquirers; acquirers not in the dataset are not considered

---

## Known Limitations & Failure Modes

**Data coverage:**
- Healthcare Services has ~46 transactions in the dataset, ~12 in the $100–400M size band. The model compensates via adjacent-sector weighting (0.7×) and the `expand_candidate_pool` node, but shortlists may include cross-sector buyers making transferable rather than sector-native arguments
- Acquirers not represented in the 500-row dataset are not surfaced — a known Fortune 500 strategic buyer with no deals in this dataset would not appear

**LLM failure modes:**
- If the OpenAI API is unavailable or quota-exhausted, individual rationale calls fail with a stub page; if all 10 fail, the PDF renders with 10 stub pages
- The rerank LLM sometimes calls `get_acquirer_profile` for every candidate in the pool (20+), extending run time to 3–5 minutes. This is analytically thorough but unpredictable in latency
- Repair loops add 1–2 extra LLM calls per affected acquirer; under high concurrency this can trigger OpenAI rate-limit backoff

**Infrastructure:**
- Run history is in-memory and is lost on server restart. The frontend auto-restores the most recent completed run on page load, but older runs are gone
- PDFs are stored on the container filesystem; they are lost on Railway redeploy unless a persistent volume is mounted
- No request queuing: concurrent runs share the same OpenAI rate limits. Practical limit is 2–3 simultaneous sessions on Tier 1

**Edge cases:**
- Targets in sectors with fewer than 10 matching acquirers in the dataset may produce a shortlist that is partially or entirely cross-sector buyers
- If a target EV is at the extreme high end (>$2B), deal size scoring may plateau and fail to differentiate acquirers who are all "below median"

---

## Output Non-Determinism

**What is deterministic (same every run for the same input):**
- The quantitative scoring model — all 107 acquirers receive identical scores for identical inputs
- The routing decision (thin vs. sufficient coverage) — pure Python comparison
- Conviction levels — Python-enforced from composite score thresholds (>80 High, 50–79 Medium, <50 Low)
- Rationale ordering — sorted by composite score descending, rank numbers reassigned accordingly
- Which precedent deals are fetched — deterministic tool calls with fixed sort order

**What varies between runs:**
- Which 10 acquirers the LLM selects in `llm_rerank` — acquirers near the 10th/11th score boundary may swap. The top 6–7 are stable; edge positions vary
- The rationale text for each section — different word choices, emphasis, and sentence structure
- Which risk flags are chosen per acquirer (the categories are guided, not mandated)
- The rerank LLM's tool-calling pattern — it may fetch 5 profiles one run and 18 the next

**How we handle this:**
- We do **not** set `temperature=0` or a fixed seed. The LLM's judgment in `llm_rerank` benefits from variation — fully deterministic output would reduce qualitative differentiation between acquirers
- The anomaly signal system pre-computes all quantitative values (specific multiples, percentages, turn counts) and injects them as mandatory numbers in the prompt, preventing the LLM from inventing different numbers on different runs even as prose varies
- Conviction labels are Python-enforced post-generation, so conviction levels do not vary between runs for the same target

---

## What I Would Improve Given More Time

1. **SQLite persistence** — Replace the in-memory `RunStore` with a local SQLite database (no new dependencies). Schema: `runs` and `events` tables. The `RunStore` interface stays identical so no agent code changes. Runs survive server restarts indefinitely

2. **Richer target profile** — Accept optional known financials: revenue range, EBITDA %, recent headcount, key service lines. More target data enables more precise fit arguments (e.g. "their portfolio company margin profile suggests they'd pay a premium for this margin level")

3. **Reviewer feedback loop** — Add a `PATCH /api/runs/{id}/rationale/{n}` endpoint so an MD can flag a weak rationale and trigger a single-page regeneration with their feedback as an additional system message

4. **Fine-grained sector taxonomy** — "Healthcare Services" covers outpatient, home care, behavioral, physician management, and more. Mapping CSV sub-sectors to a richer ontology would improve sector affinity scoring for targets in narrow sub-sectors

5. **Request queuing** — `asyncio.Semaphore` on graph invocations to prevent concurrent sessions from fighting over OpenAI rate limits

---

## Concurrency Model

- Each `POST /api/analyze` runs in a FastAPI `BackgroundTask` (asyncio, not a thread)
- All 10 rationale LLM calls fire simultaneously via `asyncio.gather`, throttled to 5 at a time by `asyncio.Semaphore(5)`. Total latency ≈ 1 LLM call, not 10×
- Per-run `asyncio.Queue` ensures SSE event streams from concurrent runs don't bleed
- Model split: GPT-4o for `llm_rerank` (tool-calling judgment); GPT-4o-mini for `generate_rationales` (synthesis from pre-assembled data, ~10× higher TPM limits eliminates rate-limit stalls)

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Start a run |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}` | Run metadata |
| GET | `/api/runs/{id}/events` | Full event log (JSON) |
| GET | `/api/runs/{id}/stream` | Live SSE stream |
| GET | `/api/runs/{id}/result` | Final rationales JSON |
| GET | `/api/runs/{id}/pdf` | Download PDF report |
| GET | `/health` | Server health + loaded counts |
| GET | `/api/graph` | Agent graph visualization |

## Sample Request

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sector": "Healthcare Services",
    "deal_size_mm": 200,
    "geography": "Midwest",
    "ownership": "Private",
    "profile_description": "Mid-market, private, regional, strong EBITDA margins"
  }'
# {"run_id":"abc123...","stream_url":"/api/runs/abc123.../stream","events_url":"..."}
```

All fields have defaults — sending `{}` runs the standard Healthcare Services test case.
