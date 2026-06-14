import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI

from backend.core.config import settings
from backend.core.loader import load_transactions
from backend.core.profiler import build_acquirer_profiles
from backend.agent.tools import build_tools
from backend.observability.logging import configure_logging
from backend.services.app_state import AppState
import structlog

logger = structlog.get_logger(__name__)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "backend/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in the 'before yield' block runs once at startup.
    Everything after yield runs once on shutdown.
    """
    configure_logging()
    logger.info("server_starting")

    # Load and pre-process the CSV (one time only)
    df = load_transactions(settings.csv_path)

    # Build acquirer profiles from the loaded data (one time only)
    profiles = build_acquirer_profiles(df)

    # Create the LLM client once — reused across all requests
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    # Faster model for the rerank step — ranking 20 candidates requires
    # less reasoning depth than writing a full rationale, so mini saves
    # ~12s without meaningful quality loss on the selection decision.
    llm_fast = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    # Build tools once, closed over the loaded data
    tools = build_tools(df, profiles)

    # Attach everything to app.state — accessible on every request
    # via request.app.state, with no circular imports
    app.state.app_state = AppState(df=df, acquirer_profiles=profiles, llm=llm, llm_fast=llm_fast, tools=tools)

    # RunStore is imported here to avoid circular imports at module level
    from backend.services.run_store import RunStore
    app.state.run_store = RunStore()

    logger.info(
        "server_ready",
        acquirers_loaded=len(profiles),
        transactions_loaded=len(df),
    )

    yield  # server is running — handle requests

    logger.info("server_shutting_down")


app = FastAPI(
    title="M&A Acquirer Identification Engine",
    description="Identifies the 10 most likely acquirers for a target company using historical M&A data and LLM synthesis.",
    version="1.0.0",
    lifespan=lifespan,
)

_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
_allowed_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Each router already includes the full /api prefix in its own definition
from backend.api.routes import analyze, runs  # noqa: E402
app.include_router(analyze.router)
app.include_router(runs.router)


@app.get("/api/runs/{run_id}/pdf")
async def download_pdf(run_id: str, request: Request):
    """Download the generated PDF for a completed run."""
    pdf_path = OUTPUT_DIR / f"{run_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not yet generated or run not found")

    # Build a descriptive filename: Sector_Acquirer_Analysis_MMDDYYYY.pdf
    run = request.app.state.run_store.get_run_summary(run_id)
    sector = (run or {}).get("target", {}).get("sector", "Acquirer") if run else "Acquirer"
    started_at = (run or {}).get("started_at", "")
    try:
        from datetime import datetime
        date_str = datetime.fromisoformat(started_at).strftime("%m%d%Y")
    except Exception:
        date_str = datetime.utcnow().strftime("%m%d%Y")

    safe_sector = "_".join(sector.split())  # "Healthcare Services" → "Healthcare_Services"
    filename = f"{safe_sector}_Acquirer_Analysis_{date_str}.pdf"

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
    )


@app.get("/api/graph", response_class=HTMLResponse)
async def view_graph():
    """Render the LangGraph agent workflow as an interactive diagram."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>M&A Agent — Workflow Graph</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f4f8;
      color: #1a202c;
      min-height: 100vh;
    }
    header {
      background: #003087;
      color: #fff;
      padding: 16px 36px;
      display: flex;
      align-items: baseline;
      gap: 16px;
    }
    header h1 { font-size: 18px; font-weight: 700; letter-spacing: 0.3px; }
    header span { font-size: 12px; color: #c8d8ee; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 24px;
      padding: 28px 36px;
      max-width: 1400px;
      margin: 0 auto;
    }
    .card {
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.09);
      padding: 28px 32px;
    }
    .card h2 {
      font-size: 13px;
      font-weight: 700;
      color: #003087;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 20px;
      padding-bottom: 8px;
      border-bottom: 1.5px solid #e2e8f0;
    }
    .mermaid { display: flex; justify-content: center; }
    .node-list { display: flex; flex-direction: column; gap: 14px; }
    .node {
      border-left: 3px solid #003087;
      padding: 10px 14px;
      background: #f7f9fc;
      border-radius: 0 6px 6px 0;
    }
    .node.router { border-left-color: #7c3aed; background: #faf5ff; }
    .node.tool    { border-left-color: #d97706; background: #fffbeb; }
    .node.llm     { border-left-color: #0ea5e9; background: #f0f9ff; }
    .node-name {
      font-size: 12px;
      font-weight: 700;
      color: #1a202c;
      margin-bottom: 4px;
    }
    .node-tag {
      display: inline-block;
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 1px 6px;
      border-radius: 3px;
      margin-bottom: 5px;
    }
    .tag-pure   { background: #dcfce7; color: #166534; }
    .tag-router { background: #ede9fe; color: #5b21b6; }
    .tag-llm    { background: #e0f2fe; color: #075985; }
    .tag-tool   { background: #fef3c7; color: #92400e; }
    .node-desc { font-size: 11.5px; color: #4a5568; line-height: 1.55; }
    .tools-used {
      margin-top: 6px;
      font-size: 10.5px;
      color: #718096;
    }
    .tools-used span {
      background: #edf2f7;
      border-radius: 3px;
      padding: 1px 5px;
      margin-right: 4px;
      font-family: monospace;
    }
    .legend {
      margin-top: 20px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #4a5568; }
    .legend-dot {
      width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
    }
  </style>
</head>
<body>
<header>
  <h1>M&A Acquirer Identification — Agent Workflow</h1>
  <span>LangGraph StateGraph &nbsp;·&nbsp; GPT-4o &nbsp;·&nbsp; William Blair AI</span>
</header>

<div class="layout">
  <!-- Left: diagram -->
  <div class="card">
    <h2>Orchestration Graph</h2>
    <div class="mermaid">
%%{init: {"theme": "base", "themeVariables": {
  "primaryColor": "#003087",
  "primaryTextColor": "#ffffff",
  "primaryBorderColor": "#001a4d",
  "lineColor": "#64748b",
  "edgeLabelBackground": "#374151",
  "secondaryColor": "#EAF0F8",
  "tertiaryColor": "#f0f4f8",
  "fontSize": "14px"
}}}%%
flowchart TD
    START(["▶  START"]):::startend
    END_(["⏹  END"]):::startend

    A["⚙  score_and_rank\n─────────────────\nScores all 107 acquirers\nacross 6 dimensions\n(pure Python, no LLM)"]:::pure

    B["⚖  evaluate_coverage\n─────────────────\nCounts acquirers above\nscore threshold of 30"]:::router

    C["🔍  expand_candidate_pool\n─────────────────\nWidens pool to top-25\nwhen coverage is thin"]:::pure

    D["🤖  llm_rerank\n─────────────────\nLLM selects final 10\nwith tool-calling loop\n(up to 3 rounds)"]:::llm

    E["✍  generate_rationales\n─────────────────\n10 LLM calls, 3 at a time\n(Semaphore throttle)\nStructured output +\nrepair loop"]:::llm

    START --> A
    A --> B
    B -->|"sufficient coverage"| D
    B -->|"thin coverage"| C
    C --> D
    D --> E
    E --> END_

    classDef startend fill:#1a202c,stroke:#1a202c,color:#fff,rx:20
    classDef pure    fill:#003087,stroke:#001a4d,color:#fff
    classDef router  fill:#7c3aed,stroke:#5b21b6,color:#fff
    classDef llm     fill:#0ea5e9,stroke:#0284c7,color:#fff
    </div>

    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#003087"></div>Pure Python node</div>
      <div class="legend-item"><div class="legend-dot" style="background:#7c3aed"></div>Routing decision</div>
      <div class="legend-item"><div class="legend-dot" style="background:#0ea5e9"></div>LLM node</div>
    </div>
  </div>

  <!-- Right: node reference -->
  <div class="card">
    <h2>Node Reference</h2>
    <div class="node-list">

      <div class="node">
        <div class="node-tag tag-pure">Pure Python</div>
        <div class="node-name">score_and_rank</div>
        <div class="node-desc">
          Scores every acquirer in the dataset across 6 weighted dimensions
          (sector affinity 35%, deal size 20%, rationale tags 20%, recency 10%,
          outcome quality 10%, ownership match 5%). Returns a sorted list in ~5ms.
          No LLM involved. Sector affinity scoring is dynamic — anchored to
          <code>target.sector</code> at runtime, with healthcare-family adjacency
          weights applied only when the target is within that sector family.
          Acquirer profiles (including sub-sector counts, platform cadence, and
          bolt-on history) are pre-computed at startup and reused here.
        </div>
      </div>

      <div class="node router">
        <div class="node-tag tag-router">Conditional Router</div>
        <div class="node-name">evaluate_coverage</div>
        <div class="node-desc">
          Deterministic Python function — counts acquirers scoring above 30.
          Routes to <strong>expand_candidate_pool</strong> if fewer than 15 qualify
          (thin sector data); otherwise proceeds directly to <strong>llm_rerank</strong>.
        </div>
      </div>

      <div class="node">
        <div class="node-tag tag-pure">Pure Python · Conditional</div>
        <div class="node-name">expand_candidate_pool</div>
        <div class="node-desc">
          Widens the candidate window from top-20 to top-25 so the LLM has
          more options when primary-sector data is sparse (e.g. Healthcare Services
          has ~46 transactions vs 500 total).
        </div>
      </div>

      <div class="node llm">
        <div class="node-tag tag-llm">LLM · Agentic Tool Use</div>
        <div class="node-name">llm_rerank</div>
        <div class="node-desc">
          GPT-4o receives quantitative scores and selects the final 10 acquirers,
          applying qualitative judgment the scoring model cannot capture (bolt-on
          vs platform logic, geographic appetite, buyer diversity). Tool-calling
          loop runs up to 3 rounds.
        </div>
        <div class="tools-used">
          Tools: <span>search_transactions</span><span>get_acquirer_profile</span>
        </div>
      </div>

      <div class="node llm">
        <div class="node-tag tag-llm">LLM · Structured Output</div>
        <div class="node-name">generate_rationales</div>
        <div class="node-desc">
          Fires 10 <strong>gpt-4o-mini</strong> calls via <code>asyncio.gather</code>,
          throttled to 5 at a time by <code>asyncio.Semaphore(5)</code>. Mini's
          ~10× higher TPM limits over gpt-4o eliminate rate-limit stalls; pure
          synthesis from pre-assembled data requires no tool-use judgment.<br/><br/>
          Before each call, Python pre-computes <strong>6 anomaly signals</strong>
          injected as ⚠/✓ attention markers: deal size routing (4 branches),
          completion rate gate, ownership mismatch, valuation posture with
          pre-computed premium, EBITDA margin attribution block, and co-acquirer
          exit buyer list for PE sponsors. Conviction level is enforced in Python
          post-generation — the LLM writes text calibrated to the level but never
          controls the label.<br/><br/>
          Evidence packet: full acquirer profile, sub-sector counts, platform
          cadence, two-pass precedent deal fetch (up to 5, sector-first), and
          market valuation comps. Returns a validated
          <code>AcquirerRationale</code> Pydantic object with one automated repair
          attempt on validation failure.
        </div>
        <div class="tools-used">
          Tools: <span>get_acquirer_precedent_deals</span><span>get_valuation_comps</span>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
  mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/health")
async def health(request: Request):
    state = request.app.state.app_state
    return {
        "status": "ok",
        "acquirers_loaded": len(state.acquirer_profiles),
        "transactions_loaded": len(state.df),
    }


# ── Static frontend (production) ──────────────────────────────────────────────
# In dev, Vite's server handles the frontend; this block is skipped entirely.
# In prod (Docker), the React build is copied to frontend/dist/ at image build
# time and FastAPI serves it here so the whole app runs on one port / origin.
FRONTEND_DIST = Path("frontend/dist")
if FRONTEND_DIST.exists():
    # Vite outputs hashed JS/CSS bundles to dist/assets/ — serve them directly
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="static-assets",
    )

    @app.get("/{catch_all:path}", include_in_schema=False)
    async def serve_spa(catch_all: str):
        # FastAPI's own /docs, /redoc, /openapi.json, and /health routes are
        # registered before this catch-all and will match first, but guard
        # explicitly in case of unusual routing edge cases.
        _reserved = {"docs", "redoc", "openapi.json", "health"}
        if catch_all in _reserved or catch_all.startswith("api/"):
            raise HTTPException(status_code=404)
        return FileResponse(str(FRONTEND_DIST / "index.html"))
