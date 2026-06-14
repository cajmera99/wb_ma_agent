"""
POST /api/analyze

Accepts a target company profile, launches an agent run in the background,
and immediately returns the run_id. The caller then subscribes to
GET /api/runs/{run_id}/stream to receive live SSE events.

When the run completes, the PDF is generated and the download URL is included
in the RUN_COMPLETED event payload.
"""

import uuid
import asyncio
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_app_state, get_run_store
from backend.services.app_state import AppState
from backend.services.run_store import RunStore
from backend.services.event_emitter import EventEmitter
from backend.services.pdf_generator import generate_pdf
from backend.models.target import TargetProfile
from backend.models.events import EventType
from backend.agent.graph import compiled_graph
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analyze"])  # full prefix


class AnalyzeRequest(BaseModel):
    sector: str = "Healthcare Services"
    deal_size_mm: float = 200.0
    geography: str = "Midwest"
    ownership: str = "Private"
    profile_description: str = "Mid-market, private, regional, strong EBITDA margins"


class AnalyzeResponse(BaseModel):
    run_id: str
    stream_url: str
    events_url: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    app_state: AppState = Depends(get_app_state),
    run_store: RunStore = Depends(get_run_store),
):
    """
    Launch an M&A acquirer identification run.

    Returns immediately with a run_id. The agent runs in the background.
    Connect to /api/runs/{run_id}/stream for live progress.
    Poll /api/runs/{run_id}/result once status is 'completed'.
    """
    run_id = str(uuid.uuid4())

    target = TargetProfile(
        sector=body.sector,
        deal_size_mm=body.deal_size_mm,
        geography=body.geography,
        ownership=body.ownership,
        profile_description=body.profile_description,
    )

    run_store.create_run(run_id, target_summary=target.model_dump())

    log = logger.bind(run_id=run_id)
    log.info("run_queued", sector=target.sector, deal_size=target.deal_size_mm)

    background_tasks.add_task(_run_agent, run_id, target, app_state, run_store)

    return AnalyzeResponse(
        run_id=run_id,
        stream_url=f"/api/runs/{run_id}/stream",
        events_url=f"/api/runs/{run_id}/events",
    )


async def _run_agent(
    run_id: str,
    target: TargetProfile,
    app_state: AppState,
    run_store: RunStore,
) -> None:
    """
    The actual agent execution — runs inside FastAPI's BackgroundTasks.
    Emits RUN_STARTED at the top, RUN_COMPLETED or RUN_FAILED at the end.
    """
    log = logger.bind(run_id=run_id)
    emitter = EventEmitter(run_id=run_id, store=run_store)

    emitter.emit(EventType.RUN_STARTED, data={
        "target": target.model_dump(),
    })

    initial_state = {
        "run_id": run_id,
        "target": target,
        "scored_candidates": [],
        "coverage_sufficient": False,
        "top_candidates": [],
        "final_acquirer_names": [],
        "rerank_reasoning": "",
        "rationales": [],
        "errors": [],
        "quality_gate_result": {},
        "regeneration_attempted": False,
    }

    # Pass emitter and app_state via config["configurable"] so every node
    # can access them without global state or circular imports.
    run_config = {
        "configurable": {
            "run_id": run_id,
            "emitter": emitter,
            "app_state": app_state,
        }
    }

    try:
        final_state = await compiled_graph.ainvoke(initial_state, config=run_config)
        log.info("agent_run_complete", acquirers=final_state.get("final_acquirer_names"))

        # Sort by composite score so conviction level always matches rank order,
        # then reassign rank numbers to match the sorted positions.
        rationales = sorted(
            final_state.get("rationales", []),
            key=lambda r: -r.get("composite_score", 0),
        )
        for i, r in enumerate(rationales):
            r["rank"] = i + 1

        loop = asyncio.get_running_loop()
        pdf_path = await loop.run_in_executor(
            None, generate_pdf, run_id, target, rationales
        )

        emitter.emit(EventType.RUN_COMPLETED, data={
            "rationales": rationales,
            "rerank_reasoning": final_state.get("rerank_reasoning", ""),
            "errors": final_state.get("errors", []),
            "pdf_path": pdf_path,
            "pdf_url": f"/api/runs/{run_id}/pdf",
        })

    except Exception as e:
        log.error("agent_run_failed", error=str(e))
        emitter.emit(EventType.RUN_FAILED, data={"error": str(e)})
