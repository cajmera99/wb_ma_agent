"""
Read-only routes for run history and SSE event streaming.

GET /api/runs                   — list all runs (newest first)
GET /api/runs/{run_id}          — summary metadata for one run
GET /api/runs/{run_id}/events   — full event log for a run (past events)
GET /api/runs/{run_id}/stream   — live SSE stream for an active run
GET /api/runs/{run_id}/result   — final rationales from a completed run
"""

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_run_store
from backend.services.run_store import RunStore

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
def list_runs(store: RunStore = Depends(get_run_store)):
    return store.list_runs()


@router.get("/{run_id}")
def get_run(run_id: str, store: RunStore = Depends(get_run_store)):
    run = store.get_run_summary(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/events")
def get_events(run_id: str, store: RunStore = Depends(get_run_store)):
    """Return the complete event log for a run. Works for both active and completed runs."""
    run = store.get_run_summary(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    events = store.get_events(run_id)
    return [e.model_dump(mode="json") for e in events]


@router.get("/{run_id}/stream")
async def stream_run(run_id: str, store: RunStore = Depends(get_run_store)):
    """
    Server-Sent Events stream for a live run.

    - If the run is already completed, replays all stored events then closes.
    - If the run is active, streams events as they arrive then closes when
      a RUN_COMPLETED or RUN_FAILED event is received.

    The frontend subscribes to this endpoint immediately after POST /api/analyze
    returns the run_id. Events arrive in real time as each agent node executes.
    """
    run = store.get_run_summary(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        # First: replay any events that have already been stored
        # (handles the case where the client connects slightly after the run starts)
        past_events = store.get_events(run_id)
        for event in past_events:
            # No named "event:" field — browser EventSource.onmessage only fires
            # for the default "message" type. The event_type is in the JSON payload.
            yield {"data": json.dumps(event.model_dump(mode="json"))}

        # If the run is already done, no need to tail the queue
        run_summary = store.get_run_summary(run_id)
        if run_summary and run_summary["status"] in ("completed", "failed"):
            return

        # Tail the live queue until a terminal event arrives
        queue = store.get_queue(run_id)
        if queue is None:
            return

        # Track events already replayed so we don't double-send
        already_sent = {e.event_type.value + e.timestamp.isoformat() for e in past_events}

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"event_type": "keepalive"})}
                continue

            dedup_key = event.event_type.value + event.timestamp.isoformat()
            if dedup_key not in already_sent:
                yield {"data": json.dumps(event.model_dump(mode="json"))}

            if event.event_type.value in ("run.completed", "run.failed"):
                break

    return EventSourceResponse(event_generator())


@router.get("/{run_id}/result")
def get_result(run_id: str, store: RunStore = Depends(get_run_store)):
    """
    Return the final rationales once a run is completed.
    The rationales are stored in the RUN_COMPLETED event's data payload.
    """
    run = store.get_run_summary(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] == "running":
        raise HTTPException(status_code=202, detail="Run still in progress")
    if run["status"] == "failed":
        raise HTTPException(status_code=500, detail="Run failed")

    # The rationales are stored in the RUN_COMPLETED event's data dict
    events = store.get_events(run_id)
    for event in reversed(events):
        if event.event_type.value == "run.completed":
            return event.data

    raise HTTPException(status_code=404, detail="Result not found")
