import asyncio
from datetime import datetime, timezone
from backend.models.events import RunEvent, EventType
import structlog

logger = structlog.get_logger(__name__)


class RunStore:
    """
    In-memory store for all agent runs and their events.

    Production upgrade path: replace the dicts with Postgres tables.
    The interface (add_event, get_events, get_run_summary) stays the same.
    """

    def __init__(self):
        # run_id -> list of RunEvent
        self._events: dict[str, list[RunEvent]] = {}

        # run_id -> metadata (status, start time, target profile)
        self._runs: dict[str, dict] = {}

        # run_id -> asyncio.Queue for active SSE connections
        self._queues: dict[str, asyncio.Queue] = {}

    def create_run(self, run_id: str, target_summary: dict) -> None:
        self._events[run_id] = []
        self._runs[run_id] = {
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "target": target_summary,
        }
        self._queues[run_id] = asyncio.Queue()
        logger.info("run_created", run_id=run_id)

    def add_event(self, event: RunEvent) -> None:
        """Store event and push it to the SSE queue for this run."""
        if event.run_id not in self._events:
            return

        self._events[event.run_id].append(event)

        queue = self._queues.get(event.run_id)
        if queue:
            queue.put_nowait(event)

        # Update run status on terminal events
        if event.event_type == EventType.RUN_COMPLETED:
            self._runs[event.run_id]["status"] = "completed"
            self._runs[event.run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        elif event.event_type == EventType.RUN_FAILED:
            self._runs[event.run_id]["status"] = "failed"
            self._runs[event.run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    def get_events(self, run_id: str) -> list[RunEvent]:
        return self._events.get(run_id, [])

    def get_run_summary(self, run_id: str) -> dict | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict]:
        return list(reversed(list(self._runs.values())))

    def get_queue(self, run_id: str) -> asyncio.Queue | None:
        return self._queues.get(run_id)
