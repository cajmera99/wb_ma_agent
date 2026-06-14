from typing import Any
from backend.models.events import RunEvent, EventType
from backend.services.run_store import RunStore
import structlog

logger = structlog.get_logger(__name__)


class EventEmitter:
    """
    Thin helper passed into the LangGraph agent.
    Each node calls emitter.emit(...) — it handles storage and SSE delivery.
    """

    def __init__(self, run_id: str, store: RunStore):
        self.run_id = run_id
        self._store = store

    def emit(
        self,
        event_type: EventType,
        node: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = RunEvent(
            run_id=self.run_id,
            event_type=event_type,
            node=node,
            data=data or {},
        )
        self._store.add_event(event)
        logger.debug(
            "event_emitted",
            run_id=self.run_id,
            event_type=event_type.value,
            node=node,
        )
