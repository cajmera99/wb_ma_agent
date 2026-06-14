from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RUN_STARTED         = "run.started"
    NODE_STARTED        = "node.started"
    NODE_COMPLETED      = "node.completed"
    NODE_ERROR          = "node.error"
    ROUTING_DECISION    = "routing.decision"
    TOOL_CALLED         = "tool.called"
    TOOL_RESULT         = "tool.result"
    RATIONALE_GENERATED = "rationale.generated"
    VALIDATION_FAILED   = "validation.failed"
    VALIDATION_REPAIRED = "validation.repaired"
    LLM_TOKENS_USED     = "llm.tokens_used"
    RUN_COMPLETED       = "run.completed"
    RUN_FAILED          = "run.failed"


class RunEvent(BaseModel):
    run_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
