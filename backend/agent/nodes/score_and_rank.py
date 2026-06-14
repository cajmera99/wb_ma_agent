from langchain_core.runnables import RunnableConfig
from backend.agent.state import AgentState
from backend.core.scorer import rank_acquirers
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)


def node_score_and_rank(state: AgentState, config: RunnableConfig) -> dict:
    """
    Score every acquirer in the dataset against the target profile.
    Pure Python — no LLM call. Result is a sorted list, best score first.
    """
    emitter = config["configurable"]["emitter"]
    app_state = config["configurable"]["app_state"]

    emitter.emit(EventType.NODE_STARTED, node="score_and_rank")

    target = state["target"]
    scored = rank_acquirers(app_state.acquirer_profiles, target)

    top = scored[0] if scored else {}
    emitter.emit(EventType.NODE_COMPLETED, node="score_and_rank", data={
        "total_acquirers_scored": len(scored),
        "top_acquirer": top.get("acquirer_name"),
        "top_score": top.get("composite_score"),
    })

    logger.info("scoring_complete", total=len(scored),
                top=top.get("acquirer_name"), score=top.get("composite_score"))

    return {"scored_candidates": scored}
