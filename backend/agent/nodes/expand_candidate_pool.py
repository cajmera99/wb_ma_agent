from langchain_core.runnables import RunnableConfig
from backend.agent.state import AgentState
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)

# When coverage is thin, expand the pool to this many candidates
EXPANDED_POOL_SIZE = 25


def node_expand_candidate_pool(state: AgentState, config: RunnableConfig) -> dict:
    """
    Called when evaluate_coverage finds too few strong candidates.

    Strategy: lower the effective threshold by taking a larger slice of
    scored_candidates (sorted best-first) and letting the LLM re-ranker
    decide which ones are actually credible. We widen the funnel;
    the LLM narrows it back to 10.

    This is honest — we're not pretending to find new data.
    We're just relaxing our pre-filter and trusting the LLM to be
    more selective at the next stage.
    """
    emitter = config["configurable"]["emitter"]

    emitter.emit(EventType.NODE_STARTED, node="expand_candidate_pool")

    scored = state["scored_candidates"]
    expanded = scored[:EXPANDED_POOL_SIZE]

    emitter.emit(EventType.NODE_COMPLETED, node="expand_candidate_pool", data={
        "original_pool_size": len(state.get("top_candidates", [])),
        "expanded_pool_size": len(expanded),
        "lowest_score_included": expanded[-1]["composite_score"] if expanded else None,
    })

    logger.info("pool_expanded", new_size=len(expanded))

    return {"top_candidates": expanded}
