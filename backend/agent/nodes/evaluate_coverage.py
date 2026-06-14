from langchain_core.runnables import RunnableConfig
from backend.agent.state import AgentState
from backend.core.config import settings
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)

# Minimum number of candidates that must score above the threshold
# before we consider coverage sufficient
COVERAGE_THRESHOLD_SCORE = 30.0
COVERAGE_MIN_CANDIDATES = 15


def node_evaluate_coverage(state: AgentState, config: RunnableConfig) -> dict:
    """
    Checks whether the scoring model produced enough viable candidates.

    If yes  → graph routes to llm_rerank with the top-N pool.
    If no   → graph routes to expand_candidate_pool to lower the threshold
              and widen the pool before re-ranking.

    This is a deterministic routing decision — no LLM involved.
    The LLM shouldn't waste tokens re-discovering what we can check in one line.
    """
    emitter = config["configurable"]["emitter"]

    emitter.emit(EventType.NODE_STARTED, node="evaluate_coverage")

    scored = state["scored_candidates"]
    above_threshold = [c for c in scored if c["composite_score"] >= COVERAGE_THRESHOLD_SCORE]
    sufficient = len(above_threshold) >= COVERAGE_MIN_CANDIDATES

    # Always send the top pre_rerank_count candidates to the re-ranker
    top_candidates = scored[:settings.pre_rerank_count]

    emitter.emit(
        EventType.ROUTING_DECISION,
        node="evaluate_coverage",
        data={
            "candidates_above_threshold": len(above_threshold),
            "threshold_score": COVERAGE_THRESHOLD_SCORE,
            "coverage_sufficient": sufficient,
            "routing_to": "llm_rerank" if sufficient else "expand_candidate_pool",
            "pool_size_for_rerank": len(top_candidates),
        },
    )

    logger.info(
        "coverage_evaluated",
        above_threshold=len(above_threshold),
        sufficient=sufficient,
    )

    return {
        "coverage_sufficient": sufficient,
        "top_candidates": top_candidates,
    }


def route_after_coverage(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node based on coverage result.
    """
    return "llm_rerank" if state["coverage_sufficient"] else "expand_candidate_pool"
