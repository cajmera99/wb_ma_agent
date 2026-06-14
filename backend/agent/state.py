from typing import TypedDict
from backend.models.target import TargetProfile


class AgentState(TypedDict):
    """
    The single object that flows through every node in the LangGraph graph.
    Each node receives the full state and returns a dict of fields to update.
    Fields are accumulated — a node only needs to return what it changed.
    """
    # Set at the start of every run — never mutated after
    run_id: str
    target: TargetProfile

    # Populated by score_and_rank
    scored_candidates: list[dict]   # all acquirers, scored and sorted

    # Routing flag set by evaluate_coverage
    coverage_sufficient: bool

    # Top-N pool sent to the LLM re-ranker (subset of scored_candidates)
    top_candidates: list[dict]

    # Set by llm_rerank — final 10 names in order
    final_acquirer_names: list[str]
    rerank_reasoning: str

    # Set by generate_rationales — one dict per acquirer
    rationales: list[dict]

    # Non-fatal errors — logged and surfaced to frontend without crashing
    errors: list[str]
