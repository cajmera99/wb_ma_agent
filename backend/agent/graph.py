"""
LangGraph agent graph for the M&A Acquirer Identification Engine.

Graph topology:
                                  ┌──────────────────────┐
                                  │    score_and_rank     │  (pure Python, all acquirers)
                                  └──────────┬───────────┘
                                             │
                                  ┌──────────▼───────────┐
                                  │  evaluate_coverage    │  (deterministic routing)
                                  └──────────┬───────────┘
                            sufficient?      │       not sufficient?
                 ┌───────────────────────────┤
                 │                           │
  ┌──────────────▼──────────┐   ┌────────────▼─────────────┐
  │      llm_rerank          │◄──│  expand_candidate_pool   │
  └──────────────┬──────────┘   └──────────────────────────┘
                 │
  ┌──────────────▼──────────┐
  │   generate_rationales    │  (10 concurrent LLM calls)
  └──────────────┬───────────┘
                 │
  ┌──────────────▼──────────┐
  │      quality_gate        │  (LLM-driven routing — qualitative cross-comparison)
  └──────────────┬───────────┘
     proceed?    │    regenerate_weak?
  ┌──────────────┤───────────────────────────────────────┐
  │              │                                       │
 END  ┌──────────▼──────────┐                           │
      │ targeted_regeneration│  (re-runs 1–3 weak LLM   │
      │                      │   calls with tighter      │
      └──────────────────────┘   constraints)            │
                 │                                       │
                END ◄───────────────────────────────────┘

All nodes receive app_state, emitter, and run_id through config["configurable"].
This avoids global state and keeps each run fully isolated.
"""

from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.nodes import (
    node_score_and_rank,
    node_evaluate_coverage,
    route_after_coverage,
    node_expand_candidate_pool,
    node_llm_rerank,
    node_generate_rationales,
    node_quality_gate,
    route_after_quality_gate,
    node_targeted_regeneration,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready to be invoked with:
        graph.ainvoke(initial_state, config={"configurable": {...}})

    Called once at server startup; the compiled graph object is reused
    for every incoming request (it holds no per-run state itself).
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("score_and_rank", node_score_and_rank)
    graph.add_node("evaluate_coverage", node_evaluate_coverage)
    graph.add_node("expand_candidate_pool", node_expand_candidate_pool)
    graph.add_node("llm_rerank", node_llm_rerank)
    graph.add_node("generate_rationales", node_generate_rationales)
    graph.add_node("quality_gate", node_quality_gate)
    graph.add_node("targeted_regeneration", node_targeted_regeneration)

    # Entry point
    graph.set_entry_point("score_and_rank")

    # Linear edges
    graph.add_edge("score_and_rank", "evaluate_coverage")

    # Conditional edge: evaluate_coverage → llm_rerank OR expand_candidate_pool
    # route_after_coverage reads state["coverage_sufficient"] and returns
    # the node name string. LangGraph calls the function with state at runtime.
    graph.add_conditional_edges(
        "evaluate_coverage",
        route_after_coverage,
        {
            "llm_rerank": "llm_rerank",
            "expand_candidate_pool": "expand_candidate_pool",
        },
    )

    # After pool expansion, always proceed to rerank
    graph.add_edge("expand_candidate_pool", "llm_rerank")

    # After rerank, generate all rationales
    graph.add_edge("llm_rerank", "generate_rationales")

    # After rationale generation, run the LLM-driven quality gate
    graph.add_edge("generate_rationales", "quality_gate")

    # Conditional edge: quality_gate → END (proceed_to_pdf) OR targeted_regeneration
    # route_after_quality_gate reads state["quality_gate_result"]["routing"].
    # This is the primary LLM-driven routing decision in the graph.
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "proceed_to_pdf": END,
            "regenerate_weak": "targeted_regeneration",
        },
    )

    # After targeted regeneration, always proceed to PDF
    graph.add_edge("targeted_regeneration", END)

    return graph.compile()


# Module-level compiled graph — built once, shared across all requests
compiled_graph = build_graph()
