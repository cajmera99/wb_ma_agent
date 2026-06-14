import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from openai import RateLimitError, APIError

from backend.agent.state import AgentState
from backend.agent.prompts import SYSTEM_PROMPT, RERANK_PROMPT_TEMPLATE
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
def _call_llm_with_retry(llm_with_tools, messages: list) -> object:
    return llm_with_tools.invoke(messages)


async def node_llm_rerank(state: AgentState, config: RunnableConfig) -> dict:
    """
    The LLM receives the top-N scored candidates and selects the final 10.

    Tool use happens here — the LLM may call get_acquirer_profile or
    search_transactions to dig deeper before committing to its ranking.
    This is real agentic behavior: the LLM decides what additional
    context it needs, fetches it, then makes its decision.
    """
    emitter = config["configurable"]["emitter"]
    app_state = config["configurable"]["app_state"]

    emitter.emit(EventType.NODE_STARTED, node="llm_rerank")

    target = state["target"]
    candidates = state["top_candidates"]

    # Build a lean summary for each candidate (not the full profile — too many tokens)
    candidate_summaries = [
        {
            "acquirer_name": c["acquirer_name"],
            "acquirer_type": c["acquirer_type"],
            "composite_score": c["composite_score"],
            "sub_scores": c["sub_scores"],
            "total_deals": c["total_deals"],
            "adjacent_sector_deals": c["adjacent_sector_deals"],
            "median_deal_size_mm": c["median_deal_size_mm"],
            "median_ev_ebitda": c["median_ev_ebitda"],
            "top_rationale_tags": c["top_rationale_tags"],
            "recent_deal_count": c["recent_deal_count"],
        }
        for c in candidates
    ]

    prompt = RERANK_PROMPT_TEMPLATE.format(
        candidate_count=len(candidates),
        sector=target.sector,
        deal_size_mm=target.deal_size_mm,
        geography=target.geography,
        ownership=target.ownership,
        profile_description=target.profile_description or "Not specified",
        candidates_json=json.dumps(candidate_summaries, indent=2),
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Use gpt-4o (not mini) for rerank: gpt-4o calls 2-3 targeted profiles before
    # committing; mini calls all 20, bloating the context and adding noise to the log.
    llm_with_tools = app_state.llm.bind_tools(app_state.tools)

    # Tool-calling loop — the LLM may call tools before committing to its ranking
    tool_map = {t.name: t for t in app_state.tools}
    max_tool_rounds = 3

    for round_num in range(max_tool_rounds):
        try:
            response = _call_llm_with_retry(llm_with_tools, messages)
        except Exception as e:
            emitter.emit(EventType.NODE_ERROR, node="llm_rerank", data={"error": str(e)})
            logger.error("llm_rerank_failed", error=str(e))
            # Graceful fallback: use top 10 from scoring model directly
            fallback = [c["acquirer_name"] for c in candidates[:10]]
            return {
                "final_acquirer_names": fallback,
                "rerank_reasoning": "LLM rerank failed — using top 10 from scoring model.",
                "errors": state.get("errors", []) + [f"llm_rerank error: {e}"],
            }

        # If the LLM wants to call tools, execute them and continue the loop
        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                emitter.emit(EventType.TOOL_CALLED, node="llm_rerank", data={
                    "tool": tc["name"], "args": tc["args"]
                })
                tool_fn = tool_map.get(tc["name"])
                if tool_fn:
                    tool_result = tool_fn.invoke(tc["args"])
                    emitter.emit(EventType.TOOL_RESULT, node="llm_rerank", data={
                        "tool": tc["name"],
                        "result_preview": str(tool_result)[:200],
                    })
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tc["id"],
                    ))
        else:
            # No more tool calls — parse the final response
            break

    # Extract token usage if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        emitter.emit(EventType.LLM_TOKENS_USED, node="llm_rerank", data={
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
        })

    # Parse the JSON ranking from the LLM response
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        ranked = parsed["ranked_acquirers"][:10]
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        logger.warning("rerank_parse_failed", error=str(e), raw=response.content[:300])
        ranked = [c["acquirer_name"] for c in candidates[:10]]
        reasoning = "Parse failed — using top 10 from scoring model."

    emitter.emit(EventType.NODE_COMPLETED, node="llm_rerank", data={
        "final_acquirers": ranked,
        "reasoning_preview": reasoning[:200],
    })

    logger.info("rerank_complete", final=ranked)

    return {
        "final_acquirer_names": ranked,
        "rerank_reasoning": reasoning,
    }
