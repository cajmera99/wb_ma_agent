"""
Targeted regeneration node.

Receives 1–3 weak acquirer names from the quality gate and re-runs
_generate_one() for those specific acquirers only. All other rationales
are preserved unchanged. Routes unconditionally to END.
"""

import asyncio
from langchain_core.runnables import RunnableConfig

from backend.agent.state import AgentState
from backend.agent.nodes.generate_rationales import _generate_one, _normalize_acquirer_type
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)


async def node_targeted_regeneration(state: AgentState, config: RunnableConfig) -> dict:
    """
    Re-generate the 1–3 weak rationales identified by node_quality_gate.

    Uses the same _generate_one() function as the original generate_rationales
    node — same evidence assembly, same anomaly signals, same repair loop.
    Replaces the weak entries in state["rationales"] in-place.
    """
    emitter = config["configurable"]["emitter"]
    app_state = config["configurable"]["app_state"]

    emitter.emit(EventType.NODE_STARTED, node="targeted_regeneration")

    quality_result = state.get("quality_gate_result", {})
    weak_names = quality_result.get("weak_acquirers", [])
    issues = quality_result.get("issues", {})

    if not weak_names:
        emitter.emit(EventType.NODE_COMPLETED, node="targeted_regeneration", data={"regenerated": 0})
        return {"regeneration_attempted": True}

    target = state["target"]
    scored_map = {c["acquirer_name"]: c for c in state["scored_candidates"]}
    tool_map = {t.name: t for t in app_state.tools}
    final_names = state["final_acquirer_names"]

    strategic_names = [
        n for n in final_names
        if _normalize_acquirer_type(scored_map.get(n, {}).get("acquirer_type", "Strategic")) == "Strategic"
    ]

    sem = asyncio.Semaphore(3)

    async def _regen(name: str):
        async with sem:
            candidate = scored_map.get(name, {})
            rank = (final_names.index(name) + 1) if name in final_names else 0
            is_pe = _normalize_acquirer_type(candidate.get("acquirer_type", "Strategic")) == "Financial Sponsor"
            co_strategics = strategic_names if is_pe else []
            issue = issues.get(name, "")
            logger.info("targeted_regen_starting", acquirer=name, issue=issue)
            result = await _generate_one(
                name, rank, candidate, target, tool_map, app_state, emitter, co_strategics
            )
            return name, result

    regen_results = await asyncio.gather(
        *[_regen(n) for n in weak_names],
        return_exceptions=True,
    )

    # Build replacement map and merge into existing rationales list
    regen_map = {}
    for item in regen_results:
        if isinstance(item, Exception):
            logger.error("targeted_regen_item_failed", error=str(item))
        else:
            name, rationale = item
            regen_map[name] = rationale
            logger.info("rationale_regenerated", acquirer=name)

    updated = list(state["rationales"])
    for i, r in enumerate(updated):
        name = r.get("acquirer_name", "")
        if name in regen_map:
            updated[i] = regen_map[name]

    succeeded = len(regen_map)
    failed = len(weak_names) - succeeded

    emitter.emit(EventType.NODE_COMPLETED, node="targeted_regeneration", data={
        "regenerated": succeeded,
        "failed": failed,
        "acquirers": list(regen_map.keys()),
    })
    logger.info("targeted_regeneration_complete", succeeded=succeeded, failed=failed)

    return {"rationales": updated, "regeneration_attempted": True}
