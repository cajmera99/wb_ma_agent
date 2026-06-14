"""
LLM-driven quality gate node.

After generate_rationales, assess the quality of all 10 rationale summaries
using a compact representation. Route to targeted_regeneration if 1–3 weak
rationales are identified, or proceed directly to END if quality is acceptable.

This is the primary LLM-driven routing node in the graph. The routing decision
requires qualitative judgment — template detection across 10 rationales, citation
density assessment, cross-acquirer comparison — that cannot be reduced to a Python
threshold check.
"""

import json
import re
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.agent.state import AgentState
from backend.agent.prompts import QUALITY_GATE_PROMPT_TEMPLATE
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)


def _citation_count(text: str) -> int:
    """Count specific data points in text: dollar amounts, multiples, percentages, years, deal counts."""
    if not text:
        return 0
    return len(re.findall(
        r'\$[\d,]+(?:\.\d+)?[MBK]?'   # $200M, $1.2B
        r'|\d+\.?\d*x'                 # 12.5x, 16x
        r'|\d+\.?\d*%'                 # 18%, 79%
        r'|\b20\d{2}\b'                # 2019, 2023
        r'|\b\d+\s+deals?\b',          # 8 deals, 1 deal
        text,
        re.IGNORECASE,
    ))


async def node_quality_gate(state: AgentState, config: RunnableConfig) -> dict:
    """
    LLM-driven quality check on all 10 rationales before PDF generation.

    Builds compact per-rationale summaries (Section 2 preview, conviction sentence,
    risk flag names, citation count) and sends them to the LLM in one call.
    The LLM identifies 0–3 weak rationales and returns a routing decision.

    Routes to:
      - END ("proceed_to_pdf") if quality is acceptable
      - targeted_regeneration ("regenerate_weak") if 1–3 failures are identified
    """
    emitter = config["configurable"]["emitter"]
    app_state = config["configurable"]["app_state"]

    emitter.emit(EventType.NODE_STARTED, node="quality_gate")

    rationales = state["rationales"]

    # Compact summaries — enough for cross-acquirer comparison without sending
    # 10 full rationales. Section 2 preview covers the most templating-prone section;
    # conviction_rationale and risk_flag_names cover the other two graded sections.
    summaries = []
    for r in rationales:
        if "error" in r:
            summaries.append({
                "acquirer_name": r.get("acquirer_name", "Unknown"),
                "section_2_preview": "[GENERATION FAILED — stub page]",
                "conviction_rationale": "[FAILED]",
                "risk_flag_names": [],
                "section_2_citation_count": 0,
                "conviction_level": "Unknown",
            })
            continue

        section_2 = r.get("strategic_fit_thesis", "")
        summaries.append({
            "acquirer_name": r.get("acquirer_name", "Unknown"),
            "section_2_preview": section_2[:220].strip(),
            "conviction_rationale": r.get("conviction_rationale", ""),
            "risk_flag_names": [
                f.get("risk_type", "") for f in r.get("risk_flags", [])
                if isinstance(f, dict)
            ],
            "section_2_citation_count": _citation_count(section_2),
            "conviction_level": r.get("conviction_level", "Unknown"),
        })

    prompt = QUALITY_GATE_PROMPT_TEMPLATE.format(
        count=len(summaries),
        rationale_summaries=json.dumps(summaries, indent=2),
    )

    quality_gate_result = {
        "routing": "proceed_to_pdf",
        "weak_acquirers": [],
        "issues": {},
        "reasoning": "defaulted to proceed — gate LLM did not run",
    }

    try:
        response = await app_state.llm_fast.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip markdown fences if the model wraps output in ```json ... ```
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)

        # Enforce max 3 regenerations regardless of LLM output
        weak = parsed.get("weak_acquirers", [])[:3]
        quality_gate_result = {
            "routing": "regenerate_weak" if weak else "proceed_to_pdf",
            "weak_acquirers": weak,
            "issues": parsed.get("issues", {}),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        # On gate failure: always proceed — the quality check should never block delivery
        logger.error("quality_gate_llm_failed", error=str(e))

    routing = quality_gate_result["routing"]
    weak = quality_gate_result["weak_acquirers"]
    reasoning = quality_gate_result["reasoning"]

    emitter.emit(EventType.ROUTING_DECISION, node="quality_gate", data={
        "routing_to": routing,
        "weak_acquirers": weak,
        "reasoning": reasoning,
    })
    emitter.emit(EventType.NODE_COMPLETED, node="quality_gate", data={
        "routing": routing,
        "weak_count": len(weak),
        "weak_acquirers": weak,
    })

    logger.info("quality_gate_complete", routing=routing, weak_acquirers=weak)
    return {"quality_gate_result": quality_gate_result}


def route_after_quality_gate(state: AgentState) -> str:
    """LangGraph conditional edge — reads the gate result and returns the target node name."""
    return state.get("quality_gate_result", {}).get("routing", "proceed_to_pdf")
