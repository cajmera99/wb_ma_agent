import asyncio
import json
import re
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.agent.state import AgentState
from backend.agent.prompts import SYSTEM_PROMPT, RATIONALE_PROMPT_TEMPLATE
from backend.models.rationale import AcquirerRationale
from backend.models.events import EventType
import structlog

logger = structlog.get_logger(__name__)


def _normalize_acquirer_type(raw: str) -> str:
    """Map raw acquirer_type strings to the two-value Literal the model expects."""
    if any(t in raw.lower() for t in ["pe", "private equity", "financial sponsor", "fund", "buyout"]):
        return "Financial Sponsor"
    return "Strategic"


async def _generate_one(
    acquirer_name: str,
    rank: int,
    candidate: dict,
    target,
    tool_map: dict,
    app_state,
    emitter,
    co_strategics: list[str] | None = None,
    cached_comps_json: str | None = None,
) -> dict:
    """
    Build the full evidence packet for one acquirer then call the LLM.
    This function runs concurrently for all 10 acquirers via asyncio.gather.

    Evidence is gathered by calling tools directly in Python — not via LLM tool
    use. The LLM's job here is synthesis, not retrieval. This keeps latency
    down and prevents the LLM from fetching data it doesn't need.
    """
    # sub_scores come from scorer.py already in 0-100 range
    sub_scores = candidate.get("sub_scores", {})
    acquirer_type_raw = candidate.get("acquirer_type", "Strategic")
    acquirer_type = _normalize_acquirer_type(acquirer_type_raw)

    # ── Sector-specific deal count ─────────────────────────────────────────────
    # sector_counts is a dict like {"Healthcare Services": 6, "Behavioral Health": 2}.
    # Passing only sub_sector_counts previously made it impossible for the LLM to
    # write "completed 6 Healthcare Services deals" — we had no sector-level count.
    primary_sector_deal_count = candidate.get("sector_counts", {}).get(target.sector, 0)

    # ── Deal size range ────────────────────────────────────────────────────────
    # Min-max range across ALL deals. "Typical range: $50M–$800M" is meaningful
    # context the LLM can use without needing individual deal sizes.
    _min_sz = candidate.get("min_deal_size_mm")
    _max_sz = candidate.get("max_deal_size_mm")
    deal_size_range = (
        f"${_min_sz:.0f}M–${_max_sz:.0f}M"
        if _min_sz is not None and _max_sz is not None
        else "N/A"
    )

    # ── Deals near target EV ───────────────────────────────────────────────────
    # Count deals that fall within 0.5x–2.0x of the target EV. For a $200M target
    # this is $100M–$400M. Enables "N deals in the comparable size range" sentences
    # without fabricating a range the data doesn't support.
    _deal_sizes = candidate.get("deal_sizes_mm", [])
    _band_low = target.deal_size_mm * 0.5
    _band_high = target.deal_size_mm * 2.0
    deals_near_target = sum(1 for s in _deal_sizes if _band_low <= s <= _band_high)
    target_size_band = f"${_band_low:.0f}M–${_band_high:.0f}M"

    # Compute a conviction baseline in Python so the LLM has a principled anchor.
    # Without this, similar composite scores produce inconsistent conviction levels
    # because the LLM decides freely based on whichever signal it emphasises first.
    composite = candidate.get("composite_score", 0)
    if composite > 80:
        conviction_baseline = "High"
    elif composite >= 50:
        conviction_baseline = "Medium"
    else:
        conviction_baseline = "Low"

    # Pre-compute anomaly flags — injected directly into the prompt as ⚠ attention signals.
    # When these appear the LLM reliably addresses them; without explicit flagging it defaults
    # to geographic expansion boilerplate and misses the most differentiating data signals.
    # NOTE: anomaly_parts is intentionally a list here so valuation posture (computed after
    # the comps fetch below) can be appended before the final string is assembled.
    anomaly_parts = []
    median_size = candidate.get("median_deal_size_mm")
    max_size = candidate.get("max_deal_size_mm")
    if median_size:
        size_ratio = target.deal_size_mm / median_size
        # Median alone is misleading when an acquirer's deal history spans a wide range.
        # An acquirer with median $100M but max $500M is NOT stretching to do a $200M deal.
        # Flag genuine size stretch only when the target exceeds their entire demonstrated
        # range — i.e., no prior deal within $75M of the target EV.
        has_range_experience = (max_size is not None and max_size >= (target.deal_size_mm - 75))

        if size_ratio > 1.5 and not has_range_experience:
            # Genuine stretch: above median AND no prior deal near target size
            max_display = f"${max_size:.0f}M" if max_size else "unknown"
            anomaly_parts.append(
                f"⚠ DEAL SIZE GENUINE STRETCH: Target EV (${target.deal_size_mm:.0f}M) is "
                f"{size_ratio:.1f}x above this acquirer's median (${median_size:.0f}M) AND "
                f"their largest prior deal ({max_display}) is more than $75M below the target — "
                f"they have no demonstrated experience at this deal size. "
                f"Section 2 should address this. In Section 5 category (b), use EXACTLY: "
                f"'{size_ratio:.1f}x Above Median Deal Size — target EV of "
                f"${target.deal_size_mm:.0f}M vs acquirer median ${median_size:.0f}M and "
                f"max prior deal of {max_display}.' Direction is ABOVE. Never say 'Below.'"
            )
        elif size_ratio > 1.5 and has_range_experience:
            # Above median but they've done deals at this size before — not a valid risk.
            max_display = f"${max_size:.0f}M" if max_size else "above target"
            anomaly_parts.append(
                f"⚠ AT-SIZE (RANGE COVERS TARGET): Target EV ${target.deal_size_mm:.0f}M is "
                f"{size_ratio:.1f}x above their median (${median_size:.0f}M), BUT their "
                f"largest prior deal is {max_display} — target is within their demonstrated range. "
                f"Do NOT use category (b) Deal Size Mismatch in Section 5 — the median is "
                f"not the right benchmark here; they have done deals at this size before. "
                f"Use (c), (d), (e), (f), (g), or (h) instead."
            )
        elif size_ratio < 0.5:
            inv_ratio = round(1 / size_ratio, 1)
            anomaly_parts.append(
                f"⚠ DEAL SIZE BELOW MEDIAN: Target EV (${target.deal_size_mm:.0f}M) is "
                f"well BELOW this acquirer's historical median deal size (${median_size:.0f}M) — "
                f"the target is {inv_ratio:.1f}x smaller than their typical deal. "
                f"In Section 5 category (b), use EXACTLY: "
                f"'{inv_ratio:.1f}x Below Median Deal Size — target EV of "
                f"${target.deal_size_mm:.0f}M vs acquirer median ${median_size:.0f}M.' "
                f"Direction is BELOW (target is SMALLER than their median). Never say 'Above.'"
            )
        else:
            # Ratio between 0.5x and 1.5x — sizes are close enough to be immaterial.
            _at_size_dir = "LARGER than" if size_ratio > 1.0 else "SMALLER than"
            _at_size_pct = abs(size_ratio - 1.0) * 100
            anomaly_parts.append(
                f"⚠ AT-SIZE DEAL: Target EV ${target.deal_size_mm:.0f}M vs acquirer median "
                f"${median_size:.0f}M = {size_ratio:.2f}x ratio (within 0.5-1.5x normal band). "
                f"DIRECTION: The target is {_at_size_dir} this acquirer's median "
                f"({_at_size_pct:.0f}% difference). "
                f"If ratio > 1.0 the target is LARGER; if < 1.0 it is SMALLER. "
                f"Never invert this when comparing sizes in Section 2. "
                f"Do NOT use category (b) Deal Size Mismatch in Section 5 — this is not a "
                f"material risk. Choose from (c), (d), (e), (f), (g), or (h) instead."
            )

    outcome_score = sub_scores.get("outcome", 100)
    closed = candidate.get("closed_deals", 0)
    total = candidate.get("total_deals", 1)
    pct = round(closed / total * 100) if total else 0
    if outcome_score < 70:
        anomaly_parts.append(
            f"⚠ DEAL COMPLETION RATE {outcome_score:.0f}/100: This acquirer has closed "
            f"only {closed} of {total} deals ({pct}% completion rate). At least one "
            f"risk flag must specifically reference this deal completion track record — "
            f"name the withdrawn or pending deals visible in the precedent data."
        )
    else:
        # Explicitly block category (e) so the LLM doesn't use it as a lazy filler
        # risk when the acquirer actually has a solid completion track record.
        anomaly_parts.append(
            f"✓ COMPLETION RATE STRONG: {closed} of {total} deals closed ({pct}%). "
            f"Do NOT use category (e) Deal Completion Rate in Section 5 — "
            f"this acquirer's track record is solid and it is not a valid risk. "
            f"Choose from (c), (d), (f), (g), or (h) instead."
        )

    # Ownership mismatch — flag when the acquirer rarely acquires private companies
    # but the target is private. The scoring model captures this as a sub-score but
    # the LLM ignores numeric scores without an explicit signal.
    ownership_score = sub_scores.get("ownership", 100)
    if ownership_score < 25:
        target_ownership = target.ownership
        ownership_counts = candidate.get("target_ownership_counts", {})
        private_count = ownership_counts.get("Private", 0) + ownership_counts.get("PE-Backed", 0)
        anomaly_parts.append(
            f"⚠ OWNERSHIP MISMATCH: Target is {target_ownership} but only {private_count} of "
            f"{total} prior deals involved private or PE-backed targets (ownership score "
            f"{ownership_score:.0f}/100). This acquirer predominantly acquires non-private "
            f"companies. Flag this in Section 5 using category (c) Deal Type Mismatch or a "
            f"specific risk noting their limited experience with private-company transactions."
        )

    # EBITDA margin differentiation signal is appended to anomaly_parts after the
    # precedent deals fetch (Step 1) — it references historical acquisition margins
    # from acquired_co_ebitda_margin_pct, which requires the fetched deal data.

    # Valuation posture is computed AFTER the comps fetch (market median needed).
    # Placeholder — appended to anomaly_parts below after Step 2.
    acquirer_median_ebitda = candidate.get("median_ev_ebitda")

    # For PE sponsors: inject the list of strategic co-acquirers in this run as
    # named exit buyer candidates. Without this, all 5 PE sponsors default to the
    # same generic "regional hospital system or national health services platform" phrase.
    if acquirer_type == "Financial Sponsor" and co_strategics:
        co_list = "\n".join(f"  - {name}" for name in co_strategics)
        co_acquirer_context = (
            "STRATEGIC CO-ACQUIRERS IN THIS ANALYSIS (use as exit buyer candidates)\n"
            "------------------------------------------------------------------------\n"
            "These strategic buyers appear in the same 10-acquirer shortlist. For the\n"
            "exit optionality statement in Section 2, name at least one specifically\n"
            "and explain why they would pay a premium for what this PE firm is building:\n"
            + co_list + "\n\n"
        )
    else:
        co_acquirer_context = ""

    # Step 1: Fetch precedent deals — two-pass to prioritise target-sector deals
    # first, then fill remaining slots with any-sector deals up to a total of 8.
    # Use the canonical name from the scored profile, not the LLM-output name, to
    # guard against the rerank LLM slightly altering a name (e.g. "Nordic Capital AB"
    # → "Nordic Capital") which would cause the tool's exact-match filter to return 0.
    canonical_name = candidate.get("acquirer_name") or acquirer_name
    deals_tool = tool_map.get("get_acquirer_precedent_deals")
    precedent_deals_json = json.dumps({"acquirer": canonical_name, "count": 0, "deals": []})
    if deals_tool:
        try:
            # Pass 1: deals in the target's primary sector (highest relevance)
            primary_raw = deals_tool.invoke({
                "acquirer_name": canonical_name,
                "sectors": [target.sector],
                "max_results": 5,
            })
            primary_data = json.loads(
                primary_raw if isinstance(primary_raw, str) else json.dumps(primary_raw)
            )
            primary_deals = primary_data.get("deals", [])

            # Pass 2: all deals for this acquirer (used to fill remaining slots)
            all_raw = deals_tool.invoke({
                "acquirer_name": canonical_name,
                "max_results": 10,
            })
            all_data = json.loads(
                all_raw if isinstance(all_raw, str) else json.dumps(all_raw)
            )
            all_deals = all_data.get("deals", [])

            # Merge: primary-sector deals first, then non-duplicate fillers, cap at 5
            seen_ids = {d.get("transaction_id") for d in primary_deals}
            filler = [d for d in all_deals if d.get("transaction_id") not in seen_ids]
            combined = (primary_deals + filler)[:5]

            precedent_deals_json = json.dumps({
                "acquirer": canonical_name,
                "count": len(combined),
                "deals": combined,
            })
        except Exception as e:
            logger.warning("precedent_deals_fetch_failed", acquirer=canonical_name, error=str(e))

    # Flag any precedent deal >3x the target EV so the LLM must acknowledge the size
    # gap explicitly when citing these deals, rather than silently using a $4B deal
    # as evidence of readiness for a $200M target.
    try:
        _deals_parsed = json.loads(precedent_deals_json)
        _oversized = []
        for _deal in _deals_parsed.get("deals", []):
            _sz = _deal.get("deal_size_mm")
            if _sz and _sz > target.deal_size_mm * 3:
                _ratio = _sz / target.deal_size_mm
                _oversized.append(
                    f"  - {_deal.get('target_company', 'Unknown')} "
                    f"(${_sz:.0f}M = {_ratio:.1f}x the ${target.deal_size_mm:.0f}M target)"
                )
        if _oversized:
            anomaly_parts.append(
                "⚠ OVERSIZED PRECEDENTS — MANDATORY DISCLOSURE WHEN CITED:\n"
                + "\n".join(_oversized) + "\n"
                f"If ANY of the above deals appear in Section 2 or Section 6 as evidence "
                f"of fit for the ${target.deal_size_mm:.0f}M target, you MUST state: "
                f"(1) the exact size ratio, and (2) what it proves vs. does NOT prove. "
                f"Example: 'Their $Xm deal proves they can execute a complex process, "
                f"but does not validate size discipline at the ${target.deal_size_mm:.0f}M level.' "
                "Citing an oversized deal without this disclosure is a reportable failure."
            )
    except Exception:
        pass

    # Pre-compute withdrawn/terminated deal names from the fetched 5-deal sample.
    # The completion rate signal (above) tells the LLM to cite withdrawn deals by name;
    # without providing the exact names the LLM fabricates them — sometimes misattributing
    # Closed deals as withdrawn (Welsh Carson, Nordic Capital) or producing internal
    # contradictions ("6 Withdrawn Deals — None in dataset", VNS Health).
    _withdrawn_deal_names: list[str] = []
    try:
        _pd_for_wdraw = json.loads(precedent_deals_json)
        for _wd in _pd_for_wdraw.get("deals", []):
            if _wd.get("outcome") in ("Withdrawn", "Terminated"):
                _wn = _wd.get("target_company", "Unknown")
                _wy = _wd.get("year", "?")
                _withdrawn_deal_names.append(f"{_wn} ({_wy})")
    except Exception:
        pass

    if _withdrawn_deal_names:
        anomaly_parts.append(
            "⚠ WITHDRAWN/TERMINATED DEALS IN YOUR PRECEDENT DATA — EXACT NAMES (USE THESE ONLY):\n"
            + "\n".join(f"  - {n}" for n in _withdrawn_deal_names) + "\n"
            "When writing risk flag category (d) in Section 5, cite ONLY the names above. "
            "Every other deal in the precedent JSON shows 'Closed' as its outcome — "
            "NEVER describe a Closed deal as 'withdrawn pre-close'. Never invent a deal "
            "name that does not appear in this list. Fabricating a withdrawn deal name "
            "when the precedent JSON shows it as Closed is a critical factual error that "
            "will be flagged."
        )
    else:
        anomaly_parts.append(
            "⚠ WITHDRAWN DEALS: Zero of the 5 precedent deals shown have a Withdrawn or "
            "Terminated outcome — ALL are Closed. Do NOT write risk flag category (d) with "
            "any specific deal names, because no withdrawn deals appear in your precedent data. "
            "If the completion rate signal above indicates non-closed deals exist in the "
            "broader dataset (beyond the 5 shown), acknowledge the aggregate count only — "
            "e.g. 'X of Y total deals were not completed' — but NEVER invent a deal name "
            "to populate category (d). A fabricated withdrawn deal name is a factual error."
        )

    # EBITDA margin differentiation signal — computed here so we can reference the
    # historical acquisition margin profile from the fetched precedent deals.
    # "Strong EBITDA margins" is real information; the goal is to force acquirer-specific
    # framing rather than the generic boilerplate that applies to all 10 identically.
    _margin_data = []
    try:
        _deals_for_margin = json.loads(precedent_deals_json)
        for _d in _deals_for_margin.get("deals", []):
            _m = _d.get("acquired_co_ebitda_margin_pct")
            if _m is not None:
                _margin_data.append(float(_m))
    except Exception:
        pass

    if _margin_data:
        _hist_margin_str = (
            f"This acquirer's precedent acquisitions averaged "
            f"{round(sum(_margin_data) / len(_margin_data), 1):.1f}% EBITDA margins "
            f"(acquired_co_ebitda_margin_pct across {len(_margin_data)} deals above). "
            f"State whether the target's 'strong margins' suggest a profile above, at, or "
            f"below that historical threshold — and what the conclusion implies for deal "
            f"rationale or pricing."
        )
    else:
        _hist_margin_str = (
            "Historical margin data from precedent deals is not available for this acquirer. "
            "Reference the target's strong margins in relation to their valuation model "
            "(entry/exit multiple, IRR math, or sector median) instead."
        )

    if acquirer_type == "Financial Sponsor":
        anomaly_parts.append(
            "TARGET MARGIN SIGNAL — DIFFERENTIATE FOR PE SPONSOR: "
            "The target carries 'strong EBITDA margins' (no % disclosed). For a PE sponsor "
            "high entry margins mean a larger absolute EBITDA base — directly affecting what "
            "entry multiple the fund can justify to hit target IRR at the prevailing exit "
            f"multiple. {_hist_margin_str} "
            "FORBIDDEN: 'the target's strong EBITDA margins complement / align with / support "
            "this sponsor's strategy' — this phrase applies to every PE firm on the shortlist "
            "and is not analysis. Replace it with the IRR or portfolio-specific argument above."
        )
    else:
        anomaly_parts.append(
            "TARGET MARGIN SIGNAL — DIFFERENTIATE FOR STRATEGIC BUYER: "
            "The target carries 'strong EBITDA margins' (no % disclosed). Use this as a "
            "signal that means something specific to THIS acquirer — not a generic statement. "
            f"{_hist_margin_str} "
            "FORBIDDEN: 'the target's strong EBITDA margins complement / align with / support "
            "this acquirer's strategy' — this phrase applies to every strategic on the shortlist "
            "and is not analysis. Replace it with the margin comparison or a named valuation / "
            "synergy argument specific to this buyer."
        )

    # Step 2: Market valuation comps — identical for all acquirers in a run (same target
    # sector + size range). Use the pre-fetched result if available; only fall back to a
    # live fetch if this acquirer is being regenerated without the cached value.
    if cached_comps_json is not None:
        valuation_comps_json = cached_comps_json
    else:
        comps_tool = tool_map.get("get_valuation_comps")
        valuation_comps_json = "{}"
        if comps_tool:
            try:
                size = target.deal_size_mm
                comps_result = comps_tool.invoke({
                    "sectors": [target.sector],
                    "deal_size_min": size * 0.4,
                    "deal_size_max": size * 2.5,
                })
                valuation_comps_json = comps_result if isinstance(comps_result, str) else json.dumps(comps_result)
            except Exception as e:
                logger.warning("valuation_comps_fetch_failed", acquirer=acquirer_name, error=str(e))

    # Valuation posture signal — computed now that we have the market median.
    # This fixes a persistent bug where below-market buyers get labelled "Valuation Premium"
    # (they historically pay LESS than market, so the risk is a stretch upward, not a premium).
    try:
        _comps_parsed = json.loads(valuation_comps_json)
        _ev_ebitda_obj = _comps_parsed.get("ev_ebitda_multiple") or {}
        market_median_ebitda = _ev_ebitda_obj.get("median")
    except Exception:
        market_median_ebitda = None

    if acquirer_median_ebitda and market_median_ebitda:
        # gap_pct: % difference relative to market median — used for threshold checks
        # and the "above-market" description (market is the reference base there).
        gap_pct = (acquirer_median_ebitda - market_median_ebitda) / market_median_ebitda * 100
        acq_str = f"{acquirer_median_ebitda:.1f}x"
        mkt_str = f"{market_median_ebitda:.1f}x"
        # turns_diff: additive difference in EV/EBITDA multiples. "+4.5 turns above market"
        # is standard banker language. "4.5x Premium" was ambiguous — "x" suffix implies a
        # multiplier (paying 4.5 TIMES the market price), not an additive turn difference.
        turns_diff = round(acquirer_median_ebitda - market_median_ebitda, 1)
        if gap_pct > 15:
            anomaly_parts.append(
                f"⚠ ABOVE-MARKET PAYER: Historical median EV/EBITDA {acq_str} is "
                f"{gap_pct:.0f}% above market ({mkt_str}). In Section 5, name the risk "
                f"EXACTLY as: 'Above-Market Payer — {acq_str} historical median vs "
                f"{mkt_str} market median (+{turns_diff} turns, +{gap_pct:.0f}% above market); "
                f"exit multiple compression amplifies IRR risk.' "
                f"Use {acq_str}, {mkt_str}, and +{turns_diff} turns — do not substitute any other numbers."
            )
        elif gap_pct < -10:
            # stretch_pct: "must bid X% above historical comfort" means the base is the
            # acquirer's OWN median (their comfort level), NOT the market median.
            # Bug fix: old code used abs(gap_pct) which had market as denominator, giving
            # 18% instead of the correct 22% for a 9.6x acquirer vs 11.7x market.
            stretch_pct = round((market_median_ebitda - acquirer_median_ebitda) / acquirer_median_ebitda * 100)
            anomaly_parts.append(
                f"⚠ BELOW-MARKET BUYER: Historical median EV/EBITDA {acq_str} is "
                f"{abs(gap_pct):.0f}% below market ({mkt_str}). In Section 5, name the risk "
                f"EXACTLY as: 'Market Rate Stretch Required — must bid {stretch_pct}% "
                f"above historical {acq_str} comfort to win at prevailing {mkt_str} market rates.' "
                f"Do NOT call this 'Valuation Premium' — this acquirer pays BELOW market."
            )
        else:
            # At-market buyer — explicitly block the LLM from inventing a valuation risk.
            anomaly_parts.append(
                f"⚠ AT-MARKET BUYER: Historical median EV/EBITDA {acq_str} is within 15% "
                f"of market ({mkt_str}). Do NOT use category (a) Valuation Direction as a "
                f"risk flag — skip it and use categories (g) Antitrust or (h) Competitive "
                f"Process instead. Do not reference {mkt_str} as a stretch or premium."
            )

    # Zero-tolerance signal for the two highest-frequency forbidden filler phrases.
    # Instruction-only prohibition in the prompt is insufficient — the LLM skims
    # long forbidden lists and reverts to templates. A pre-computed ⚠ signal
    # immediately before the output task forces explicit attention before writing.
    anomaly_parts.append(
        "⚠ FORBIDDEN FILLER PHRASES — WRITE THE REPLACEMENT, NOT THE PHRASE:\n"
        "These appear in every first draft and contain zero analysis:\n\n"
        f"  BANNED: 'positions {acquirer_name} [uniquely/well/to] [leverage/capitalize/...]'\n"
        "  BANNED: 'positions them [uniquely/well/to/as]' in any form\n"
        "  BANNED: 'fills a [critical/specific/key/unique/strategic/operational] [X] gap'\n"
        "  BANNED: 'fills a critical need for [X]'\n"
        "  BANNED: 'adds a critical [X] capability'\n\n"
        "REQUIRED replacement pattern for each:\n"
        "  FOR 'positions them' → Write what the DATA shows they can actually do: cite a "
        "specific deal count, sub-sector from sub_sector_counts, or size from the precedent "
        "table. e.g. '4 of 24 deals in Healthcare Services concentrated in [sub-sector] make "
        "this target a direct platform extension, unlike the other sponsors with 0–1 sector deals.'\n"
        "  FOR 'fills a gap' → Name the absent sub-sector from sub_sector_counts, then "
        "state WHY this specific acquirer needs it: e.g. 'Nordic Capital has 0 Home Health "
        "deals in 4 Healthcare Services acquisitions — this target adds the one sub-sector "
        "absent from their platform.'"
    )

    # Assemble the complete anomaly flags block now that all signals are computed.
    anomaly_flags = (
        "DATA SIGNALS REQUIRING EXPLICIT TREATMENT\n"
        "------------------------------------------\n"
        + "\n".join(anomaly_parts)
        + "\n\n"
        if anomaly_parts else ""
    )

    # Step 3: Build the prompt with the full evidence packet
    most_recent_platform_year = candidate.get("most_recent_platform_year")
    platform_display = str(most_recent_platform_year) if most_recent_platform_year else "None in dataset"

    # The profiler's adjacent_sector_deals count includes the target sector itself when
    # the target sector appears in ADJACENT_SECTORS (e.g. Healthcare Services). Subtract
    # primary sector deals so the prompt doesn't double-count the same deals.
    adjacent_sector_deals_display = max(
        0,
        candidate.get("adjacent_sector_deals", 0) - primary_sector_deal_count,
    )

    prompt = RATIONALE_PROMPT_TEMPLATE.format(
        anomaly_flags=anomaly_flags,
        co_acquirer_context=co_acquirer_context,
        conviction_baseline=conviction_baseline,
        sector=target.sector,
        deal_size_mm=target.deal_size_mm,
        geography=target.geography,
        ownership=target.ownership,
        profile_description=target.profile_description or "Not specified",
        acquirer_name=acquirer_name,
        acquirer_type=acquirer_type,
        composite_score=round(candidate.get("composite_score", 0), 1),
        total_deals=candidate.get("total_deals", 0),
        closed_deals=candidate.get("closed_deals", 0),
        primary_sector_deal_count=primary_sector_deal_count,
        adjacent_sector_deals=adjacent_sector_deals_display,
        deals_near_target=deals_near_target,
        target_size_band=target_size_band,
        deal_size_range=deal_size_range,
        sector_counts=candidate.get("sector_counts", {}),
        median_deal_size_mm=candidate.get("median_deal_size_mm", "N/A"),
        median_ev_ebitda=candidate.get("median_ev_ebitda", "N/A"),
        median_ev_revenue=candidate.get("median_ev_revenue", "N/A"),
        top_rationale_tags=candidate.get("top_rationale_tags", []),
        deal_type_counts=candidate.get("deal_type_counts", {}),
        sub_sector_counts=candidate.get("sub_sector_counts", {}),
        geography_counts=candidate.get("geography_counts", {}),
        recent_deal_count=candidate.get("recent_deal_count", 0),
        most_recent_year=candidate.get("most_recent_year", "N/A"),
        most_recent_platform_year=platform_display,
        bolt_ons_since_platform=candidate.get("bolt_ons_since_platform", 0),
        is_active_rollup=candidate.get("is_active_rollup", False),
        score_sector=sub_scores.get("sector", 0),
        score_deal_size=sub_scores.get("deal_size", 0),
        score_rationale=sub_scores.get("rationale", 0),
        score_recency=sub_scores.get("recency", 0),
        score_outcome=sub_scores.get("outcome", 0),
        score_ownership=sub_scores.get("ownership", 0),
        precedent_deals_json=precedent_deals_json,
        valuation_comps_json=valuation_comps_json,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Step 4: Call LLM with structured output — Pydantic validates the response.
    # gpt-4o-mini (llm_fast): rationale generation is synthesis from pre-assembled data,
    # not judgment or tool selection. Mini handles structured output reliably and has
    # ~10× higher TPM limits than gpt-4o, eliminating the rate-limit stalls that caused
    # 40+ second delays when 6+ gpt-4o rationale calls exhausted the per-minute budget.
    llm_structured = app_state.llm_fast.with_structured_output(AcquirerRationale)
    rationale: AcquirerRationale | None = None

    try:
        rationale = await _call_structured_with_retry(llm_structured, messages)
    except ValidationError as validation_error:
        # Pydantic schema mismatch — the LLM returned a structurally invalid response.
        # Run the repair loop: send the error back and ask the LLM to fix its output.
        emitter.emit(EventType.VALIDATION_FAILED, node="generate_rationales", data={
            "acquirer": acquirer_name, "error": str(validation_error)[:300]
        })
        logger.warning(
            "structured_output_validation_failed_attempting_repair",
            acquirer=acquirer_name,
            error=str(validation_error),
        )
        repair_prompt = (
            f"Your previous response failed schema validation with this error:\n"
            f"{validation_error}\n\n"
            f"Please re-read the requirements above and produce a corrected response "
            f"that fully satisfies the schema. Pay particular attention to:\n"
            f"- precedent_deals must be a list of deal objects with all required fields\n"
            f"- risk_flags must contain between 2 and 3 items\n"
            f"- conviction_level must be exactly 'High', 'Medium', or 'Low'"
        )
        repair_messages = messages + [HumanMessage(content=repair_prompt)]
        try:
            rationale = await _call_structured_with_retry(llm_structured, repair_messages)
            emitter.emit(EventType.VALIDATION_REPAIRED, node="generate_rationales", data={
                "acquirer": acquirer_name
            })
        except Exception as repair_error:
            logger.error("repair_failed", acquirer=acquirer_name, error=str(repair_error))
            raise repair_error
    except Exception as api_error:
        # API-level failure (rate limit, quota, timeout) after all retries exhausted.
        # Do NOT run the repair loop — sending a schema-correction prompt won't fix
        # a billing quota error. Emit NODE_ERROR so the event log is accurate,
        # then re-raise so the stub entry is inserted and the PDF still renders.
        emitter.emit(EventType.NODE_ERROR, node="generate_rationales", data={
            "acquirer": acquirer_name,
            "error": str(api_error)[:300],
            "error_type": "api_error",
        })
        logger.error(
            "api_error_after_retries",
            acquirer=acquirer_name,
            error=str(api_error),
        )
        raise api_error

    emitter.emit(EventType.RATIONALE_GENERATED, node="generate_rationales", data={
        "acquirer": acquirer_name,
        "conviction": rationale.conviction_level,
        "risk_flags": len(rationale.risk_flags),
        "rank": rank,
    })

    # Inject rank, sub_scores, composite, and conviction from the scoring model.
    # conviction_level is enforced here — the LLM writes rationale TEXT calibrated
    # to the level but the label itself is always the Python-computed baseline.
    # Mini tends to downgrade High → Medium without sufficient justification.
    result = rationale.model_dump()
    result["rank"] = rank
    result["sub_scores"] = sub_scores
    result["composite_score"] = candidate.get("composite_score", result.get("composite_score", 0))
    result["conviction_level"] = conviction_baseline

    # Post-generation scan for forbidden EBITDA attribution phrases.
    # Three prompt layers already forbid this but gpt-4o-mini still produces it
    # occasionally. A targeted repair with the exact violation quoted is more
    # reliable than another instruction — the model responds to "here is what you
    # wrote and here is why it is wrong" better than a preemptive warning.
    # Only trigger repair on the generic boilerplate construction — "the/this target's
    # [strong] EBITDA margins complement/align with/support/enhance..." — not on any
    # mention of the target's margins, which is now encouraged with acquirer-specific framing.
    _ebitda_re = re.compile(
        # Catches generic boilerplate: "the/this target's [strong] EBITDA margins [verb]"
        # where the verb is standalone-generic (no acquirer-specific data needed to complete
        # the thought). "will enhance their valuation model AT X.Xx" is acquirer-specific
        # and should NOT be caught — so "will/can" are excluded. "would/should/can" plus
        # "enhance/complement/align/support" ARE included since those constructions appear
        # independent of any specific data point.
        r"(?:the|this)\s+target'?s\s+(?:strong\s+)?ebitda\s+margins?\s+"
        r"(?:complement|align\s+with|support|enhance|are\s+consistent\s+with"
        r"|(?:would|can|should)\s+(?:improve|support|enhance|complement|align"
        r"|benefit|strengthen)|are\s+attractive|are\s+aligned)",
        re.IGNORECASE,
    )
    _scan_text = " ".join(filter(None, [
        result.get("acquirer_overview", ""),
        result.get("strategic_fit_thesis", ""),
        result.get("conviction_rationale", ""),
        result.get("valuation_context", {}).get("note", "") if isinstance(result.get("valuation_context"), dict) else "",
    ] + [rf.get("description", "") for rf in result.get("risk_flags", []) if isinstance(rf, dict)]))

    if _ebitda_re.search(_scan_text):
        logger.warning("forbidden_ebitda_phrase_detected_triggering_repair", acquirer=acquirer_name)
        emitter.emit(EventType.VALIDATION_FAILED, node="generate_rationales", data={
            "acquirer": acquirer_name, "error": "forbidden_ebitda_attribution_detected"
        })
        _repair_msg = HumanMessage(content=(
            "CONTENT VIOLATION — your response must be corrected before delivery.\n\n"
            "One or more sections contain generic EBITDA margin boilerplate matching "
            "'the/this target's [strong] EBITDA margins complement / align with / support / "
            "enhance [acquirer]'s strategy' or a close variant. This is prohibited because "
            "it applies identically to every acquirer on the shortlist — it is not analysis.\n\n"
            "IMPORTANT: You may and should reference the target's strong margins — but make "
            "the claim specific to THIS acquirer. Acceptable framings:\n"
            "- Compare to this acquirer's historical acquisition margins "
            "(acquired_co_ebitda_margin_pct from precedent deals provided above)\n"
            "- For PE sponsors: explain what high entry margins mean for IRR at the expected "
            "entry and exit multiple\n"
            "- For strategic buyers: explain whether the target's profile is above, at, or "
            "below what this acquirer typically acquires, and what that implies for pricing\n\n"
            "Do NOT invent a margin %. 'Strong' is the only descriptor you have.\n\n"
            "CRITICAL — the following replacement phrases are ALSO forbidden and will "
            "trigger another rejection if you use them:\n"
            "  - 'positions them [uniquely/well/to] [leverage/capitalize/...]'\n"
            "  - 'fills a [critical/specific/key] [X] gap'\n"
            "  - 'fills a critical need for'\n"
            "  - 'adds a critical [X] capability'\n"
            "Use actual numbers from the IRR math, margin comparison, or sub-sector count "
            "instead of these connector phrases.\n\n"
            "Produce a corrected full response — all other sections unchanged."
        ))
        try:
            _repaired = await _call_structured_with_retry(llm_structured, messages + [_repair_msg])
            emitter.emit(EventType.VALIDATION_REPAIRED, node="generate_rationales", data={
                "acquirer": acquirer_name
            })
            logger.info("ebitda_repair_succeeded", acquirer=acquirer_name)
            result = _repaired.model_dump()
            result["rank"] = rank
            result["sub_scores"] = sub_scores
            result["composite_score"] = candidate.get("composite_score", result.get("composite_score", 0))
            result["conviction_level"] = conviction_baseline
        except Exception as _ebitda_repair_err:
            logger.error("ebitda_repair_failed", acquirer=acquirer_name, error=str(_ebitda_repair_err))
            # Keep original result — better than a stub

    # Post-generation scan for forbidden filler phrases.
    # Prompt-layer instructions alone do not reliably suppress these from gpt-4o-mini.
    # Scan runs on the current result (after any EBITDA repair) so it catches
    # phrases introduced by the EBITDA repair as well as first-pass output.
    _filler_re = re.compile(
        r"positions\s+(?:\w+\s+){1,3}uniquely\b"
        r"|fills?\s+a\s+(?:critical|specific|key|unique|strategic)\b"
        r"|fills?\s+a\s+gap\b"
        r"|adds?\s+a\s+critical\b",
        re.IGNORECASE,
    )
    _filler_scan = " ".join(filter(None, [
        result.get("acquirer_overview", ""),
        result.get("strategic_fit_thesis", ""),
        result.get("conviction_rationale", ""),
        result.get("valuation_context", {}).get("note", "") if isinstance(result.get("valuation_context"), dict) else "",
    ] + [rf.get("description", "") for rf in result.get("risk_flags", []) if isinstance(rf, dict)]))

    _filler_match = _filler_re.search(_filler_scan)
    if _filler_match:
        _phrase = _filler_match.group(0)
        logger.warning("forbidden_filler_phrase_detected_triggering_repair", acquirer=acquirer_name, phrase=_phrase)
        emitter.emit(EventType.VALIDATION_FAILED, node="generate_rationales", data={
            "acquirer": acquirer_name, "error": "forbidden_filler_phrase_detected"
        })
        _filler_repair_msg = HumanMessage(content=(
            f"CONTENT VIOLATION — your response must be corrected before delivery.\n\n"
            f"The phrase '{_phrase}' (or a close variant) appears in your output. "
            "This is a banned filler phrase that contains zero analysis and reads "
            "identically across every acquirer on the shortlist.\n\n"
            "REQUIRED replacement — write one of these instead, using the specific "
            "numbers from the evidence packet you already received:\n"
            "  FOR 'positions them/[name] uniquely': cite the exact deal count in this "
            "sub-sector and contrast with other acquirers. e.g. '4 of 24 deals in "
            "Healthcare Services concentrated in [sub-sector] make this a direct platform "
            "extension, unlike the other sponsors on this shortlist with 0–1 sector deals.'\n"
            "  FOR 'fills a critical/specific/key gap' or 'fills a gap': name the absent "
            "sub-sector from sub_sector_counts, then explain why this acquirer specifically "
            "needs it. e.g. '[Acquirer] has 0 Home Health deals across 4 Healthcare Services "
            "acquisitions — this target adds the one sub-sector absent from their platform.'\n"
            "  FOR 'adds a critical capability': replace with the actual capability name and "
            "the deal-count evidence that the acquirer currently lacks it.\n\n"
            "FORBIDDEN AS REPLACEMENTS (equally banned — will trigger another rejection):\n"
            "  - 'the/this target's EBITDA margins complement / align with / support / enhance'\n"
            "  - Any other 'fills a [X] gap' or 'fills a critical need for' construction\n"
            "  - 'adds a critical [X] capability'\n\n"
            "Produce a corrected full response — all other sections unchanged."
        ))
        try:
            _filler_repaired = await _call_structured_with_retry(llm_structured, messages + [_filler_repair_msg])
            emitter.emit(EventType.VALIDATION_REPAIRED, node="generate_rationales", data={
                "acquirer": acquirer_name
            })
            logger.info("filler_phrase_repair_succeeded", acquirer=acquirer_name)
            result = _filler_repaired.model_dump()
            result["rank"] = rank
            result["sub_scores"] = sub_scores
            result["composite_score"] = candidate.get("composite_score", result.get("composite_score", 0))
            result["conviction_level"] = conviction_baseline
        except Exception as _filler_repair_err:
            logger.error("filler_phrase_repair_failed", acquirer=acquirer_name, error=str(_filler_repair_err))
            # Keep current result — better than a stub

    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_structured_with_retry(llm_structured, messages: list) -> AcquirerRationale:
    # retry_if_exception_type is intentionally absent: LangChain wraps openai.RateLimitError
    # and openai.APIError before they reach tenacity, so type-matching never fires.
    # Catching all exceptions and retrying is safe — ValidationError is handled
    # by the caller after all retries are exhausted.
    return await llm_structured.ainvoke(messages)


async def node_generate_rationales(state: AgentState, config: RunnableConfig) -> dict:
    """
    Generate all 10 acquirer rationales concurrently.

    Each rationale is an LLM call that receives a fully-assembled evidence packet
    (acquirer profile + precedent deals + market comps). asyncio.gather fires all
    10 simultaneously — total latency is roughly one LLM call, not ten.
    """
    emitter = config["configurable"]["emitter"]
    app_state = config["configurable"]["app_state"]

    emitter.emit(EventType.NODE_STARTED, node="generate_rationales")

    target = state["target"]
    final_names = state["final_acquirer_names"]

    # Build fast lookup: acquirer name → scored candidate dict (has profile + scores)
    scored_map = {c["acquirer_name"]: c for c in state["scored_candidates"]}

    tool_map = {t.name: t for t in app_state.tools}

    # Identify strategic acquirers in this run — used as named exit buyer candidates
    # in PE sponsor rationales so each PE firm cites a specific strategic buyer rather
    # than defaulting to the same generic "regional hospital system" phrase.
    strategic_names_in_run = [
        name for name in final_names
        if _normalize_acquirer_type(scored_map.get(name, {}).get("acquirer_type", "Strategic")) == "Strategic"
    ]

    # Pre-fetch valuation comps once — the result is identical for every acquirer in this
    # run (same target.sector and deal_size_mm). Passing it as cached_comps_json eliminates
    # 9 of 10 redundant DataFrame filter calls inside _generate_one.
    comps_tool = tool_map.get("get_valuation_comps")
    cached_comps_json: str | None = None
    if comps_tool:
        try:
            size = target.deal_size_mm
            comps_result = comps_tool.invoke({
                "sectors": [target.sector],
                "deal_size_min": size * 0.4,
                "deal_size_max": size * 2.5,
            })
            cached_comps_json = comps_result if isinstance(comps_result, str) else json.dumps(comps_result)
        except Exception as e:
            logger.warning("valuation_comps_prefetch_failed", error=str(e))

    # Semaphore(10): run all 10 calls in a single batch. gpt-4o-mini's TPM limit is
    # high enough that 10 concurrent calls do not trigger rate-limit backoff at Tier 1.
    # Dropping to 5 added ~8-10s by forcing two sequential batches — not worth it.
    sem = asyncio.Semaphore(10)

    async def _throttled(name: str, rank: int, candidate: dict):
        async with sem:
            is_pe = _normalize_acquirer_type(candidate.get("acquirer_type", "Strategic")) == "Financial Sponsor"
            co_strategics = strategic_names_in_run if is_pe else []
            return await _generate_one(name, rank, candidate, target, tool_map, app_state, emitter, co_strategics, cached_comps_json)

    results = await asyncio.gather(
        *[_throttled(name, rank, scored_map.get(name, {})) for rank, name in enumerate(final_names, 1)],
        return_exceptions=True,
    )

    rationales = []
    errors = list(state.get("errors", []))

    for name, result in zip(final_names, results):
        if isinstance(result, Exception):
            logger.error("rationale_generation_failed", acquirer=name, error=str(result))
            errors.append(f"rationale_failed:{name}: {result}")
            # Include a stub so PDF generation can still produce all 10 pages
            rationales.append({
                "acquirer_name": name,
                "error": str(result),
                "acquirer_type": "Strategic",
                "composite_score": scored_map.get(name, {}).get("composite_score", 0),
                "rank": final_names.index(name) + 1,
            })
        else:
            rationales.append(result)

    succeeded = sum(1 for r in rationales if "error" not in r)
    failed = len(rationales) - succeeded

    emitter.emit(EventType.NODE_COMPLETED, node="generate_rationales", data={
        "rationales_generated": succeeded,
        "rationales_failed": failed,
    })

    logger.info("rationales_complete", succeeded=succeeded, failed=failed)

    return {"rationales": rationales, "errors": errors}
