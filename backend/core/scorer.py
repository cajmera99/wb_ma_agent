import numpy as np
import structlog
from backend.models.target import TargetProfile
from backend.core.config import settings

logger = structlog.get_logger(__name__)

# These tags signal the kind of thesis that fits a mid-market regional target
HIGH_RELEVANCE_TAGS = {
    "Platform Build",
    "Geographic Expansion",
    "High Growth",
    "Margin Improvement",
    "Vertical Integration",
    "Bolt-on Acquisition",
    "Cost Synergies",
}

# Full weight sectors vs partially adjacent
PRIMARY_SECTOR = "Healthcare Services"
ADJACENT_SECTORS = {"Behavioral Health", "Physician Groups", "Home Health/Hospice"}
SECONDARY_SECTORS = {"Health IT", "Dental", "Revenue Cycle"}


def _score_sector(profile: dict, target: TargetProfile) -> float:
    """
    How well does this acquirer's deal history align with the target's sector?

    Scoring:
    - Exact match to target sector → 1.0 weight
    - Healthcare-family adjacency (bidirectional) → 0.7 weight
    - Secondary healthcare overlap → 0.3 weight
    - No match → 0.0

    When target.sector is outside the healthcare family (e.g. Sports, Technology),
    only exact matches score; all others get 0 — which is the correct signal.
    The LLM is then responsible for explaining cross-sector thesis in rationales.
    """
    total = profile["total_deals"]
    if total == 0:
        return 0.0

    sector_counts = profile["sector_counts"]
    target_sector = target.sector
    weighted = 0.0

    for sector, count in sector_counts.items():
        if sector == target_sector:
            weighted += count * 1.0
        elif target_sector == PRIMARY_SECTOR and sector in ADJACENT_SECTORS:
            weighted += count * 0.7
        elif target_sector == PRIMARY_SECTOR and sector in SECONDARY_SECTORS:
            weighted += count * 0.3
        elif target_sector in ADJACENT_SECTORS and sector == PRIMARY_SECTOR:
            weighted += count * 0.7
        elif target_sector in ADJACENT_SECTORS and sector in ADJACENT_SECTORS:
            weighted += count * 0.5

    return min(1.0, weighted / total)


def _score_deal_size(profile: dict, target: TargetProfile) -> float:
    """
    Gaussian decay centred on the target deal size.
    An acquirer whose median deal is exactly $200M scores 1.0.
    One whose median is $800M scores much lower.
    """
    median = profile.get("median_deal_size_mm")
    if median is None:
        return 0.0

    sigma = target.deal_size_mm * 0.6  # wide enough to reward ±60% range
    return float(np.exp(-0.5 * ((median - target.deal_size_mm) / sigma) ** 2))


def _score_rationale_tags(profile: dict) -> float:
    """
    What fraction of this acquirer's top rationale tags are relevant
    for a mid-market regional healthcare target?
    """
    tags = profile.get("top_rationale_tags", {})
    if not tags:
        return 0.0

    total_tag_weight = sum(tags.values())
    relevant_weight = sum(
        count for tag, count in tags.items() if tag in HIGH_RELEVANCE_TAGS
    )
    return relevant_weight / total_tag_weight if total_tag_weight else 0.0


def _score_recency(profile: dict) -> float:
    """
    Acquirers who have been active recently signal ongoing appetite.
    Combines how recent their last deal was with how many recent deals they have.
    """
    most_recent = profile.get("most_recent_year", 2015)
    recent_count = profile.get("recent_deal_count", 0)

    # Decay: lose 0.15 per year since last deal (so 2024=1.0, 2021=0.55, 2018=0.1)
    years_stale = max(0, 2024 - most_recent)
    recency_decay = max(0.0, 1.0 - years_stale * 0.15)

    # Active volume: 3+ recent deals = full score
    activity_score = min(1.0, recent_count / 3)

    return 0.5 * recency_decay + 0.5 * activity_score


def _score_outcome_quality(profile: dict) -> float:
    """
    Acquirers who actually close deals score higher than those who
    frequently withdraw or get terminated.
    """
    total = profile["total_deals"]
    if total == 0:
        return 0.0
    return profile["closed_deals"] / total


def _score_ownership_match(profile: dict, target: TargetProfile) -> float:
    """
    Does this acquirer typically buy companies with the same ownership type
    as our target? (Private / PE-Backed / Public)
    """
    ownership_counts = profile.get("target_ownership_counts", {})
    total = profile["total_deals"]
    if total == 0:
        return 0.0

    # Private and PE-Backed are both "private-side" targets
    if target.ownership in ("Private", "PE-Backed"):
        match_count = ownership_counts.get("Private", 0) + ownership_counts.get("PE-Backed", 0)
    else:
        match_count = ownership_counts.get(target.ownership, 0)

    return match_count / total


def score_acquirer(profile: dict, target: TargetProfile) -> dict:
    """
    Compute weighted composite score for one acquirer against one target.
    Returns the composite score (0-100) plus each sub-score for explainability.
    """
    s = settings

    sub_scores = {
        "sector":    _score_sector(profile, target),
        "deal_size": _score_deal_size(profile, target),
        "rationale": _score_rationale_tags(profile),
        "recency":   _score_recency(profile),
        "outcome":   _score_outcome_quality(profile),
        "ownership": _score_ownership_match(profile, target),
    }

    composite = (
        sub_scores["sector"]    * s.weight_sector    +
        sub_scores["deal_size"] * s.weight_deal_size +
        sub_scores["rationale"] * s.weight_rationale +
        sub_scores["recency"]   * s.weight_recency   +
        sub_scores["outcome"]   * s.weight_outcome   +
        sub_scores["ownership"] * s.weight_ownership
    )

    return {
        **profile,
        "composite_score": round(composite * 100, 1),
        "sub_scores": {k: round(v * 100, 1) for k, v in sub_scores.items()},
    }


def rank_acquirers(
    profiles: dict[str, dict],
    target: TargetProfile,
    top_n: int | None = None,
) -> list[dict]:
    """
    Score every acquirer and return them sorted best-first.
    top_n caps the result (used to grab the pre-rerank pool).
    """
    scored = [score_acquirer(p, target) for p in profiles.values()]
    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    logger.info(
        "acquirers_ranked",
        total=len(scored),
        top_score=scored[0]["composite_score"] if scored else None,
        top_acquirer=scored[0]["acquirer_name"] if scored else None,
    )

    return scored[:top_n] if top_n else scored
