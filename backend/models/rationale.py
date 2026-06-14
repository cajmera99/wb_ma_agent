from pydantic import BaseModel, Field
from typing import Literal


class PrecedentDeal(BaseModel):
    """A specific deal from the CSV cited as evidence."""
    transaction_id: str
    target_company: str
    deal_size_mm: float
    sector: str
    deal_type: str
    deal_year: int
    ev_ebitda_multiple: float | None
    ev_revenue_multiple: float | None
    outcome: str
    rationale_tags: list[str]


class ValuationContext(BaseModel):
    """Market comp stats derived from the CSV."""
    median_ev_ebitda: float | None
    median_ev_revenue: float | None
    deal_count_in_range: int
    note: str


class RiskFlag(BaseModel):
    risk_type: str = Field(description="e.g. Antitrust, Integration Complexity, Financing Capacity")
    description: str
    severity: Literal["High", "Medium", "Low"]


class AcquirerRationale(BaseModel):
    """The full one-page rationale for a single acquirer. LLM must produce this."""
    acquirer_name: str
    acquirer_type: Literal["Strategic", "Financial Sponsor"]
    composite_score: float = Field(description="0-100 score from the scoring model")

    # Section 1
    acquirer_overview: str = Field(description="Who they are, size, strategic priorities, M&A history")

    # Section 2
    strategic_fit_thesis: str = Field(description="Why this target specifically — grounded in data")

    # Section 3
    precedent_deals: list[PrecedentDeal] = Field(min_length=1, description="Relevant deals from the CSV")

    # Section 4
    valuation_context: ValuationContext

    # Section 5
    risk_flags: list[RiskFlag] = Field(min_length=2, max_length=3, description="Exactly 2 risk flags; 3 allowed as buffer so PDF renderer picks the top 2 by severity")

    # Section 6
    conviction_level: Literal["High", "Medium", "Low"]
    conviction_rationale: str = Field(description="1-2 sentences tied to specific data signals")
