import json
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetValuationCompsInput(BaseModel):
    sectors: list[str] = Field(description="Sectors to pull comps from")
    deal_size_min: float = Field(description="Minimum deal size in $M")
    deal_size_max: float = Field(description="Maximum deal size in $M")
    acquirer_type: str | None = Field(
        default=None,
        description="Filter to 'Strategic' or 'Financial Sponsor'. Leave empty for both."
    )


def make_get_valuation_comps(df: pd.DataFrame):
    """
    Factory that closes over the pre-loaded DataFrame.
    Returns a LangChain tool the agent can call.
    """

    @tool(args_schema=GetValuationCompsInput)
    def get_valuation_comps(
        sectors: list[str],
        deal_size_min: float,
        deal_size_max: float,
        acquirer_type: str | None = None,
    ) -> str:
        """
        Retrieve EV/EBITDA and EV/Revenue valuation comps from closed deals.
        Use this to populate the Valuation Context section of each rationale.
        Returns median, 25th, and 75th percentile multiples.
        """
        mask = (
            df["sector"].isin(sectors)
            & df["deal_size_mm"].between(deal_size_min, deal_size_max)
            & (df["outcome"] == "Closed")
        )
        if acquirer_type:
            mask &= df["acquirer_type"] == acquirer_type

        comps = df[mask]

        if comps.empty:
            return json.dumps({
                "count": 0,
                "note": "No closed comps found in this range. Consider broadening sectors or size range.",
            })

        ev_ebitda = comps["ev_ebitda_multiple"].dropna()
        ev_revenue = comps["ev_revenue_multiple"].dropna()

        def stats(series: pd.Series) -> dict:
            if series.empty:
                return None
            return {
                "median": round(series.median(), 1),
                "p25":    round(series.quantile(0.25), 1),
                "p75":    round(series.quantile(0.75), 1),
                "min":    round(series.min(), 1),
                "max":    round(series.max(), 1),
                "count":  len(series),
            }

        return json.dumps({
            "sectors": sectors,
            "size_range": f"${deal_size_min}M–${deal_size_max}M",
            "deal_count": len(comps),
            "ev_ebitda_multiple": stats(ev_ebitda),
            "ev_revenue_multiple": stats(ev_revenue),
            "median_ebitda_margin_pct": round(comps["ebitda_margin_pct"].median(), 1)
                if not comps["ebitda_margin_pct"].dropna().empty else None,
        }, default=str)

    return get_valuation_comps
