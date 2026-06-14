import json
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SearchTransactionsInput(BaseModel):
    sectors: list[str] = Field(
        description="Sectors to search. Pass multiple to broaden: e.g. ['Healthcare Services', 'Behavioral Health']"
    )
    deal_size_min: float = Field(description="Minimum deal size in $M")
    deal_size_max: float = Field(description="Maximum deal size in $M")
    outcomes: list[str] = Field(
        default=["Closed"],
        description="Filter by outcome. Default is Closed only."
    )
    max_results: int = Field(default=30, description="Cap on rows returned")


def make_search_transactions(df: pd.DataFrame):
    """
    Factory that closes over the pre-loaded DataFrame.
    Returns a LangChain tool the agent can call.
    """

    @tool(args_schema=SearchTransactionsInput)
    def search_transactions(
        sectors: list[str],
        deal_size_min: float,
        deal_size_max: float,
        outcomes: list[str] = ["Closed"],
        max_results: int = 30,
    ) -> str:
        """
        Search historical M&A transactions by sector and deal size.
        Use this to find relevant precedent deals for a target profile.
        Broaden the sector list if initial results are too few.
        """
        mask = (
            df["sector"].isin(sectors)
            & df["deal_size_mm"].between(deal_size_min, deal_size_max)
            & df["outcome"].isin(outcomes)
        )
        results = df[mask].head(max_results)

        if results.empty:
            return json.dumps({"count": 0, "deals": [], "note": "No matching transactions found."})

        cols = [
            "transaction_id", "acquirer", "sector", "sub_sector",
            "deal_size_mm", "deal_type", "deal_year", "geography",
            "ev_ebitda_multiple", "ev_revenue_multiple",
            "ebitda_margin_pct", "outcome", "strategic_rationale_tags",
            "acquirer_type", "target_ownership_pre",
        ]
        deals = results[cols].fillna("N/A").to_dict(orient="records")

        return json.dumps({
            "count": len(deals),
            "sectors_searched": sectors,
            "size_range": f"${deal_size_min}M–${deal_size_max}M",
            "deals": deals,
        }, default=str)

    return search_transactions
