import json
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetAcquirerPrecedentDealsInput(BaseModel):
    acquirer_name: str = Field(description="Exact acquirer name")
    sectors: list[str] | None = Field(
        default=None,
        description="Filter to specific sectors. Leave empty for all sectors."
    )
    max_results: int = Field(default=5, description="Number of most relevant deals to return")


def make_get_acquirer_precedent_deals(df: pd.DataFrame):
    """
    Factory that closes over the pre-loaded DataFrame.
    Returns a LangChain tool the agent can call.
    """

    @tool(args_schema=GetAcquirerPrecedentDealsInput)
    def get_acquirer_precedent_deals(
        acquirer_name: str,
        sectors: list[str] | None = None,
        max_results: int = 5,
    ) -> str:
        """
        Retrieve specific historical deals for one acquirer.
        These are the precedent transactions to cite in the rationale.
        Prioritises closed deals, sorted by deal size descending.
        """
        mask = df["acquirer"] == acquirer_name
        if sectors:
            mask &= df["sector"].isin(sectors)

        results = (
            df[mask]
            .sort_values(["outcome", "deal_size_mm"], ascending=[True, False])
            .head(max_results)
        )

        if results.empty:
            return json.dumps({
                "acquirer": acquirer_name,
                "count": 0,
                "deals": [],
                "note": "No deals found for this acquirer with the given filters.",
            })

        cols = [
            "transaction_id", "target_company", "sector", "sub_sector",
            "deal_size_mm", "deal_type", "deal_year", "deal_quarter",
            "geography", "ev_ebitda_multiple", "ev_revenue_multiple",
            "ebitda_margin_pct", "revenue_growth_pct", "outcome",
            "strategic_rationale_tags", "target_ownership_pre", "days_to_close",
        ]
        deals = results[cols].fillna("N/A").to_dict(orient="records")

        return json.dumps({
            "acquirer": acquirer_name,
            "count": len(deals),
            "deals": deals,
        }, default=str)

    return get_acquirer_precedent_deals
