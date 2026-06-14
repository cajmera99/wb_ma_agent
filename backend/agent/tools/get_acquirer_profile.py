import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetAcquirerProfileInput(BaseModel):
    acquirer_name: str = Field(description="Exact acquirer name as it appears in the dataset")


def make_get_acquirer_profile(profiles: dict[str, dict]):
    """
    Factory that closes over the pre-computed acquirer profiles.
    Returns a LangChain tool the agent can call.
    """

    @tool(args_schema=GetAcquirerProfileInput)
    def get_acquirer_profile(acquirer_name: str) -> str:
        """
        Retrieve the full M&A profile for a specific acquirer.
        Returns deal counts, sector breakdown, median deal size,
        valuation multiples, top rationale tags, and recency signals.
        Use this before writing any acquirer rationale.
        """
        profile = profiles.get(acquirer_name)

        if not profile:
            # Fuzzy fallback: try case-insensitive partial match
            lower = acquirer_name.lower()
            match = next(
                (p for name, p in profiles.items() if lower in name.lower()),
                None,
            )
            if match:
                profile = match
            else:
                return json.dumps({
                    "error": f"No profile found for '{acquirer_name}'.",
                    "available_count": len(profiles),
                })

        return json.dumps(profile, default=str)

    return get_acquirer_profile
