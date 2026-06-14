import pandas as pd
from langchain_core.tools import BaseTool

from backend.agent.tools.search_transactions import make_search_transactions
from backend.agent.tools.get_acquirer_profile import make_get_acquirer_profile
from backend.agent.tools.get_acquirer_precedent_deals import make_get_acquirer_precedent_deals
from backend.agent.tools.get_valuation_comps import make_get_valuation_comps


def build_tools(df: pd.DataFrame, profiles: dict[str, dict]) -> list[BaseTool]:
    """
    Instantiate all four agent tools, closing over the pre-loaded data.
    Called once at startup; the tool list is reused across all requests.
    """
    return [
        make_search_transactions(df),
        make_get_acquirer_profile(profiles),
        make_get_acquirer_precedent_deals(df),
        make_get_valuation_comps(df),
    ]
