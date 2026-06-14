from dataclasses import dataclass, field
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool


@dataclass
class AppState:
    """
    Holds everything loaded at startup.
    One instance lives for the lifetime of the server.
    Injected into route handlers via FastAPI Depends().
    """
    df: pd.DataFrame = field(default=None)
    acquirer_profiles: dict[str, dict] = field(default_factory=dict)
    llm: ChatOpenAI = field(default=None)
    llm_fast: ChatOpenAI = field(default=None)  # gpt-4o-mini — used for rerank (ranking is simpler than synthesis)
    tools: list[BaseTool] = field(default_factory=list)
