from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"

    csv_path: Path = Path("data/ma_transactions_500.csv")

    # Scoring weights (must sum to 1.0)
    weight_sector: float = 0.35
    weight_deal_size: float = 0.20
    weight_rationale: float = 0.20
    weight_recency: float = 0.10
    weight_outcome: float = 0.10
    weight_ownership: float = 0.05

    # How many candidates to send to LLM re-ranker before cutting to 10
    pre_rerank_count: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
