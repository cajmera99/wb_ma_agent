import pandas as pd
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Columns the rest of the system depends on.
# If any are missing at startup we fail fast with a clear message
# rather than crashing silently mid-request.
REQUIRED_COLUMNS = {
    "transaction_id", "acquirer", "sector", "sub_sector",
    "deal_year", "deal_type", "geography", "deal_size_mm",
    "ev_ebitda_multiple", "ev_revenue_multiple", "ebitda_margin_pct",
    "revenue_growth_pct", "outcome", "strategic_rationale_tags",
    "acquirer_type", "target_ownership_pre",
}

ADJACENT_SECTORS = {
    "Healthcare Services",
    "Behavioral Health",
    "Physician Groups",
    "Home Health/Hospice",
}


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the M&A transactions CSV at startup.
    Called once; result is stored in AppState and reused across all requests.

    Validates required columns on load so schema changes surface immediately
    at startup rather than as cryptic runtime errors mid-request.
    """
    logger.info("loading_csv", path=str(csv_path))

    df = pd.read_csv(csv_path)

    # Fail fast if the schema changed and we're missing something we depend on
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            "Update REQUIRED_COLUMNS in loader.py if the schema has changed intentionally."
        )

    # Warn about any ADJACENT_SECTORS that don't appear in the data —
    # useful signal if the sector taxonomy changes in a future dataset version
    actual_sectors = set(df["sector"].dropna().unique())
    unknown = ADJACENT_SECTORS - actual_sectors
    if unknown:
        logger.warning(
            "adjacent_sectors_not_in_data",
            missing_sectors=sorted(unknown),
            note="Scoring weights will apply to zero deals for these sectors.",
        )

    # Normalize text columns so comparisons don't break on casing/whitespace
    text_cols = [
        "sector", "sub_sector", "acquirer", "deal_type",
        "geography", "outcome", "acquirer_type", "target_ownership_pre",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Parse rationale tags into real lists ("Tag A|Tag B" → ["Tag A", "Tag B"])
    df["rationale_tag_list"] = (
        df["strategic_rationale_tags"]
        .fillna("")
        .apply(lambda x: [t.strip() for t in x.split("|") if t.strip()])
    )

    # Pre-flag adjacent-sector deals (used by profiler and scoring model)
    df["is_adjacent_sector"] = df["sector"].isin(ADJACENT_SECTORS)

    logger.info(
        "csv_loaded",
        total_rows=len(df),
        sectors=sorted(actual_sectors),
        acquirers=df["acquirer"].nunique(),
    )

    return df
