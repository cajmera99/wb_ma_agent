from pydantic import BaseModel, Field


class TargetProfile(BaseModel):
    """What the banker enters for the company they want to sell."""
    sector: str = Field(default="Healthcare Services")
    deal_size_mm: float = Field(default=200.0, description="Enterprise value in $M")
    geography: str = Field(default="Regional")
    ownership: str = Field(default="Private", description="Private / PE-Backed / Public")
    profile_description: str = Field(
        default="",
        description="Free-text description of the target — margins, growth, competitive position, etc.",
    )
