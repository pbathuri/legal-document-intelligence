from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    page: int | None = Field(None, description="1-based page number if known")
    quote: str = Field("", description="Short verbatim excerpt from the document")


class InstrumentFact(BaseModel):
    doc_id: str = ""
    doc_label: str = ""
    doc_type: str = Field(
        default="unknown",
        description="sale deed, gift deed, partition, mutation extract, EC, RoR, RTC, tax receipt, layout approval, other",
    )
    execution_date: str | None = None
    registration_date: str | None = None
    seller_names: list[str] = Field(default_factory=list)
    buyer_names: list[str] = Field(default_factory=list)
    parcel_ids: list[str] = Field(
        default_factory=list, description="survey/plot/khata/patta/ULPIN/CTS references"
    )
    locality: str | None = None
    consideration_amount: str | None = None
    mentions_dispute: bool = False
    dispute_details: str | None = None
    mentions_encumbrance: bool = False
    encumbrance_details: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    def model_dump_json_safe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
