from __future__ import annotations

from datetime import date

from dateutil import parser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Expense Validation Service", version="0.1.0")


class DateValidationRequest(BaseModel):
    extracted_date: str | None = None
    claim_date: str
    tolerance_days: int = Field(default=0, ge=0, le=30)


class DateValidationResponse(BaseModel):
    is_valid: bool
    reason: str
    extracted_date: str | None = None
    claim_date: str | None = None


def _to_date(raw: str) -> date:
    return parser.parse(raw, dayfirst=False, fuzzy=True).date()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "validation"}


@app.post("/validate-date", response_model=DateValidationResponse)
def validate_date(payload: DateValidationRequest) -> DateValidationResponse:
    if not payload.extracted_date:
        return DateValidationResponse(
            is_valid=False,
            reason="No date extracted from receipt",
            extracted_date=None,
            claim_date=payload.claim_date,
        )

    try:
        extracted = _to_date(payload.extracted_date)
        claim = _to_date(payload.claim_date)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid date format: {exc}"
        ) from exc

    day_delta = abs((claim - extracted).days)
    is_valid = day_delta <= payload.tolerance_days

    reason = (
        "Date matches claim"
        if is_valid
        else f"Date mismatch: delta={day_delta} days exceeds tolerance={payload.tolerance_days}"
    )

    return DateValidationResponse(
        is_valid=is_valid,
        reason=reason,
        extracted_date=extracted.isoformat(),
        claim_date=claim.isoformat(),
    )
