from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    is_blurry: bool
    blur_score: float
    has_glare: bool
    glare_ratio: float
    needs_rotation: bool
    orientation_hint: str | None = None
    warnings: list[str] = Field(default_factory=list)


class OCRFields(BaseModel):
    merchant_name: str | None = None
    date: str | None = None
    amount: float | None = None
    currency: str | None = None
    line_items: list[str] = Field(default_factory=list)
    has_alcohol_item: bool = False
    raw: dict = Field(default_factory=dict)


class ValidationResult(BaseModel):
    is_valid: bool
    reason: str
    extracted_date: str | None = None
    claim_date: str | None = None


class IngestResponse(BaseModel):
    quality: QualityReport
    extracted: OCRFields
    validation: ValidationResult
