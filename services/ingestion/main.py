from __future__ import annotations

import os

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .ocr import build_ocr_extractor
from .preprocess import local_quality_checks
from .schemas import IngestResponse


app = FastAPI(title="Expense Ingestion Service", version="0.1.0")
ocr_extractor = build_ocr_extractor()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ingestion"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_receipt(
    claim_date: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload")

    quality = (
        local_quality_checks(image_bytes)
    )

    if quality["is_blurry"] or quality["has_glare"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Image quality below threshold.",
                "quality": quality,
            },
        )

    try:
        extracted = ocr_extractor.extract(image_bytes)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"OCR extraction failed: {exc}"
        ) from exc

    validation_url = os.getenv("VALIDATION_URL", "http://127.0.0.1:8001/validate-date")
    payload = {
        "claim_date": claim_date,
        "extracted_date": extracted.get("date"),
    }

    try:
        response = requests.post(validation_url, json=payload, timeout=10)
        response.raise_for_status()
        validation = response.json()
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Validation service unavailable: {exc}",
        ) from exc

    return {
        "quality": quality,
        "extracted": extracted,
        "validation": validation,
    }
