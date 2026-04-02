# System Architecture Overview

This implementation follows a microservices layout for Feature 1:

- Employee Portal clients:
  - Web: React/Next.js (external client)
  - Mobile: React Native (external client)
- Backend microservices in this repo:
  - Ingestion + OCR service (`services/ingestion`)
  - Validation service (`services/validation`)

## Request Flow

1. Portal uploads receipt image + `claim_date` to the ingestion API.
2. Ingestion service runs image quality checks:
   - Blur detection (OpenCV Laplacian variance)
   - Glare detection (high-brightness ratio)
   - Orientation hint (EXIF orientation metadata)
3. OCR extraction runs using Donut (Document Understanding Transformer).
4. Extracted fields are normalized:
   - Merchant Name
   - Date
   - Amount
   - Currency
   - Line Items
   - Alcohol detection from line-item text
5. Ingestion service calls validation service.
6. Validation service compares extracted date vs claim date and returns pass/fail.

## Model Strategy

- Default engine: Donut (`naver-clova-ix/donut-base-finetuned-cord-v2`)
- Optional strategy hook: LayoutLMv3 adapter exists and can be wired with a fine-tuned checkpoint + OCR box pipeline.

## AWS Rekognition Option

- Set `USE_AWS_REKOGNITION=true` to use Rekognition text-detection checks as an optional quality gate.
