# Expense Audit - Feature 1 Starter

This workspace implements the Feature 1 backend as two Python microservices:

- `services/ingestion`: image quality checks + OCR field extraction
- `services/validation`: OCR date vs claim date validation

## 1) Create and Activate Conda Environment

```powershell
conda env create -f environment.yml
conda activate expense-audit
```

If you update dependencies later:

```powershell
conda env update -f environment.yml --prune
```

## 2) Run Services

Terminal 1:

```powershell
conda activate expense-audit
uvicorn services.validation.main:app --host 0.0.0.0 --port 8001 --reload
```

Terminal 2:

```powershell
conda activate expense-audit
uvicorn services.ingestion.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Test API

```powershell
curl -X POST "http://127.0.0.1:8000/ingest" ^
  -F "claim_date=2026-03-20" ^
  -F "file=@C:\path\to\receipt.jpg"
```

## 4) Run Frontend (Vite + Tailwind)

The frontend lives in `frontend/` and is plain JavaScript (no TypeScript).

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (typically `http://127.0.0.1:5173`).

Dev proxy routes are already configured:

- `/api-ingestion/*` -> `http://127.0.0.1:8000/*`
- `/api-validation/*` -> `http://127.0.0.1:8001/*`

## Notes

- Default OCR engine is `donut` using model `naver-clova-ix/donut-base-finetuned-cord-v2`.
- Set `OCR_ENGINE=layoutlmv3` only after wiring a fine-tuned LayoutLMv3 extraction model.
- Local image quality checks use OpenCV blur/glare + EXIF orientation.
- Optional AWS Rekognition checks can be enabled with `USE_AWS_REKOGNITION=true`.
