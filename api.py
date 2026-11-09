"""
FastAPI backend for the Email/SMS Spam Detection system.

Endpoints
---------
GET  /health   – health-check
POST /predict  – classify a text as spam or ham

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predict import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(
    title="Email Spam Detection API",
    description="Classify email / SMS text as spam or ham.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Email / SMS body text")


class PredictResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the MailGuard web frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/style.css", include_in_schema=False)
async def serve_style():
    """Serve frontend stylesheet."""
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/script.js", include_in_schema=False)
async def serve_script():
    """Serve frontend JavaScript."""
    return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Simple health-check."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Classify input text as spam or ham."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    try:
        result = predict(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal server error")

    return PredictResponse(**result)
