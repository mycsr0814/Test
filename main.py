"""
DeliveryLearning 음성 분석 API (FastAPI).

실행 예 (프로젝트 Server/DeliveryLearning 디렉터리에서):
  uvicorn main:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from delivery_learning.config import settings
from delivery_learning.voice_job import run_feedback_voice_analysis


class VoiceAnalyzeRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    feedback_id: int = Field(..., ge=1)


def _check_internal_secret(x_secret: str | None) -> None:
    expected = settings.INTERNAL_API_SECRET
    if not expected:
        return
    if not x_secret or x_secret != expected:
        raise HTTPException(status_code=401, detail="유효하지 않은 내부 API 시크릿입니다.")


app = FastAPI(title="DeliveryLearning", version="1.0.0")

_cors = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_origins = _cors.split(",") if _cors and _cors != "*" else ["*"]
_use_creds = "*" not in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_use_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/voice/analyze")
def voice_analyze(
    body: VoiceAnalyzeRequest,
    x_delivery_learning_secret: str | None = Header(default=None, alias="X-Delivery-Learning-Secret"),
):
    _check_internal_secret(x_delivery_learning_secret)
    try:
        return run_feedback_voice_analysis(body.user_id, body.feedback_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"오디오 파일 오류: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {e!s}") from e
