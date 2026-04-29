"""
user_id + feedback_id 로 DB에서 오디오 키를 읽고, S3에서 내려받아 STT·학습 모델 분석 후 Audio_analysis 행을 갱신합니다.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from delivery_learning.config import settings
from delivery_learning.ml_models import TrainedModelBundle
from delivery_learning.predict_models import transcribe_then_label_with_bundle


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_model_dir() -> str:
    p = Path(settings.MODEL_DIR)
    if p.is_absolute():
        return str(p)
    return str((_package_root() / p).resolve())


def _require_voice_api_config() -> None:
    if not (settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY and settings.S3_BUCKET_NAME):
        raise RuntimeError("S3 다운로드를 위해 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME 가 필요합니다.")


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


def _download_audio_to_temp(s3_key: str) -> str:
    ext = Path(s3_key).suffix.lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.close()
    _s3_client().download_file(settings.S3_BUCKET_NAME, s3_key, tmp.name)
    return tmp.name


def _feedback_owned_by_user(conn: Connection, feedback_id: int, user_id: int) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1 AS ok
            FROM [Feedback]
            WHERE id = :feedback_id AND user_id = :user_id
            """
        ),
        {"feedback_id": feedback_id, "user_id": user_id},
    ).fetchone()
    return row is not None


def _get_feedback_owner_user_id(conn: Connection, feedback_id: int) -> int | None:
    row = conn.execute(
        text(
            """
            SELECT user_id
            FROM [Feedback]
            WHERE id = :feedback_id
            """
        ),
        {"feedback_id": feedback_id},
    ).fetchone()
    if row is None:
        return None
    # SQLAlchemy row: access by column name
    owner = row.user_id if hasattr(row, "user_id") else row[0]
    if owner is None:
        return None
    return int(owner)


def _list_audio_rows(conn: Connection, feedback_id: int) -> list[Any]:
    return list(
        conn.execute(
            text(
                """
                SELECT id, slide_index, audio_key
                FROM [Audio_analysis]
                WHERE feedback_id = :feedback_id
                  AND audio_key IS NOT NULL
                  AND LTRIM(RTRIM(audio_key)) <> ''
                  AND (
                        transcript_text IS NULL
                        OR LTRIM(RTRIM(transcript_text)) = ''
                      )
                ORDER BY slide_index
                """
            ),
            {"feedback_id": feedback_id},
        ).fetchall()
    )


def _update_audio_analysis_row(
    conn: Connection,
    audio_analysis_id: int,
    payload: dict[str, Any],
    analyzer_version: str,
) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.execute(
        text(
            """
            UPDATE [Audio_analysis]
            SET
                transcript_text = :transcript_text,
                duration_sec = :duration_sec,
                word_count = :word_count,
                wpm = :wpm,
                speed_label = :speed_label,
                filler_count = :filler_count,
                filler_ratio = :filler_ratio,
                filler_label = :filler_label,
                filler_tokens_json = :filler_tokens_json,
                analyzer_version = :analyzer_version,
                whisper_model = :whisper_model,
                updated_at = :updated_at
            WHERE id = :id
            """
        ),
        {
            "id": audio_analysis_id,
            "transcript_text": payload["transcript_text"],
            "duration_sec": payload["duration_sec"],
            "word_count": payload["word_count"],
            "wpm": payload["wpm"],
            "speed_label": payload["speed_label"],
            "filler_count": payload["filler_count"],
            "filler_ratio": payload["filler_ratio"],
            "filler_label": payload["filler_label"],
            "filler_tokens_json": payload["filler_tokens_json"],
            "analyzer_version": analyzer_version,
            "whisper_model": payload["whisper_model"],
            "updated_at": now,
        },
    )


def run_feedback_voice_analysis(user_id: int | None, feedback_id: int) -> dict[str, Any]:
    """
    Raises:
        PermissionError: 본인 feedback 이 아님
        ValueError: 처리할 오디오 행 없음
        RuntimeError: 환경 설정 누락
    """
    _require_voice_api_config()
    model_dir = resolve_model_dir()
    if not Path(model_dir).is_dir():
        raise RuntimeError(f"MODEL_DIR 경로가 없습니다: {model_dir}")

    bundle = TrainedModelBundle.load(model_dir)
    engine = create_engine(settings.db_connection_string, fast_executemany=True)

    with engine.begin() as conn:
        # 다른 서버에서 user_id를 몰라도 feedback_id만으로 호출할 수 있게,
        # 필요하면 DB에서 owner user_id를 조회합니다.
        effective_user_id = user_id
        if effective_user_id is None:
            owner = _get_feedback_owner_user_id(conn, feedback_id)
            if owner is None:
                raise ValueError(f"feedback_id={feedback_id} 에 해당하는 데이터를 찾을 수 없습니다.")
            effective_user_id = owner

        if not _feedback_owned_by_user(conn, feedback_id, effective_user_id):
            raise PermissionError("해당 피드백에 대한 접근 권한이 없습니다.")

        rows = _list_audio_rows(conn, feedback_id)
        if not rows:
            # 이미 transcript_text가 채워진 경우도 많으므로, "없음"은 조용히 통과합니다.
            return {
                "feedback_id": feedback_id,
                "user_id": effective_user_id,
                "slides_processed": 0,
                "slides": [],
            }

        slide_results: list[dict[str, Any]] = []

        for row in rows:
            audio_analysis_id = int(row.id)
            slide_index = int(row.slide_index)
            s3_key = str(row.audio_key).strip()
            local_path: str | None = None
            try:
                local_path = _download_audio_to_temp(s3_key)
                payload = transcribe_then_label_with_bundle(
                    local_path,
                    bundle=bundle,
                    openai_api_key=settings.OPENAI_API_KEY,
                )
                _update_audio_analysis_row(
                    conn,
                    audio_analysis_id,
                    payload,
                    settings.ANALYZER_VERSION,
                )
                slide_results.append(
                    {
                        "audio_analysis_id": audio_analysis_id,
                        "slide_index": slide_index,
                        "speed_label": payload["speed_label"],
                        "filler_label": payload["filler_label"],
                    }
                )
            finally:
                if local_path and os.path.isfile(local_path):
                    try:
                        os.unlink(local_path)
                    except OSError:
                        pass

    return {
        "feedback_id": feedback_id,
        "user_id": effective_user_id,
        "slides_processed": len(slide_results),
        "slides": slide_results,
    }
