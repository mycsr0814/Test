import argparse
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from delivery_learning.config import settings
from delivery_learning.features import analyze_transcript_for_features, build_feature_vector
from delivery_learning.ml_models import TrainedModelBundle, predict_speed_and_filler


def _get_engine():
    return create_engine(settings.db_connection_string, fast_executemany=True)


def _build_filler_tokens_json(filler_token_counts: dict[str, int]) -> str:
    items = [
        {"word": k, "count": int(v)}
        for k, v in sorted(filler_token_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    return json.dumps(items, ensure_ascii=False)


def _fetch_distinct_feedback_ids_with_transcripts(conn: Connection) -> list[int | None]:
    """transcript_text가 있는 audio_analysis 행이 존재하는 feedback_id 목록(DB 조회)."""
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT feedback_id
            FROM audio_analysis
            WHERE transcript_text IS NOT NULL
            ORDER BY feedback_id
            """
        )
    ).fetchall()
    return [r.feedback_id for r in rows]


def _fetch_audio_rows_for_feedback(conn: Connection, feedback_id: int | None) -> list[Any]:
    if feedback_id is None:
        q = text(
            """
            SELECT id, transcript_text, duration_sec
            FROM audio_analysis
            WHERE transcript_text IS NOT NULL
              AND feedback_id IS NULL
            """
        )
        return list(conn.execute(q).fetchall())
    q = text(
        """
        SELECT id, transcript_text, duration_sec
        FROM audio_analysis
        WHERE transcript_text IS NOT NULL
          AND feedback_id = :feedback_id
        """
    )
    return list(conn.execute(q, {"feedback_id": feedback_id}).fetchall())


def _apply_model_to_rows(
    conn: Connection,
    bundle: TrainedModelBundle,
    rows: list[Any],
    analyzer_version: str,
) -> None:
    for row in rows:
        audio_analysis_id = int(row.id)
        transcript_text = row.transcript_text or ""
        duration_sec = float(row.duration_sec) if row.duration_sec is not None else None

        stats = analyze_transcript_for_features(transcript_text, duration_sec)
        feature_row = build_feature_vector(transcript_text, duration_sec)
        pred = predict_speed_and_filler(bundle=bundle, feature_row=feature_row)

        filler_tokens_json = _build_filler_tokens_json(stats.filler_token_counts)

        conn.execute(
            text(
                """
                UPDATE audio_analysis
                SET
                    word_count = :word_count,
                    wpm = :wpm,
                    speed_label = :speed_label,
                    filler_count = :filler_count,
                    filler_ratio = :filler_ratio,
                    filler_label = :filler_label,
                    filler_tokens_json = :filler_tokens_json,
                    analyzer_version = :analyzer_version,
                    updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {
                "id": audio_analysis_id,
                "word_count": stats.word_count,
                "wpm": stats.wpm,
                "speed_label": pred["speed_label"],
                "filler_count": stats.filler_count,
                "filler_ratio": stats.filler_ratio,
                "filler_label": pred["filler_label"],
                "filler_tokens_json": filler_tokens_json,
                "analyzer_version": analyzer_version,
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )


def apply_model(
    model_dir: str,
    feedback_id: int | None,
    analyzer_version: str,
):
    bundle = TrainedModelBundle.load(model_dir=model_dir)
    engine = _get_engine()

    with engine.begin() as conn:
        if feedback_id is not None:
            feedback_ids: list[int | None] = [feedback_id]
            print(f"단일 feedback_id={feedback_id} 만 적용합니다.")
        else:
            feedback_ids = _fetch_distinct_feedback_ids_with_transcripts(conn)
            preview = feedback_ids[:30]
            more = f" 외 {len(feedback_ids) - 30}개" if len(feedback_ids) > 30 else ""
            print(
                f"DB에서 transcript가 있는 audio_analysis 기준 feedback_id {len(feedback_ids)}개 조회: "
                f"{preview}{more}"
            )

        total_rows = 0
        for fid in feedback_ids:
            rows = _fetch_audio_rows_for_feedback(conn, fid)
            if not rows:
                continue
            _apply_model_to_rows(conn, bundle, rows, analyzer_version)
            total_rows += len(rows)
            print(f"  feedback_id={fid!s}: {len(rows)} rows")

        if total_rows == 0:
            print("업데이트할 row가 없습니다.")
            return

    if feedback_id is not None:
        print(f"모델 적용 완료: feedback_id={feedback_id}, rows={total_rows}")
    else:
        print(
            f"모델 적용 완료: 전체 세션(DB 조회 feedback_id {len(feedback_ids)}개), 총 rows={total_rows}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="delivery_learning_models")
    parser.add_argument("--feedback-id", type=int, default=None)
    parser.add_argument("--analyzer-version", default="delivery_learning_v1")
    args = parser.parse_args()

    apply_model(
        model_dir=args.model_dir,
        feedback_id=args.feedback_id,
        analyzer_version=args.analyzer_version,
    )


if __name__ == "__main__":
    main()

