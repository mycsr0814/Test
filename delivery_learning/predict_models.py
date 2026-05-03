import json
import os
from pathlib import Path

from openai import OpenAI

from delivery_learning.config import settings
from delivery_learning.consts import DEFAULT_FILLER_HIGH_RATIO
from delivery_learning.features import analyze_transcript_for_features, build_feature_vector
from delivery_learning.ml_models import TrainedModelBundle, predict_speed_and_filler
from delivery_learning.stt_durations import resolve_duration_for_metrics


def predict_labels_from_transcript(
    transcript_text: str,
    duration_sec: float | None,
    model_dir: str,
) -> dict:
    bundle = TrainedModelBundle.load(model_dir)
    feature_row = build_feature_vector(transcript_text, duration_sec)
    return predict_speed_and_filler(bundle=bundle, feature_row=feature_row)


def _openai_verbose_transcribe(
    audio_path: Path,
    transcribe_model: str | None,
    openai_api_key: str | None,
) -> tuple[str, float | None, str]:
    api_key = openai_api_key or settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다(동작: OpenAI STT).")
    client = OpenAI(api_key=api_key)
    model = transcribe_model or os.environ.get("TRANSCRIBE_MODEL", "whisper-1")

    with audio_path.open("rb") as f:
        res = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )

    res_dict = res if isinstance(res, dict) else getattr(res, "__dict__", {})
    transcript_text = res_dict.get("text") or getattr(res, "text", "") or ""

    segments = res_dict.get("segments") or getattr(res, "segments", None)
    segments_list = segments if isinstance(segments, list) else None
    duration_sec = resolve_duration_for_metrics(segments_list, audio_path)

    return transcript_text, duration_sec, model


def _local_whisper_transcribe(
    audio_path: Path,
    local_model_name: str | None = None,
) -> tuple[str, float | None, str]:
    import whisper

    model_name = local_model_name or os.environ.get("LOCAL_WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), verbose=False)

    transcript_text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    segments_list = segments if isinstance(segments, list) else None
    duration_sec = resolve_duration_for_metrics(segments_list, audio_path)

    return transcript_text, duration_sec, f"local-whisper:{model_name}"


def _resolve_stt_backend(openai_api_key: str | None) -> str:
    """
    STT_BACKEND:
    - openai: OpenAI Whisper API 강제
    - local: 로컬 Whisper 강제
    - auto(기본): OPENAI_API_KEY 있으면 openai, 없으면 local
    """
    backend = (os.environ.get("STT_BACKEND", "auto") or "auto").strip().lower()
    if backend not in {"auto", "openai", "local"}:
        backend = "auto"
    if backend == "auto":
        return "openai" if (openai_api_key or settings.OPENAI_API_KEY) else "local"
    return backend


def _filler_tokens_json(filler_token_counts: dict[str, int]) -> str:
    items = [
        {"word": k, "count": int(v)}
        for k, v in sorted(filler_token_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    return json.dumps(items, ensure_ascii=False)


def _stabilize_filler_label(pred: dict, filler_ratio: float) -> str:
    """
    모델 예측이 과민하게 '많음'으로 치우치는 것을 방지하기 위한 후처리.
    필러 비율이 임계치보다 낮으면 '보통'으로 보정합니다.
    """
    label = str(pred.get("filler_label", "보통"))
    if filler_ratio < DEFAULT_FILLER_HIGH_RATIO:
        return "보통"
    return label


def transcribe_then_label_with_bundle(
    audio_path: str | Path,
    bundle: TrainedModelBundle,
    transcribe_model: str | None = None,
    openai_api_key: str | None = None,
) -> dict:
    """
    오디오 -> OpenAI STT -> 통계/특징 -> 학습된 분류기로 speed/filler 라벨.
    DB 저장에 필요한 필드를 한 번에 반환합니다.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    backend = _resolve_stt_backend(openai_api_key)
    if backend == "openai":
        transcript_text, duration_sec, whisper_model = _openai_verbose_transcribe(
            audio_path, transcribe_model, openai_api_key
        )
    else:
        transcript_text, duration_sec, whisper_model = _local_whisper_transcribe(
            audio_path,
            local_model_name=transcribe_model,
        )
    stats = analyze_transcript_for_features(transcript_text, duration_sec)
    feature_row = build_feature_vector(transcript_text, duration_sec)
    pred = predict_speed_and_filler(bundle=bundle, feature_row=feature_row)

    return {
        "transcript_text": transcript_text,
        "duration_sec": duration_sec,
        "whisper_model": whisper_model,
        "word_count": stats.word_count,
        "wpm": stats.wpm,
        "filler_count": stats.filler_count,
        "filler_ratio": stats.filler_ratio,
        "filler_tokens_json": _filler_tokens_json(stats.filler_token_counts),
        "speed_label": pred["speed_label"],
        "filler_label": _stabilize_filler_label(pred, stats.filler_ratio),
    }


def transcribe_and_predict(
    audio_path: str | Path,
    model_dir: str,
    transcribe_model: str | None = None,
    openai_api_key: str | None = None,
) -> dict:
    """
    오디오 파일 -> STT -> transcript_text + duration_sec -> 특징치 -> speed/filler 라벨 예측
    """
    audio_path = Path(audio_path)
    bundle = TrainedModelBundle.load(model_dir)
    return transcribe_then_label_with_bundle(
        audio_path,
        bundle=bundle,
        transcribe_model=transcribe_model,
        openai_api_key=openai_api_key,
    )

