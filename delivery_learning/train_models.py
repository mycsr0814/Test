import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

from delivery_learning.config import settings
from delivery_learning.features import build_feature_vector
from delivery_learning.ml_models import train_speed_and_filler_models


def _iter_voice_samples(voice_dir: Path, label_csv: Path | None) -> Iterable[tuple[Path, str, str]]:
    """
    반환: (audio_path, speed_label, filler_label)
    label_csv가 있으면 CSV에서 라벨을 읽고, 없으면 파일명에서 라벨을 추정합니다.
    """
    if label_csv and label_csv.exists():
        with label_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file") or row.get("audio_file") or row.get("filename")
                speed_label = row.get("speed_label") or row.get("speed") or ""
                filler_label = row.get("filler_label") or row.get("filler") or ""
                if not file_name or not speed_label or not filler_label:
                    continue
                audio_path = voice_dir / file_name
                if not audio_path.exists():
                    continue
                yield (audio_path, speed_label, filler_label)
        return

    # CSV가 없으면 파일명 기반 추정(정확하지 않을 수 있으니 CSV 권장)
    for p in sorted(voice_dir.glob("*")):
        if p.suffix.lower() not in (".m4a", ".mp3", ".wav", ".webm"):
            continue
        name = p.name

        speed_label = "보통"
        if "빠름" in name:
            speed_label = "빠름"
        elif "느림" in name:
            speed_label = "느림"

        filler_label = "보통"
        if "많음" in name:
            filler_label = "많음"

        yield (p, speed_label, filler_label)


def transcribe_audio_openai(audio_path: Path, model: str) -> tuple[str, float | None]:
    """
    OpenAI Whisper STT:
    오디오 -> transcript_text + duration_sec(segments end 기반 추정)
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다(동작: stt-backend=openai).")

    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    with audio_path.open("rb") as f:
        res = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",
        )

    res_dict = res if isinstance(res, dict) else getattr(res, "__dict__", {})
    transcript_text = res_dict.get("text") if isinstance(res_dict, dict) else None
    transcript_text = (transcript_text or getattr(res, "text", "")).strip()  # type: ignore[attr-defined]

    segments = res_dict.get("segments") or getattr(res, "segments", None)  # type: ignore[attr-defined]
    duration_sec = None
    if isinstance(segments, list) and segments:
        ends: list[float] = []
        for s in segments:
            if isinstance(s, dict):
                end = s.get("end")
                if isinstance(end, (int, float)):
                    ends.append(float(end))
        if ends:
            duration_sec = max(ends)

    return transcript_text, duration_sec


def transcribe_audio_local(audio_path: Path, model_name: str) -> tuple[str, float | None]:
    """
    로컬 Whisper STT:
    오디오 -> transcript_text + duration_sec(segments end 기반 추정)
    """
    import whisper

    m = whisper.load_model(model_name)
    result = m.transcribe(str(audio_path), verbose=False)

    transcript_text = (result.get("text") or "").strip()
    segments = result.get("segments") or []

    duration_sec = None
    if isinstance(segments, list) and segments:
        ends: list[float] = []
        for s in segments:
            if isinstance(s, dict):
                end = s.get("end")
                if isinstance(end, (int, float)):
                    ends.append(float(end))
        if ends:
            duration_sec = max(ends)

    return transcript_text, duration_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-dir", default=str(Path("voice")))
    parser.add_argument("--label-csv", default=str(Path("voice_labels.csv")))
    parser.add_argument("--model-dir", default=str(Path("delivery_learning_models")))
    parser.add_argument("--stt-backend", default="local", choices=["local", "openai"])
    parser.add_argument("--openai-transcribe-model", default=os.environ.get("TRANSCRIBE_MODEL", "whisper-1"))
    parser.add_argument("--local-whisper-model", default=os.environ.get("LOCAL_WHISPER_MODEL", "base"))
    args = parser.parse_args()

    voice_dir = Path(args.voice_dir).resolve()
    label_csv = Path(args.label_csv)
    if not label_csv.is_absolute():
        label_csv = (voice_dir.parent / label_csv).resolve()

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, float]] = []
    speed_labels: list[str] = []
    filler_labels: list[str] = []

    cache_path = model_dir / "transcripts_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    samples = list(_iter_voice_samples(voice_dir, label_csv if label_csv.exists() else None))
    if not samples:
        raise RuntimeError(f"학습 데이터가 없습니다. voice_dir={voice_dir}, label_csv={label_csv}")

    stt_tag = f"stt={args.stt_backend}"
    for audio_path, speed_label, filler_label in samples:
        # backend/model이 바뀌면 transcript가 달라질 수 있으니 캐시 키에 포함
        model_tag = args.openai_transcribe_model if args.stt_backend == "openai" else args.local_whisper_model
        key = f"{audio_path}|{stt_tag}|model={model_tag}"
        if key in cache:
            transcript_text = cache[key].get("transcript_text", "")
            duration_sec = cache[key].get("duration_sec", None)
        else:
            if args.stt_backend == "openai":
                transcript_text, duration_sec = transcribe_audio_openai(
                    audio_path=audio_path,
                    model=args.openai_transcribe_model,
                )
            else:
                transcript_text, duration_sec = transcribe_audio_local(
                    audio_path=audio_path,
                    model_name=args.local_whisper_model,
                )
            cache[key] = {"transcript_text": transcript_text, "duration_sec": duration_sec}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

        feature_rows.append(build_feature_vector(transcript_text, duration_sec))
        speed_labels.append(speed_label)
        filler_labels.append(filler_label)

    bundle = train_speed_and_filler_models(
        feature_rows=feature_rows,
        speed_labels=speed_labels,
        filler_labels=filler_labels,
    )
    bundle.save(str(model_dir))
    print(f"모델 학습 완료: {model_dir}")


if __name__ == "__main__":
    main()

