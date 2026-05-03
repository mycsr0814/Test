"""
Whisper STT 세그먼트·오디오 파일에서 WPM/특징치용 duration_sec을 추정합니다.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable


def _iter_segment_start_end(segments: list[Any] | None) -> Iterable[tuple[float, float]]:
    if not segments:
        return
    for s in segments:
        start: float | None = None
        end: float | None = None
        if isinstance(s, dict):
            start = s.get("start")  # type: ignore[assignment]
            end = s.get("end")  # type: ignore[assignment]
        else:
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            yield float(start), float(end)


def speech_span_duration_sec(segments: list[Any] | None) -> float | None:
    """세그먼트별 (end - start) 합. 슬라이드 내 긴 무음 구간이 있어도 '말한 시간'에 가깝게 WPM 분모를 잡습니다."""
    total = 0.0
    for s, e in _iter_segment_start_end(segments):
        total += max(0.0, e - s)
    return total if total > 1e-6 else None


def timeline_end_sec(segments: list[Any] | None) -> float | None:
    """첫 발화부터 마지막 세그먼트 끝까지의 타임라인 길이(기존 max(end)와 동일)."""
    mx: float | None = None
    for _, e in _iter_segment_start_end(segments):
        mx = e if mx is None else max(mx, e)
    return mx


def ffprobe_duration_sec(audio_path: Path) -> float | None:
    """ffprobe가 PATH에 있을 때만 컨테이너 상의 오디오 길이(초)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return None
        out = (proc.stdout or "").strip()
        if not out:
            return None
        return float(out)
    except (FileNotFoundError, ValueError, subprocess.SubprocessError, OSError):
        return None


def resolve_duration_for_metrics(segments: list[Any] | None, audio_path: Path | None) -> float | None:
    """
    WPM·ML 특징에 쓸 duration(초) 우선순위:
    1) 세그먼트 발화 구간 합
    2) 마지막 세그먼트 end (세그먼트는 있으나 합이 0에 가까울 때)
    3) ffprobe로 파일 길이 (세그먼트 없을 때)
    """
    sp = speech_span_duration_sec(segments)
    te = timeline_end_sec(segments)
    if sp is not None and sp > 0:
        # 비정상적으로 짧은 합(버그·빈 세그먼트)이면 타임라인 end로 완화
        if te is not None and te > 0 and sp < 0.15 * te:
            return te
        return sp
    if te is not None and te > 0:
        return te
    if audio_path is not None and audio_path.is_file():
        return ffprobe_duration_sec(audio_path)
    return None
