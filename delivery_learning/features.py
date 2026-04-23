import math
import re
from dataclasses import dataclass

from delivery_learning.consts import FILLER_WORDS


@dataclass(frozen=True)
class TranscriptStats:
    word_count: int
    wpm: float | None
    filler_count: int
    filler_ratio: float
    filler_token_counts: dict[str, int]


def _tokenize_korean_words(text: str) -> list[str]:
    """
    전사 텍스트에서 한국어 '어절/토큰'에 가까운 단위를 뽑기 위한 단순 토크나이저.
    """
    if not text:
        return []

    rough = re.split(r"\s+", text.strip())
    tokens: list[str] = []

    for t in rough:
        t = t.strip()
        if not t:
            continue
        # 기호 제거
        t = re.sub(
            r"^[^\uAC00-\uD7A3\u0030-\u0039a-zA-Z]+|[^\uAC00-\uD7A3\u0030-\u0039a-zA-Z]+$",
            "",
            t,
        )
        if t:
            tokens.append(t)
    return tokens


def analyze_transcript_for_features(transcript_text: str, duration_sec: float | None) -> TranscriptStats:
    tokens = _tokenize_korean_words(transcript_text)
    word_count = len(tokens)

    if duration_sec is None or duration_sec <= 0:
        wpm = None
        filler_ratio = 0.0
    else:
        wpm = word_count / (duration_sec / 60.0) if duration_sec > 0 else None

        # filler_ratio는 word_count 기반이라 duration과 무관하게 계산 가능하지만,
        # 여기서는 원형 계산 후 아래에서 안전장치를 둡니다.
        filler_ratio = 0.0

    filler_token_counts: dict[str, int] = {}
    filler_count = 0

    filler_words_set = set(FILLER_WORDS)
    for tok in tokens:
        if tok in filler_words_set:
            filler_token_counts[tok] = filler_token_counts.get(tok, 0) + 1
            filler_count += 1

    filler_ratio = filler_count / max(word_count, 1)

    # NaN 방지
    if math.isnan(filler_ratio) or math.isinf(filler_ratio):
        filler_ratio = 0.0

    return TranscriptStats(
        word_count=word_count,
        wpm=wpm,
        filler_count=filler_count,
        filler_ratio=filler_ratio,
        filler_token_counts=filler_token_counts,
    )


def build_feature_vector(transcript_text: str, duration_sec: float | None) -> dict[str, float]:
    """
    ML 입력용 특징치.
    - speed_label/filler_label 분류에 직접 쓰일 최소한의 수치들 + filler 단어 후보별 카운트 일부
    """
    stats = analyze_transcript_for_features(transcript_text, duration_sec)

    wpm_val = stats.wpm if stats.wpm is not None else 0.0
    duration_val = float(duration_sec) if duration_sec is not None else 0.0

    filler_word_features = {
        f"filler_{w}": float(stats.filler_token_counts.get(w, 0))
        for w in FILLER_WORDS
    }

    return {
        "duration_sec": duration_val,
        "word_count": float(stats.word_count),
        "wpm": float(wpm_val),
        "filler_count": float(stats.filler_count),
        "filler_ratio": float(stats.filler_ratio),
        **filler_word_features,
    }

