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
    전사 텍스트에서 단어 수·WPM용 토큰.
    띄어쓰기만 쓰면 Whisper 한국어 전사에서 어절이 한 덩어리로 붙을 때 단어 수가 크게 과소됩니다.
    연속 한글·영숫자 덩어리를 토큰으로 잡습니다.
    """
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", normalized)


def _is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and "\uAC00" <= ch <= "\uD7A3"


def _boundary_char(ch: str) -> bool:
    if not ch:
        return True
    return ch in " \t\n\r,.!?;:\"'…~·[](){}<>/\\|!=+-*&^%$#@`"


def _filler_span_valid(raw: str, j: int, wlen: int) -> bool:
    """필러 후보 w가 raw[j:j+wlen]에 있을 때, 앞뒤 경계가 '말버릇'으로 셀 만한지."""
    if j < 0 or j + wlen > len(raw):
        return False
    prev_c = raw[j - 1] if j > 0 else ""
    next_c = raw[j + wlen] if j + wlen < len(raw) else ""

    if wlen == 1 and _is_hangul_syllable(raw[j]):
        # 양옆이 모두 한글 음절이면 일반 단어 안의 한 글자로 간주 (예: '아니'의 '아')
        if _is_hangul_syllable(prev_c) and _is_hangul_syllable(next_c):
            return False

    left_ok = j == 0 or _boundary_char(prev_c) or not _is_hangul_syllable(prev_c)
    right_ok = j + wlen >= len(raw) or _boundary_char(next_c) or not _is_hangul_syllable(next_c)
    return left_ok and right_ok


def _filler_token_counts_boundary(text: str) -> dict[str, int]:
    """
    전사 문자열에서 필러 후보를 경계 규칙과 함께 센다.
    (1) 2글자 이상 필러를 먼저 매칭해 가리면, '근데' 안의 '근' 등 오탐을 줄인다.
    (2) 매칭한 구간은 플레이스홀더로 덮어 이후 한 글자 필러 탐색에서 제외한다.
    """
    raw = text.strip()
    counts: dict[str, int] = {w: 0 for w in FILLER_WORDS}
    if not raw:
        return counts

    multi = sorted([w for w in FILLER_WORDS if len(w) >= 2], key=len, reverse=True)
    chars = list(raw)

    for w in multi:
        wlen = len(w)
        start = 0
        while True:
            idx = "".join(chars).find(w, start)
            if idx < 0:
                break
            if _filler_span_valid("".join(chars), idx, wlen):
                counts[w] += 1
                for k in range(idx, idx + wlen):
                    chars[k] = "\ufffc"  # object replacement / placeholder
                start = idx + wlen
            else:
                start = idx + 1

    masked = "".join(chars)
    singles = [w for w in FILLER_WORDS if len(w) == 1]

    for w in singles:
        start = 0
        wlen = 1
        while True:
            idx = masked.find(w, start)
            if idx < 0:
                break
            if masked[idx] == "\ufffc":
                start = idx + 1
                continue
            if _filler_span_valid(masked, idx, wlen):
                counts[w] += 1
                masked = masked[:idx] + "\ufffc" + masked[idx + 1 :]
                start = idx + 1
            else:
                start = idx + 1

    return counts


def analyze_transcript_for_features(transcript_text: str, duration_sec: float | None) -> TranscriptStats:
    tokens = _tokenize_korean_words(transcript_text)
    word_count = len(tokens)

    if duration_sec is None or duration_sec <= 0:
        wpm = None
    else:
        wpm = word_count / (duration_sec / 60.0) if duration_sec > 0 else None

    filler_token_counts = _filler_token_counts_boundary(transcript_text)
    filler_count = int(sum(filler_token_counts.values()))

    filler_ratio = filler_count / max(word_count, 1)
    filler_ratio = min(1.0, filler_ratio)

    # NaN 방지
    if math.isnan(filler_ratio) or math.isinf(filler_ratio):
        filler_ratio = 0.0

    return TranscriptStats(
        word_count=word_count,
        wpm=wpm,
        filler_count=filler_count,
        filler_ratio=filler_ratio,
        filler_token_counts={k: v for k, v in filler_token_counts.items() if v > 0},
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
