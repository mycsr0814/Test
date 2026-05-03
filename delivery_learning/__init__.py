"""
말의 속도/필러(많음/보통) 라벨을 학습/예측하는 모듈.

핵심 파이프라인:
1) voice 오디오 -> STT -> transcript_text + (세그먼트 발화 구간 합·ffprobe 등으로 duration_sec)
2) transcript_text + duration_sec -> 특징치(feature) 계산
3) 라벨(speed_label, filler_label)로 분류기 학습 (scikit-learn)
4) 학습된 모델로 audio_analysis 테이블의 라벨/수치 예측 업데이트
"""

