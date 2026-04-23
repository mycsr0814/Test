import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from delivery_learning.consts import SPEED_LABELS, FILLER_LABELS


@dataclass
class TrainedModelBundle:
    speed_model: Any
    filler_model: Any
    feature_order: list[str]

    def save(self, model_dir: str) -> None:
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.speed_model, d / "speed_model.joblib")
        joblib.dump(self.filler_model, d / "filler_model.joblib")
        (d / "feature_order.json").write_text(
            json.dumps(self.feature_order, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def load(model_dir: str) -> "TrainedModelBundle":
        d = Path(model_dir)
        feature_order = json.loads((d / "feature_order.json").read_text(encoding="utf-8"))
        speed_model = joblib.load(d / "speed_model.joblib")
        filler_model = joblib.load(d / "filler_model.joblib")
        return TrainedModelBundle(speed_model=speed_model, filler_model=filler_model, feature_order=feature_order)


def _make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                ),
            ),
        ]
    )


def _to_matrix(feature_rows: list[dict[str, float]], feature_order: list[str]):
    import numpy as np

    x = []
    for row in feature_rows:
        x.append([float(row.get(k, 0.0)) for k in feature_order])
    return np.array(x, dtype=float)


def train_speed_and_filler_models(
    feature_rows: list[dict[str, float]],
    speed_labels: list[str],
    filler_labels: list[str],
) -> TrainedModelBundle:
    """
    speed_label(느림/보통/빠름)과 filler_label(보통/많음)을 각각 별도 분류기로 학습.
    """
    if not feature_rows:
        raise ValueError("feature_rows가 비어 있습니다.")

    sample = feature_rows[0]
    feature_order = sorted(sample.keys())
    x = _to_matrix(feature_rows, feature_order)

    if len(set(speed_labels)) < 1:
        raise ValueError("speed_labels가 유효하지 않습니다.")
    if len(set(filler_labels)) < 1:
        raise ValueError("filler_labels가 유효하지 않습니다.")

    speed_model = _make_classifier()
    filler_model = _make_classifier()

    speed_model.fit(x, speed_labels)
    filler_model.fit(x, filler_labels)

    return TrainedModelBundle(speed_model=speed_model, filler_model=filler_model, feature_order=feature_order)


def predict_speed_and_filler(bundle: TrainedModelBundle, feature_row: dict[str, float]) -> dict[str, Any]:
    x = [[float(feature_row.get(k, 0.0)) for k in bundle.feature_order]]

    if hasattr(bundle.speed_model, "predict_proba"):
        sp_proba = bundle.speed_model.predict_proba(x)[0]
        sp_idx = int(sp_proba.argmax())
        speed_label = bundle.speed_model.classes_[sp_idx]
        speed_conf = float(sp_proba[sp_idx])
    else:
        speed_label = bundle.speed_model.predict(x)[0]
        speed_conf = None

    if hasattr(bundle.filler_model, "predict_proba"):
        fi_proba = bundle.filler_model.predict_proba(x)[0]
        fi_idx = int(fi_proba.argmax())
        filler_label = bundle.filler_model.classes_[fi_idx]
        filler_conf = float(fi_proba[fi_idx])
    else:
        filler_label = bundle.filler_model.predict(x)[0]
        filler_conf = None

    return {
        "speed_label": speed_label,
        "speed_confidence": speed_conf,
        "filler_label": filler_label,
        "filler_confidence": filler_conf,
    }

