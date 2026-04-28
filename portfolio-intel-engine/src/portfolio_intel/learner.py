from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


@dataclass(frozen=True)
class LearnerConfig:
    model_path: Path = Path("data") / "models" / "sgd_logreg.json"
    feature_cols: tuple[str, ...] = (
        "roc",
        "slope_20",
        "zscore",
        "compression",
        "breakout_strength",
        "liq_sweep",
        "structure_break",
        "vol_ratio",
    )


class OnlineLogisticLearner:
    """
    Online-ish logistic regression using SGDClassifier(log_loss).
    Stores weights to disk for reproducibility.
    """

    def __init__(self, cfg: LearnerConfig | None = None) -> None:
        self.cfg = cfg or LearnerConfig()
        self.cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1, learning_rate="optimal")
        self.is_fitted = False
        self._load()

    def _load(self) -> None:
        if not self.cfg.model_path.exists():
            return
        payload = json.loads(self.cfg.model_path.read_text(encoding="utf-8"))
        self.model.coef_ = np.array(payload["coef"], dtype=float)
        self.model.intercept_ = np.array(payload["intercept"], dtype=float)
        self.model.classes_ = np.array([0, 1], dtype=int)
        self.is_fitted = True

    def _save(self) -> None:
        payload = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "feature_cols": list(self.cfg.feature_cols),
        }
        self.cfg.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def predict_proba_up(self, feat_row: pd.Series) -> float:
        x = self._row_to_x(feat_row)
        if not self.is_fitted:
            # neutral prior before learning
            return 0.5
        proba = self.model.predict_proba([x])[0, 1]
        return float(np.clip(proba, 0.0, 1.0))

    def partial_fit(self, feat_df: pd.DataFrame, future_return: pd.Series) -> None:
        """
        Train on rows where target is known:
          y=1 if next-period return > 0 else 0
        """
        df = feat_df.copy()
        y = (future_return.astype(float) > 0).astype(int)
        df = df.loc[y.index].dropna(subset=list(self.cfg.feature_cols))
        y = y.loc[df.index]
        if df.empty:
            return
        X = df[list(self.cfg.feature_cols)].astype(float).values
        if not self.is_fitted:
            self.model.partial_fit(X, y.values, classes=np.array([0, 1]))
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y.values)
        self._save()

    def _row_to_x(self, feat_row: pd.Series) -> np.ndarray:
        return np.array([float(feat_row.get(c, 0.0)) for c in self.cfg.feature_cols], dtype=float)

