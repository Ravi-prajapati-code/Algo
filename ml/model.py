"""
ML Model Handler — loads, predicts, and manages the trade-quality model.

Supports XGBoost (primary) with scikit-learn RandomForest as fallback.
The model outputs two values per trade candidate:
  win_probability  : P(trade is profitable ≥ 3 %)
  expected_return  : Estimated net return % (regression)

The signal pipeline uses win_probability as a gate:
  win_probability ≥ ML_MIN_CONFIDENCE → trade allowed
  confidence also scales position size (via RiskManager)

Model persistence: models are saved/loaded from ml/models/ directory.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES, build_feature_vector

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
CLASSIFIER_PATH = MODEL_DIR / "trade_classifier.pkl"
REGRESSOR_PATH  = MODEL_DIR / "return_regressor.pkl"
METADATA_PATH   = MODEL_DIR / "model_metadata.pkl"

# Minimum win probability to allow a trade when ML is active
ML_MIN_CONFIDENCE = float(os.getenv("ML_MIN_CONFIDENCE") or 0.55)
ML_ENABLED        = (os.getenv("ML_ENABLED") or "true").lower() in ("true", "1", "yes")


class MLModelHandler:
    """
    Manages the ML trade-quality model lifecycle.

    Features
    --------
    * Loads pre-trained classifier and regressor from disk.
    * Predicts win probability and expected return for BUY candidates.
    * Falls back gracefully to rule-based system if model not found.
    * Exposes `is_ready` property for conditional activation.

    Usage
    -----
    >>> handler = MLModelHandler()
    >>> if handler.is_ready:
    ...     prob, ret = handler.predict(ind_dict, regime="BULL_TREND")
    ...     if prob >= handler.min_confidence:
    ...         # proceed with trade
    """

    def __init__(self):
        self._classifier = None
        self._regressor  = None
        self._metadata   = {}
        self._load_models()

    # ── Model persistence ──────────────────────────────────────────────────

    def _load_models(self):
        """Load classifier and regressor from disk (silent if not found)."""
        if CLASSIFIER_PATH.exists():
            try:
                with open(CLASSIFIER_PATH, "rb") as f:
                    self._classifier = pickle.load(f)
                logger.info("[ML] Classifier loaded from %s", CLASSIFIER_PATH)
            except Exception as e:
                logger.warning("[ML] Failed to load classifier: %s", e)

        if REGRESSOR_PATH.exists():
            try:
                with open(REGRESSOR_PATH, "rb") as f:
                    self._regressor = pickle.load(f)
                logger.info("[ML] Regressor loaded from %s", REGRESSOR_PATH)
            except Exception as e:
                logger.warning("[ML] Failed to load regressor: %s", e)

        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "rb") as f:
                    self._metadata = pickle.load(f)
            except Exception:
                pass

    def save_models(self, classifier, regressor, metadata: dict = None):
        """Persist trained models to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(classifier, f)
        with open(REGRESSOR_PATH, "wb") as f:
            pickle.dump(regressor, f)
        if metadata:
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(metadata, f)
            self._metadata = metadata
        self._classifier = classifier
        self._regressor  = regressor
        logger.info("[ML] Models saved to %s", MODEL_DIR)

    @property
    def is_ready(self) -> bool:
        """True if classifier is loaded and ML is globally enabled."""
        return ML_ENABLED and self._classifier is not None

    @property
    def min_confidence(self) -> float:
        return ML_MIN_CONFIDENCE

    @property
    def metadata(self) -> dict:
        return self._metadata

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(
        self,
        ind: dict,
        regime: str = "BULL_TREND",
    ) -> tuple[float, float]:
        """
        Predict win probability and expected return for a single candidate.

        Parameters
        ----------
        ind    : Indicator dict from indicators/composite.py
        regime : Current market regime string

        Returns
        -------
        (win_probability, expected_return_pct)
          win_probability   : float in [0, 1]
          expected_return   : float (e.g. 0.08 = 8 % expected return)
        If model not ready → returns (0.5, 0.0) as neutral defaults.
        """
        if not self.is_ready:
            return 0.5, 0.0

        try:
            features = build_feature_vector(ind, regime)
            X = pd.DataFrame([features], columns=FEATURE_NAMES)

            win_prob = float(self._classifier.predict_proba(X)[0][1])

            expected_ret = 0.0
            if self._regressor is not None:
                expected_ret = float(self._regressor.predict(X)[0])

            return round(win_prob, 4), round(expected_ret, 4)

        except Exception as e:
            logger.warning("[ML] Prediction error: %s", e)
            return 0.5, 0.0

    def predict_batch(
        self,
        indicators: dict,
        regime: str = "BULL_TREND",
    ) -> dict:
        """
        Predict for all BUY candidates at once.

        Parameters
        ----------
        indicators : {symbol: ind_dict}
        regime     : Current market regime

        Returns
        -------
        {symbol: {"win_prob": float, "expected_ret": float}}
        """
        if not self.is_ready:
            return {sym: {"win_prob": 0.5, "expected_ret": 0.0} for sym in indicators}

        results = {}
        for symbol, ind in indicators.items():
            wp, er = self.predict(ind, regime)
            results[symbol] = {"win_prob": wp, "expected_ret": er}
        return results

    def score_with_ml(self, base_score: float, win_prob: float) -> float:
        """
        Blend rule-based score with ML win probability.

        Final score = base_score × 0.6 + win_prob_score × 0.4

        win_prob_score maps win_prob [0.5, 1.0] → [0, 100]
        Trades below min_confidence get a 0-score (effectively filtered out).
        """
        if win_prob < self.min_confidence:
            return 0.0
        win_prob_score = (win_prob - 0.5) / 0.5 * 100   # map [0.5,1] → [0,100]
        return round(base_score * 0.6 + win_prob_score * 0.4, 1)


# ── Module-level singleton ─────────────────────────────────────────────────
_handler: Optional[MLModelHandler] = None


def get_model_handler() -> MLModelHandler:
    """Return the module-level MLModelHandler singleton (lazy-loaded)."""
    global _handler
    if _handler is None:
        _handler = MLModelHandler()
    return _handler
