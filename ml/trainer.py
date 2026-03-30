"""
ML Training Pipeline.

Trains a classifier (win/loss) and regressor (expected return) on
historical trade data produced by the backtesting engine.

Algorithm selection (in priority order):
  1. XGBoost  — if `xgboost` is installed
  2. LightGBM — if `lightgbm` is installed
  3. RandomForest (scikit-learn) — always available as fallback

Training flow
-------------
1. Load completed trades from DB.
2. Fetch indicator snapshots at trade entry dates from DB/OHLCV cache.
3. Build feature matrix (ml/features.py).
4. Label trades: win=1 if net_return ≥ MIN_WIN_RETURN, else 0.
5. Train classifier + regressor with cross-validation.
6. Print metrics (accuracy, AUC, precision/recall).
7. Save models via MLModelHandler.

Usage
-----
  python main.py train_ml
  python main.py train_ml --start 2022-01-01 --end 2024-12-31
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ml.features import build_feature_matrix, label_trades, FEATURE_NAMES
from ml.model import MLModelHandler, MODEL_DIR

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 30   # Refuse to train if fewer trades available
MIN_WIN_RETURN = 0.03       # 3 % net return = win label


def _get_classifier():
    """Return the best available classifier."""
    try:
        import xgboost as xgb
        logger.info("[Trainer] Using XGBoost classifier")
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        logger.info("[Trainer] Using LightGBM classifier")
        return lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        pass

    from sklearn.ensemble import RandomForestClassifier
    logger.info("[Trainer] Using RandomForest classifier (fallback)")
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def _get_regressor():
    """Return the best available regressor."""
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        pass

    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1,
    )


def train(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> bool:
    """
    Full training pipeline.

    1. Load trades from DB.
    2. Reconstruct indicator snapshots from OHLCV cache.
    3. Build features + labels.
    4. Train classifier and regressor.
    5. Evaluate with cross-validation.
    6. Save models.

    Returns True on success, False if insufficient data.
    """
    from db import repository as repo
    from data.fetcher import fetch_all
    from data.universe import get_all_symbols
    from indicators.composite import compute_indicators

    logger.info("[Trainer] Loading trade history from DB...")
    trades = repo.load_trades()

    if start_date:
        trades = [t for t in trades if t.entry_date >= start_date]
    if end_date:
        trades = [t for t in trades if t.entry_date <= end_date]

    # Only use closed trades with full data
    trades = [
        t for t in trades
        if t.exit_date is not None and t.net_pnl is not None
    ]

    if len(trades) < MIN_TRAINING_SAMPLES:
        logger.error(
            "[Trainer] Only %d trades available; need ≥ %d. Run backtest first.",
            len(trades), MIN_TRAINING_SAMPLES,
        )
        return False

    logger.info("[Trainer] Building features for %d trades...", len(trades))

    # Rebuild indicator snapshots at each trade's entry date
    symbols_needed = list({t.symbol for t in trades})
    all_dates = sorted({t.entry_date for t in trades})
    lookback = (max(all_dates) - min(all_dates)).days + 300

    data = fetch_all(symbols_needed, lookback_days=lookback)

    indicator_records = []
    valid_trades = []

    for trade in trades:
        sym = trade.symbol
        if sym not in data:
            continue
        df = data[sym]
        hist = df[df.index <= trade.entry_date]
        if len(hist) < 60:
            continue
        ind = compute_indicators(hist)
        if ind is None:
            continue
        indicator_records.append(ind)
        valid_trades.append(trade)

    if len(valid_trades) < MIN_TRAINING_SAMPLES:
        logger.error("[Trainer] Only %d valid trades after feature building.", len(valid_trades))
        return False

    # Build feature matrix and labels
    X = build_feature_matrix(indicator_records)
    y_class = np.array(label_trades(valid_trades, min_return_pct=MIN_WIN_RETURN))

    # Regression target: actual return %
    y_reg = np.array([
        t.net_pnl / (t.entry_price * t.shares)
        if (t.entry_price and t.shares and t.entry_price * t.shares > 0) else 0.0
        for t in valid_trades
    ])

    logger.info(
        "[Trainer] Features: %d×%d  |  Win rate: %.1f%%",
        X.shape[0], X.shape[1], y_class.mean() * 100,
    )

    # ── Cross-validation ──────────────────────────────────────────────
    _evaluate_classifier(X, y_class)

    # ── Final training on full dataset ────────────────────────────────
    classifier = _get_classifier()
    classifier.fit(X, y_class)

    regressor = _get_regressor()
    regressor.fit(X, y_reg)

    # ── Save ──────────────────────────────────────────────────────────
    handler = MLModelHandler()
    handler.save_models(
        classifier=classifier,
        regressor=regressor,
        metadata={
            "n_trades":       len(valid_trades),
            "win_rate":       float(y_class.mean()),
            "feature_names":  FEATURE_NAMES,
            "train_start":    str(min(t.entry_date for t in valid_trades)),
            "train_end":      str(max(t.entry_date for t in valid_trades)),
        },
    )

    logger.info("[Trainer] Training complete. Models saved to %s", MODEL_DIR)
    return True


def _evaluate_classifier(X: pd.DataFrame, y: np.ndarray):
    """Run 5-fold cross-validation and print key metrics."""
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import make_scorer, roc_auc_score

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = _get_classifier()

        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        auc_scores = cross_val_score(
            clf, X, y, cv=cv,
            scoring=make_scorer(roc_auc_score, needs_proba=True),
        )

        logger.info(
            "[Trainer] CV Accuracy: %.3f ± %.3f  |  AUC: %.3f ± %.3f",
            acc_scores.mean(), acc_scores.std(),
            auc_scores.mean(), auc_scores.std(),
        )
    except Exception as e:
        logger.warning("[Trainer] CV evaluation failed: %s", e)


def print_feature_importance(model, top_n: int = 10):
    """Print top feature importances (works for tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )[:top_n]
    print("\nTop Feature Importances:")
    for name, score in importances:
        bar = "█" * int(score * 200)
        print(f"  {name:<22} {score:.4f}  {bar}")
