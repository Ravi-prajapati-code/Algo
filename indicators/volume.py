"""Volume indicators: volume ratio and spike detection."""

import pandas as pd
from config.settings import VOLUME_SPIKE_MULTIPLIER


def compute_volume(df: pd.DataFrame, avg_period: int = 20) -> dict:
    """
    Returns:
        vol_avg         — 20-day average volume
        vol_today       — latest volume
        vol_ratio       — today / avg (>1.5 = spike)
        vol_spike       — True if today > 1.5x average
        vol_increasing  — last 3 days volume trending up
    """
    volume = df["volume"]
    avg_vol = volume.rolling(avg_period).mean()

    last_vol = float(volume.iloc[-1])
    last_avg = float(avg_vol.iloc[-1]) if not pd.isna(avg_vol.iloc[-1]) else 1.0
    ratio = last_vol / last_avg if last_avg > 0 else 0.0

    # Check if volume has been increasing over last 3 days
    if len(volume) >= 3:
        v1, v2, v3 = float(volume.iloc[-3]), float(volume.iloc[-2]), float(volume.iloc[-1])
        increasing = v3 > v2 > v1
    else:
        increasing = False

    return {
        "vol_avg": round(last_avg),
        "vol_today": round(last_vol),
        "vol_ratio": round(ratio, 2),
        "vol_spike": ratio >= VOLUME_SPIKE_MULTIPLIER,
        "vol_increasing": increasing,
    }
