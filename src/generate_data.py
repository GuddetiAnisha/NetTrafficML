"""Generate reproducible synthetic, 5G-like traffic observations.

This is a methodology demo only. It does not contain Ericsson or production data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

REGIONS = {"Stockholm": 1.15, "Gothenburg": 1.00, "Malmo": 0.92, "Uppsala": 0.78}

def generate_traffic(n: int = 60000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = np.array(list(REGIONS))
    region = rng.choice(names, n, p=[0.42, 0.27, 0.21, 0.10])
    region_factor = np.array([REGIONS[r] for r in region])
    timestamp = pd.date_range("2026-01-01", periods=n, freq="100ms")
    hour = timestamp.hour.to_numpy()
    dow = timestamp.dayofweek.to_numpy()

    peak = 0.55 + 0.45 * (
        np.exp(-((hour - 12) / 4.5) ** 2) + np.exp(-((hour - 20) / 3.0) ** 2)
    )
    peak = np.clip(peak, 0.45, 1.35)
    active_users = np.maximum(10, (rng.normal(1000, 300, n) * peak * region_factor).astype(int))
    packet_rate = rng.gamma(3.0, 380.0, n) * peak * region_factor
    signal_quality = np.clip(rng.normal(-78, 9, n) - active_users / 5000, -120, -45)
    queue_depth = np.maximum(0, rng.poisson(np.clip(active_users / 260 + packet_rate / 2200, 0.1, None)))
    latency_ms = np.clip(
        rng.normal(13, 4.5, n) + active_users / 650 + queue_depth * 0.65
        + np.maximum(-90 - signal_quality, 0) * 0.18,
        1, None
    )
    retransmission_rate = np.clip(
        rng.beta(1.4, 26, n) + latency_ms / 1700
        + np.maximum(-88 - signal_quality, 0) / 850,
        0, 0.40
    )
    congestion_score = (
        0.0024 * active_users + 0.0014 * packet_rate + 0.16 * queue_depth
        + 8.0 * retransmission_rate + 0.040 * latency_ms
    )
    congested = (congestion_score > np.quantile(congestion_score, 0.88)).astype(int)
    downlink_mbps = np.maximum(
        0, 0.052 * active_users + 0.019 * packet_rate - 1.65 * queue_depth
        + 0.58 * (signal_quality + 100) - 13.0 * congested + rng.normal(0, 15, n)
    )
    return pd.DataFrame({
        "timestamp": timestamp, "region": region, "hour": hour, "day_of_week": dow,
        "active_users": active_users, "packet_rate": packet_rate.round(3),
        "latency_ms": latency_ms.round(3),
        "retransmission_rate": retransmission_rate.round(5),
        "queue_depth": queue_depth, "signal_quality_dbm": signal_quality.round(3),
        "downlink_mbps": downlink_mbps.round(3), "congested": congested,
    })

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_traffic()
    csv_path = DATA_DIR / "network_traffic.csv"
    parquet_path = DATA_DIR / "network_traffic.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df):,} rows")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print("\nClass distribution:")
    print(df["congested"].value_counts(normalize=True).rename("share"))
    print("\nRegional distribution:")
    print(df["region"].value_counts(normalize=True).rename("share"))

if __name__ == "__main__":
    main()
