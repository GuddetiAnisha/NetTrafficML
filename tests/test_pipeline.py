import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_data import generate_traffic

def test_generator_schema_and_size():
    df = generate_traffic(n=2000, seed=7)
    expected = {
        "timestamp", "region", "hour", "day_of_week", "active_users",
        "packet_rate", "latency_ms", "retransmission_rate", "queue_depth",
        "signal_quality_dbm", "downlink_mbps", "congested",
    }
    assert set(df.columns) == expected
    assert len(df) == 2000

def test_imbalance_is_present():
    df = generate_traffic(n=5000, seed=8)
    rate = df["congested"].mean()
    assert 0.07 < rate < 0.18

def test_ranges_are_sensible():
    df = generate_traffic(n=1000, seed=9)
    assert (df["latency_ms"] > 0).all()
    assert (df["downlink_mbps"] >= 0).all()
    assert df["retransmission_rate"].between(0, 0.4).all()
    assert set(np.unique(df["congested"])).issubset({0, 1})
