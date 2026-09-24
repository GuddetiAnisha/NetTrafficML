"""Compare CSV/Parquet storage and loading; demonstrate chunked processing."""
from pathlib import Path
import json
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def timed_read(fn):
    t0 = time.perf_counter()
    df = fn()
    return df, time.perf_counter() - t0

def main():
    csv_path = DATA / "network_traffic.csv"
    pq_path = DATA / "network_traffic.parquet"

    csv_df, csv_s = timed_read(lambda: pd.read_csv(csv_path))
    pq_df, pq_s = timed_read(lambda: pd.read_parquet(pq_path))

    t0 = time.perf_counter()
    rows = 0
    weighted_sum = 0.0
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        rows += len(chunk)
        weighted_sum += chunk["downlink_mbps"].sum()
    chunk_s = time.perf_counter() - t0

    result = {
        "rows": rows,
        "csv_size_mb": csv_path.stat().st_size / 1024**2,
        "parquet_size_mb": pq_path.stat().st_size / 1024**2,
        "csv_read_seconds": csv_s,
        "parquet_read_seconds": pq_s,
        "chunked_csv_seconds": chunk_s,
        "chunked_mean_downlink_mbps": weighted_sum / rows,
        "dataframes_same_rows": len(csv_df) == len(pq_df),
    }
    (OUT / "io_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
