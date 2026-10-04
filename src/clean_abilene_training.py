from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "abilene_traffic.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "abilene_traffic_clean.csv"


def main():
    print("=" * 70)
    print("CLEANING ABILENE TRAFFIC DATA")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nOriginal rows: {len(df):,}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["total_traffic"] = pd.to_numeric(df["total_traffic"], errors="coerce")
    df = df.dropna(subset=["timestamp", "week", "total_traffic"]).copy()
    df = df.sort_values(["week", "timestamp"]).reset_index(drop=True)

    p999 = df["total_traffic"].quantile(0.999)
    extreme_threshold = p999 * 10
    extreme_mask = df["total_traffic"] > extreme_threshold
    extreme_rows = df.loc[extreme_mask, ["timestamp", "week", "total_traffic"]]

    print(f"\n99.9 percentile : {p999:,.2f}")
    print(f"Extreme threshold: {extreme_threshold:,.2f}")
    print(f"Extreme rows found: {extreme_mask.sum()}")

    if len(extreme_rows) > 0:
        print("\nRows removed:")
        print(extreme_rows.to_string(index=False))

    df = df.loc[~extreme_mask].copy()

    columns_to_remove = [
        "target_next_traffic",
        "traffic_lag_1",
        "traffic_lag_2",
        "traffic_lag_3",
        "traffic_lag_6",
        "traffic_lag_12",
        "traffic_roll_12",
        "traffic_roll_36",
    ]
    columns_to_remove = [c for c in columns_to_remove if c in df.columns]
    df = df.drop(columns=columns_to_remove)
    df = df.sort_values(["week", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("week", sort=False)["total_traffic"]
    df["traffic_lag_1"] = grouped.shift(1)
    df["traffic_lag_2"] = grouped.shift(2)
    df["traffic_lag_3"] = grouped.shift(3)
    df["traffic_lag_6"] = grouped.shift(6)
    df["traffic_lag_12"] = grouped.shift(12)

    df["traffic_roll_12"] = df.groupby("week", sort=False)["total_traffic"].transform(
        lambda s: s.shift(1).rolling(window=12, min_periods=1).mean()
    )
    df["traffic_roll_36"] = df.groupby("week", sort=False)["total_traffic"].transform(
        lambda s: s.shift(1).rolling(window=36, min_periods=1).mean()
    )

    df["target_next_traffic"] = df.groupby("week", sort=False)["total_traffic"].shift(-1)

    before_target_drop = len(df)
    df = df.dropna(subset=["target_next_traffic"]).copy()
    boundary_rows_removed = before_target_drop - len(df)
    print(f"\nWeek-boundary rows removed: {boundary_rows_removed}")

    wrong_boundaries = 0
    for _, group in df.groupby("week", sort=False):
        if len(group) < 2:
            continue
        expected = group["total_traffic"].shift(-1)
        comparison = pd.DataFrame(
            {"expected": expected, "actual": group["target_next_traffic"]}
        ).dropna()
        wrong_boundaries += (
            ~np.isclose(
                comparison["expected"],
                comparison["actual"],
                rtol=1e-9,
                atol=1e-6,
            )
        ).sum()

    print(f"Target consistency errors: {wrong_boundaries}")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("CLEAN DATASET CREATED")
    print("=" * 70)
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Minimum traffic: {df['total_traffic'].min():,.2f}")
    print(f"Median traffic: {df['total_traffic'].median():,.2f}")
    print(f"Maximum traffic: {df['total_traffic'].max():,.2f}")
    print(f"\nSaved to:\n{OUTPUT_FILE}")
    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()
