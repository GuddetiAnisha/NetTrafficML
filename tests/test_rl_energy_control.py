import numpy as np

from src.rl_energy_control import (
    RESOURCE_LEVELS,
    TrafficEnergyEnv,
    estimate_system_state,
    evaluate_policy,
    generate_daily_traffic,
    rule_based_policy,
)


def test_daily_traffic_curve_schema_and_range():
    df = generate_daily_traffic(steps=288, seed=5)
    assert list(df.columns) == ["step", "hour", "traffic_load", "active_users"]
    assert len(df) == 288
    assert df["traffic_load"].between(0.10, 1.0).all()
    assert (df["active_users"] > 0).all()


def test_more_resources_reduce_latency_for_same_load():
    low_latency, _, _ = estimate_system_state(0.75, float(RESOURCE_LEVELS[0]))
    high_latency, _, _ = estimate_system_state(0.75, float(RESOURCE_LEVELS[-1]))
    assert high_latency < low_latency


def test_more_resources_have_energy_cost():
    _, low_power, _ = estimate_system_state(0.40, float(RESOURCE_LEVELS[0]))
    _, high_power, _ = estimate_system_state(0.40, float(RESOURCE_LEVELS[-1]))
    assert high_power > low_power


def test_environment_runs_one_full_day():
    traffic = generate_daily_traffic(steps=48, seed=7)
    env = TrafficEnergyEnv(traffic=traffic)
    obs, _ = env.reset()
    assert obs.shape == (4,)

    done = False
    count = 0
    while not done:
        obs, reward, done, truncated, info = env.step(rule_based_policy(info["traffic_load"]) if count else 2)
        assert np.isfinite(reward)
        assert not truncated
        count += 1
    assert count == len(traffic)


def test_rule_based_policy_evaluation_returns_metrics():
    traffic = generate_daily_traffic(steps=96, seed=8)
    metrics, trace = evaluate_policy(traffic, rule_based_policy)
    assert metrics.total_energy_wh > 0
    assert metrics.mean_latency_ms > 0
    assert len(trace) == 96
    assert {"resource_level", "latency_ms", "power_w", "reward"}.issubset(trace.columns)
