"""Software-only traffic-aware resource optimisation with PPO.

This extension simulates an energy-aware container resource controller inspired by
traffic-driven RAN resource management. It does not control real CPU frequency,
DVFS, production RAN nodes, or Ericsson hardware.

The environment exposes a daily traffic curve and lets an RL agent choose one of
four abstract resource levels. Each action changes simulated service capacity and
power draw. The reward trades off energy use, SLA latency violations, overload,
and excessive resource switching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

RESOURCE_LEVELS = np.array([0.50, 0.75, 1.00, 1.25], dtype=np.float32)


@dataclass(frozen=True)
class ControlMetrics:
    total_energy_wh: float
    mean_latency_ms: float
    sla_violations: int
    overload_steps: int
    resource_changes: int
    cumulative_reward: float


def generate_daily_traffic(
    steps: int = 288,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate one synthetic day of traffic at fixed time intervals.

    The curve contains morning and evening demand peaks plus small stochastic
    variation. It is intentionally synthetic and is not a production trace.
    """
    if steps < 24:
        raise ValueError("steps must be at least 24")

    rng = np.random.default_rng(seed)
    minutes = np.linspace(0.0, 24.0 * 60.0, steps, endpoint=False)
    hour = minutes / 60.0

    morning = 0.52 * np.exp(-0.5 * ((hour - 8.0) / 1.8) ** 2)
    midday = 0.24 * np.exp(-0.5 * ((hour - 13.0) / 2.8) ** 2)
    evening = 0.72 * np.exp(-0.5 * ((hour - 19.0) / 2.2) ** 2)
    base = 0.20 + morning + midday + evening
    noise = rng.normal(0.0, 0.025, size=steps)

    traffic_load = np.clip(base + noise, 0.10, 1.0)
    active_users = np.maximum(
        20, np.round(1200 * traffic_load + rng.normal(0, 30, steps)).astype(int)
    )

    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "hour": hour,
            "traffic_load": traffic_load.astype(np.float32),
            "active_users": active_users,
        }
    )


def estimate_system_state(
    traffic_load: float,
    resource_level: float,
) -> tuple[float, float, bool]:
    """Return simulated (latency_ms, power_w, overloaded)."""
    resource_level = float(resource_level)
    traffic_load = float(traffic_load)

    # Abstract service capacity grows with allocated compute.
    capacity = 0.82 * resource_level
    utilisation = traffic_load / max(capacity, 1e-6)

    # Latency rises sharply near and above capacity.
    base_latency = 7.5 + 8.0 * utilisation
    queue_penalty = 52.0 * max(0.0, utilisation - 0.82) ** 2
    latency_ms = base_latency + queue_penalty

    # Software model of energy cost; not a physical DVFS measurement.
    idle_power = 42.0
    dynamic_power = 58.0 * (resource_level ** 2.2) * min(utilisation, 1.15)
    power_w = idle_power + dynamic_power

    overloaded = utilisation > 1.0
    return float(latency_ms), float(power_w), bool(overloaded)


class TrafficEnergyEnv(gym.Env):
    """Gymnasium environment for software-only traffic/resource optimisation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        traffic: pd.DataFrame | None = None,
        sla_latency_ms: float = 25.0,
        seed: int = 42,
    ):
        super().__init__()
        self.traffic = (
            generate_daily_traffic(seed=seed)
            if traffic is None
            else traffic.reset_index(drop=True).copy()
        )
        if "traffic_load" not in self.traffic:
            raise ValueError("traffic must contain a traffic_load column")

        self.sla_latency_ms = float(sla_latency_ms)
        self.action_space = spaces.Discrete(len(RESOURCE_LEVELS))
        # [traffic_load, previous_resource, sin(hour), cos(hour)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.2, 1.5, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._idx = 0
        self._previous_action = 2
        self._cumulative_reward = 0.0

    def _observation(self) -> np.ndarray:
        row = self.traffic.iloc[min(self._idx, len(self.traffic) - 1)]
        hour = float(row.get("hour", 24.0 * self._idx / len(self.traffic)))
        angle = 2.0 * np.pi * hour / 24.0
        return np.array(
            [
                float(row["traffic_load"]),
                float(RESOURCE_LEVELS[self._previous_action]),
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self._previous_action = 2
        self._cumulative_reward = 0.0
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        resource = float(RESOURCE_LEVELS[action])
        load = float(self.traffic.iloc[self._idx]["traffic_load"])
        latency_ms, power_w, overloaded = estimate_system_state(load, resource)

        energy_penalty = power_w / 100.0
        sla_penalty = max(0.0, latency_ms - self.sla_latency_ms) / 10.0
        overload_penalty = 2.0 if overloaded else 0.0
        switching_penalty = 0.08 if action != self._previous_action else 0.0
        reward = -(energy_penalty + 2.5 * sla_penalty + overload_penalty + switching_penalty)

        self._previous_action = action
        self._idx += 1
        self._cumulative_reward += reward
        terminated = self._idx >= len(self.traffic)

        info = {
            "traffic_load": load,
            "resource_level": resource,
            "latency_ms": latency_ms,
            "power_w": power_w,
            "overloaded": overloaded,
            "sla_violation": latency_ms > self.sla_latency_ms,
        }

        obs = self._observation() if not terminated else np.zeros(4, dtype=np.float32)
        return obs, float(reward), terminated, False, info


def static_policy(_: float, action: int = 2) -> int:
    return int(action)


def rule_based_policy(load: float) -> int:
    if load < 0.28:
        return 0
    if load < 0.48:
        return 1
    if load < 0.72:
        return 2
    return 3


def evaluate_policy(
    traffic: pd.DataFrame,
    policy,
    sla_latency_ms: float = 25.0,
) -> tuple[ControlMetrics, pd.DataFrame]:
    """Evaluate a callable policy(load)->action on the simulated day."""
    records = []
    previous_action = 2
    cumulative_reward = 0.0

    step_minutes = 24.0 * 60.0 / len(traffic)

    for _, row in traffic.iterrows():
        load = float(row["traffic_load"])
        action = int(policy(load))
        resource = float(RESOURCE_LEVELS[action])
        latency_ms, power_w, overloaded = estimate_system_state(load, resource)

        sla_violation = latency_ms > sla_latency_ms
        reward = -(
            power_w / 100.0
            + 2.5 * max(0.0, latency_ms - sla_latency_ms) / 10.0
            + (2.0 if overloaded else 0.0)
            + (0.08 if action != previous_action else 0.0)
        )
        cumulative_reward += reward

        records.append(
            {
                "step": int(row["step"]),
                "hour": float(row["hour"]),
                "traffic_load": load,
                "action": action,
                "resource_level": resource,
                "latency_ms": latency_ms,
                "power_w": power_w,
                "overloaded": overloaded,
                "sla_violation": sla_violation,
                "reward": reward,
            }
        )
        previous_action = action

    results = pd.DataFrame(records)
    energy_wh = float((results["power_w"] * (step_minutes / 60.0)).sum())
    changes = int((results["action"].diff().fillna(0) != 0).sum())

    metrics = ControlMetrics(
        total_energy_wh=energy_wh,
        mean_latency_ms=float(results["latency_ms"].mean()),
        sla_violations=int(results["sla_violation"].sum()),
        overload_steps=int(results["overloaded"].sum()),
        resource_changes=changes,
        cumulative_reward=float(cumulative_reward),
    )
    return metrics, results


def evaluate_ppo(model, traffic: pd.DataFrame, sla_latency_ms: float = 25.0):
    env = TrafficEnergyEnv(traffic=traffic, sla_latency_ms=sla_latency_ms)
    obs, _ = env.reset()
    records = []
    previous_action = 2
    cumulative_reward = 0.0
    step_minutes = 24.0 * 60.0 / len(traffic)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, _, info = env.step(action)
        cumulative_reward += reward
        records.append(
            {
                "step": len(records),
                "hour": float(traffic.iloc[len(records)]["hour"]),
                "traffic_load": info["traffic_load"],
                "action": action,
                "resource_level": info["resource_level"],
                "latency_ms": info["latency_ms"],
                "power_w": info["power_w"],
                "overloaded": info["overloaded"],
                "sla_violation": info["sla_violation"],
                "reward": reward,
            }
        )
        previous_action = action
        if terminated:
            break

    results = pd.DataFrame(records)
    energy_wh = float((results["power_w"] * (step_minutes / 60.0)).sum())
    changes = int((results["action"].diff().fillna(0) != 0).sum())
    metrics = ControlMetrics(
        total_energy_wh=energy_wh,
        mean_latency_ms=float(results["latency_ms"].mean()),
        sla_violations=int(results["sla_violation"].sum()),
        overload_steps=int(results["overloaded"].sum()),
        resource_changes=changes,
        cumulative_reward=float(cumulative_reward),
    )
    return metrics, results


def train_ppo(
    total_timesteps: int = 20_000,
    seed: int = 42,
):
    """Train a small PPO agent using Stable-Baselines3."""
    from stable_baselines3 import PPO

    MODEL_DIR.mkdir(exist_ok=True)
    env = TrafficEnergyEnv(seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=288,
        batch_size=72,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_DIR / "ppo_traffic_energy")
    return model


def _metrics_dict(metrics: ControlMetrics) -> dict:
    return {
        "total_energy_wh": round(metrics.total_energy_wh, 3),
        "mean_latency_ms": round(metrics.mean_latency_ms, 3),
        "sla_violations": metrics.sla_violations,
        "overload_steps": metrics.overload_steps,
        "resource_changes": metrics.resource_changes,
        "cumulative_reward": round(metrics.cumulative_reward, 3),
    }


def main() -> None:
    traffic = generate_daily_traffic()

    static_metrics, static_results = evaluate_policy(
        traffic, lambda load: static_policy(load, action=2)
    )
    rule_metrics, rule_results = evaluate_policy(traffic, rule_based_policy)

    print("Static baseline:", _metrics_dict(static_metrics))
    print("Rule-based baseline:", _metrics_dict(rule_metrics))

    model = train_ppo()
    ppo_metrics, ppo_results = evaluate_ppo(model, traffic)
    print("PPO:", _metrics_dict(ppo_metrics))

    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison = pd.DataFrame(
        [
            {"policy": "static", **_metrics_dict(static_metrics)},
            {"policy": "rule_based", **_metrics_dict(rule_metrics)},
            {"policy": "ppo", **_metrics_dict(ppo_metrics)},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "rl_energy_comparison.csv", index=False)
    static_results.to_csv(OUTPUT_DIR / "rl_static_trace.csv", index=False)
    rule_results.to_csv(OUTPUT_DIR / "rl_rule_trace.csv", index=False)
    ppo_results.to_csv(OUTPUT_DIR / "rl_ppo_trace.csv", index=False)

    print(f"Saved comparison to {OUTPUT_DIR / 'rl_energy_comparison.csv'}")


if __name__ == "__main__":
    main()
