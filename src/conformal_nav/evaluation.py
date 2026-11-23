"""Policy-agnostic evaluation loop and safety/efficiency metrics."""

from dataclasses import asdict, dataclass
from typing import Callable

import gymnasium as gym
import numpy as np

from conformal_nav.envs import is_success

Policy = Callable[[np.ndarray], int]


@dataclass
class EpisodeResult:
    collided: bool
    success: bool
    timeout: bool  # ran out of time without crashing or finishing ("freezing")
    mean_speed: float
    length: int
    total_reward: float


def run_episode(policy: Policy, env: gym.Env, scenario: str, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    speeds, total_reward, steps = [], 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        speeds.append(env.unwrapped.vehicle.speed)
        total_reward += float(reward)
        steps += 1

    collided = bool(env.unwrapped.vehicle.crashed)
    success = is_success(env, scenario)
    return EpisodeResult(
        collided=collided,
        success=success,
        timeout=truncated and not collided and not success,
        mean_speed=float(np.mean(speeds)),
        length=steps,
        total_reward=total_reward,
    )


def evaluate(policy: Policy, env: gym.Env, scenario: str, n_episodes: int, seed: int) -> dict:
    """Run `n_episodes` with seeds seed, seed+1, ... so every method sees identical traffic."""
    results = [run_episode(policy, env, scenario, seed + i) for i in range(n_episodes)]
    return summarize(results)


def summarize(results: list[EpisodeResult]) -> dict:
    arr = {k: np.array([asdict(r)[k] for r in results], dtype=float) for k in asdict(results[0])}
    return {
        "episodes": len(results),
        "collision_rate": arr["collided"].mean(),
        "completion_rate": arr["success"].mean(),
        "timeout_rate": arr["timeout"].mean(),
        "avg_speed": arr["mean_speed"].mean(),
        "avg_length": arr["length"].mean(),
        "avg_return": arr["total_reward"].mean(),
    }
