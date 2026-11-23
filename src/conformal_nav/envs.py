"""Environment factory and scenario-specific success checks."""

import copy

import gymnasium as gym
import highway_env  # noqa: F401  (registers intersection-v0, merge-v0, ...)

# merge-v0 terminates once the ego passes this x-coordinate (end of the merge zone).
MERGE_END_X = 370.0


def make_env(cfg: dict, render_mode: str | None = None) -> gym.Env:
    """Build a configured highway-env environment from a loaded YAML config."""
    env_config = copy.deepcopy(cfg["env_config"])
    # Without this, highway-env opens a (frozen) pygame window even for rgb_array rendering.
    env_config.setdefault("offscreen_rendering", render_mode != "human")
    kwargs = {"config": env_config, "render_mode": render_mode}
    if cfg.get("max_episode_steps"):
        kwargs["max_episode_steps"] = cfg["max_episode_steps"]
    return gym.make(cfg["env_id"], **kwargs)


def is_success(env: gym.Env, scenario: str) -> bool:
    """Did the ego finish its task (turn completed / merge zone cleared) without crashing?"""
    base = env.unwrapped
    ego = base.vehicle
    if ego.crashed:
        return False
    if scenario == "intersection":
        return bool(base.has_arrived(ego))
    if scenario == "merge":
        return bool(ego.position[0] > MERGE_END_X)
    raise ValueError(f"Unknown scenario: {scenario}")
