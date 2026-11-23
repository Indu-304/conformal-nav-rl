"""Episode replay recording (GIF) for highway-env scenarios."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from conformal_nav.evaluation import Policy


class _FrameCollector:
    """Plugs into highway-env's video hook to grab intermediate simulation frames.

    The agent acts at 1 Hz but physics runs at 15 Hz; without this, replays jump 1 s per frame.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self.frames: list[np.ndarray] = []
        self.frames_per_sec = None  # set by highway-env
        env.unwrapped.set_record_video_wrapper(self)

    def _capture_frame(self):
        self.frames.append(self.env.unwrapped.render())


def record_episode(policy: Policy, env: gym.Env, seed: int) -> list[np.ndarray]:
    """Roll out one episode and return every simulation frame. `env` needs render_mode='rgb_array'."""
    collector = _FrameCollector(env)
    obs, _ = env.reset(seed=seed)
    collector.frames.append(env.render())
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        # Final frame of each step; also re-arms auto-rendering for the next step.
        collector.frames.append(env.render())
    return collector.frames


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
    return path
