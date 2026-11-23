"""PPO training entry point shared by scripts and notebooks."""

import time

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from conformal_nav.envs import is_success, make_env


class LiveDemoCallback(BaseCallback):
    """Every `every` steps, pause training and drive one real-time episode in a pygame window.

    Uses a separate env and a fixed seed (same traffic each time), so it never alters training
    and you can see the policy's behaviour change on an identical scenario.
    """

    def __init__(self, cfg: dict, every: int, seed: int):
        super().__init__()
        self.cfg, self.every, self.seed = cfg, every, seed
        self.next_demo = 0

    def _on_training_start(self) -> None:
        pygame.init()
        # highway-env maps arrow keys to manual actions, which crash envs without lane changes.
        pygame.event.set_blocked([pygame.KEYDOWN, pygame.KEYUP])
        demo_cfg = {**self.cfg, "env_config": {**self.cfg["env_config"], "real_time_rendering": True}}
        self.demo_env = make_env(demo_cfg, render_mode="human")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_demo:
            self._demo()
            self.next_demo = self.num_timesteps + self.every
        return True

    def _on_training_end(self) -> None:
        self._demo()
        self.demo_env.close()

    def _demo(self) -> None:
        obs, _ = self.demo_env.reset(seed=self.seed)
        pygame.display.set_caption(f"{self.cfg['env_id']} | PPO after {self.num_timesteps} steps")
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(self.model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, _ = self.demo_env.step(action)

        base = self.demo_env.unwrapped
        if base.vehicle.crashed:
            outcome = "CRASH"
        elif is_success(self.demo_env, self.cfg["scenario"]):
            outcome = "SUCCESS"
        else:
            outcome = "TIMEOUT"
        print(f"[live demo @ {self.num_timesteps:>6} steps] {outcome}")


def train_ppo(
    cfg: dict, total_timesteps: int, seed: int, verbose: int = 1, watch_every: int | None = None
) -> PPO:
    """Train PPO; if `watch_every` is set, show a live demo episode every that many steps."""
    # Seeds python, numpy and torch RNGs in one call.
    set_random_seed(seed)

    # Parallel in-process envs give PPO more diverse rollouts; env i is seeded with seed+i.
    train_env = make_vec_env(lambda: make_env(cfg), n_envs=cfg["train"]["n_envs"], seed=seed)

    # Small MLPs run faster on CPU than on MPS (transfer overhead dominates).
    model = PPO("MlpPolicy", train_env, seed=seed, device="cpu", verbose=verbose, **cfg["ppo"])

    t0 = time.time()
    callback = LiveDemoCallback(cfg, watch_every, seed=seed + 999) if watch_every else None
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print(f"Trained {total_timesteps} steps in {time.time() - t0:.0f}s")
    return model
