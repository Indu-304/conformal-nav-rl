"""Train a PPO baseline and report safety/efficiency metrics.

Usage:
    python scripts/run_baseline.py                          # intersection, defaults from YAML
    python scripts/run_baseline.py --scenario merge
    python scripts/run_baseline.py --timesteps 5000 --eval-episodes 20   # fast smoke run
    python scripts/run_baseline.py --watch-every 1000       # live demo window during training
"""

import argparse
import json
from pathlib import Path

from conformal_nav.config import load_config
from conformal_nav.envs import make_env
from conformal_nav.evaluation import evaluate
from conformal_nav.training import train_ppo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="intersection", choices=["intersection", "merge"])
    p.add_argument("--config", default=None, help="Explicit YAML path (overrides --scenario)")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--watch-every", type=int, default=None, help="Live pygame demo every N steps")
    p.add_argument("--out", default="runs")
    return p.parse_args()


def print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ({m['episodes']} episodes) ===")
    print(f"  collision rate : {m['collision_rate']:6.1%}")
    print(f"  completion rate: {m['completion_rate']:6.1%}")
    print(f"  timeout rate   : {m['timeout_rate']:6.1%}")
    print(f"  avg speed      : {m['avg_speed']:6.2f} m/s")
    print(f"  avg ep length  : {m['avg_length']:6.1f} steps")
    print(f"  avg return     : {m['avg_return']:6.2f}")


def main():
    args = parse_args()
    cfg = load_config(args.config or args.scenario)
    scenario = cfg["scenario"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    timesteps = args.timesteps or cfg["train"]["total_timesteps"]
    n_eval = args.eval_episodes or cfg["eval"]["n_episodes"]

    model = train_ppo(cfg, total_timesteps=timesteps, seed=seed, watch_every=args.watch_every)

    out_dir = Path(args.out) / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"ppo_seed{seed}")

    eval_env = make_env(cfg)
    eval_seed = seed + cfg["eval"]["seed_offset"]
    metrics = evaluate(
        policy=lambda obs: int(model.predict(obs, deterministic=True)[0]),
        env=eval_env,
        scenario=scenario,
        n_episodes=n_eval,
        seed=eval_seed,
    )
    print_metrics(f"PPO baseline | {cfg['env_id']}", metrics)

    with open(out_dir / f"metrics_ppo_seed{seed}.json", "w") as f:
        json.dump({"seed": seed, "eval_seed": eval_seed, "timesteps": timesteps, **metrics}, f, indent=2)
    print(f"\nSaved model + metrics to {out_dir}/")


if __name__ == "__main__":
    main()
