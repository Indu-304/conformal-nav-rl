# conformal-nav-rl

Safe reinforcement learning for autonomous driving at unsignalized intersections and highway merges,
built on [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

The project studies how learned driving policies trade off safety and efficiency, and how
uncertainty-aware safety filtering based on conformal prediction can reduce collisions without
making the agent freeze.

## Scenarios

| Scenario | Env | Task | Actions |
|---|---|---|---|
| Intersection | `intersection-v0` | Unprotected left turn through cross-traffic | `SLOWER`, `IDLE`, `FASTER` |
| Merge | `merge-v0` | Highway driving past an on-ramp with merging traffic | lane changes + speed |

Each episode ends in exactly one outcome: **collision**, **completion** (turn finished / merge zone
cleared), or **timeout** (the agent stalled until the time limit).

## Setup

Requires Python 3.10–3.13.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Usage

```bash
python scripts/run_baseline.py                        # train + evaluate PPO on the intersection
python scripts/run_baseline.py --scenario merge
python scripts/run_baseline.py --watch-every 1000     # live demo window during training
```

Models and metrics are written to `runs/<scenario>/`. The notebook
[`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) walks through environment inspection,
reference policies, training, evaluation and GIF replays.

All runs are seeded. Evaluation seeds are disjoint from training seeds, and every policy is evaluated
on the same set of traffic scenarios.

## Results

PPO, 5k training steps, 20 evaluation episodes, seed 42:

| Scenario | Policy | Collision | Completion | Timeout | Avg speed (m/s) |
|---|---|---|---|---|---|
| Intersection | Always `SLOWER` | 0% | 0% | 100% | 0.5 |
| Intersection | Always `FASTER` | 30% | 70% | 0% | 8.8 |
| Intersection | PPO | 30% | 70% | 0% | 8.8 |
| Merge | Always `IDLE` | 100% | 0% | 0% | 27.9 |
| Merge | PPO | 20% | 80% | 0% | 20.3 |

On the intersection, PPO converges to the aggressive "always accelerate" policy: under the default
reward, the speed bonus outweighs the expected collision penalty. This reward-driven risk-taking is
the failure mode that motivates an explicit safety layer.

## Project structure

```
configs/            scenario, environment and PPO hyperparameters (YAML)
src/conformal_nav/
  envs.py           environment factory and per-scenario success checks
  evaluation.py     policy-agnostic evaluation loop and metrics
  training.py       PPO training with optional live demo callback
  viz.py            episode recording to GIF
scripts/            command-line entry points
notebooks/          interactive walkthroughs
tests/              pytest suite
```

## License

MIT
