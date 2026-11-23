import numpy as np
import pytest

from conformal_nav.config import load_config
from conformal_nav.envs import make_env
from conformal_nav.evaluation import evaluate


@pytest.mark.parametrize("scenario", ["intersection", "merge"])
def test_reset_is_reproducible(scenario):
    cfg = load_config(scenario)
    env = make_env(cfg)
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    assert env.observation_space.contains(obs_a)


def test_intersection_action_space():
    env = make_env(load_config("intersection"))
    assert env.action_space.n == 3  # SLOWER, IDLE, FASTER


@pytest.mark.parametrize("scenario", ["intersection", "merge"])
def test_evaluate_outcomes_are_consistent(scenario):
    cfg = load_config(scenario)
    env = make_env(cfg)
    m = evaluate(lambda obs: 1, env, scenario, n_episodes=2, seed=0)  # constant IDLE policy
    assert m["episodes"] == 2
    for key in ("collision_rate", "completion_rate", "timeout_rate"):
        assert 0.0 <= m[key] <= 1.0
    assert m["collision_rate"] + m["completion_rate"] + m["timeout_rate"] <= 1.0 + 1e-9
