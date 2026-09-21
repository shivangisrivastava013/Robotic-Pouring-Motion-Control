import torch  # noqa: F401 - Must be imported first on Windows for DLL loading
from stable_baselines3.common.env_checker import check_env

from pouring_sim.environment import RoboticPouringEnv


def test_gymnasium_env_compliance():
    env = RoboticPouringEnv()
    check_env(env)


def test_reset_and_step_shapes():
    env = RoboticPouringEnv()
    obs, info = env.reset(seed=42)

    assert obs.shape == (7,)
    assert "tilt_angle_rad" in info
    assert "poured_volume_ml" in info

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
