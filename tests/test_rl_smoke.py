import torch  # noqa: F401 - Must be imported first on Windows for DLL loading
from stable_baselines3 import PPO, SAC

from pouring_sim.controllers import SB3PolicyWrapper
from pouring_sim.environment import RoboticPouringEnv


def test_ppo_rl_smoke_train_and_predict():
    env = RoboticPouringEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=128)

    wrapper = SB3PolicyWrapper(model, name="PPO")
    obs, info = env.reset(seed=42)
    act = wrapper.select_action(obs, info)

    assert act.shape == (1,)


def test_sac_rl_smoke_train_and_predict():
    env = RoboticPouringEnv()
    model = SAC("MlpPolicy", env, learning_starts=10, batch_size=16, verbose=0)
    model.learn(total_timesteps=64)

    wrapper = SB3PolicyWrapper(model, name="SAC")
    obs, info = env.reset(seed=42)
    act = wrapper.select_action(obs, info)

    assert act.shape == (1,)
