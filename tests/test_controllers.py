import numpy as np

from pouring_sim.controllers import ConstantActionController, PIDController, RuleBasedController
from pouring_sim.environment import RoboticPouringEnv


def test_constant_action_controller():
    ctrl = ConstantActionController(0.4)
    obs = np.zeros(7, dtype=np.float32)
    act = ctrl.select_action(obs)
    assert act.shape == (1,)
    assert np.isclose(act[0], 0.4)


def test_rule_based_controller():
    ctrl = RuleBasedController(target_volume=200.0)
    obs = np.zeros(7, dtype=np.float32)
    act = ctrl.select_action(obs)
    assert act.shape == (1,)
    assert -1.0 <= act[0] <= 1.0


def test_pid_controller():
    env = RoboticPouringEnv(target_volume=200.0)
    ctrl = PIDController()
    obs, info = env.reset(seed=42)

    for _ in range(5):
        act = ctrl.select_action(obs, info)
        assert act.shape == (1,)
        assert -1.0 <= act[0] <= 1.0
        obs, _reward, _terminated, _truncated, info = env.step(act)
