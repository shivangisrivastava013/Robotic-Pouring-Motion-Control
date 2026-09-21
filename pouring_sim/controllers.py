import numpy as np


class ConstantActionController:
    """
    Naive baseline controller applying constant positive acceleration.
    """

    def __init__(self, action_value: float = 0.3):
        self.action_value = float(action_value)
        self.name = "Constant-Action"

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        return np.array([self.action_value], dtype=np.float32)

    def reset(self):
        pass


class RuleBasedController:
    """
    Heuristic rule-based controller tilting up to threshold, holding, and righting on target.
    """

    def __init__(self, target_volume: float = 200.0):
        self.name = "Rule-Based"
        self.target_volume = float(target_volume)

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        if info is None:
            tilt_deg = obs[0] * 90.0
            poured = obs[2] * self.target_volume
            target = self.target_volume
            flow_rate = obs[6] * 100.0
        else:
            tilt_deg = info["tilt_angle_deg"]
            poured = info["poured_volume_ml"]
            target = info["target_volume_ml"]
            flow_rate = info["flow_rate_ml_s"]

        predicted_lead = flow_rate * 0.45

        # If close to target volume including deceleration lead, rotate back quickly
        if (poured + predicted_lead) >= (target - 15.5):
            return np.array([-1.0], dtype=np.float32)
        # If tilt below flow threshold (45 deg), tilt forward
        elif tilt_deg < 52.0:
            return np.array([0.4], dtype=np.float32)
        # Hold angle near threshold
        else:
            return np.array([0.0], dtype=np.float32)

    def reset(self):
        pass


class PIDController:
    """
    Feedback PID controller managing volume error and angular velocity with predictive lead.
    """

    def __init__(
        self, kp: float = 2.5, ki: float = 0.05, kd: float = 0.8, dt: float = 0.1, target_volume: float = 200.0
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.target_volume = float(target_volume)
        self.name = "PID"
        self.reset()

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        if info is None:
            tilt_deg = obs[0] * 90.0
            ang_vel = obs[1] * 3.0
            poured = obs[2] * self.target_volume
            target = self.target_volume
            flow_rate = obs[6] * 100.0
        else:
            tilt_deg = info["tilt_angle_deg"]
            ang_vel = info["angular_velocity_rad_s"]
            poured = info["poured_volume_ml"]
            target = info["target_volume_ml"]
            flow_rate = info["flow_rate_ml_s"]

        predicted_lead = flow_rate * 0.45

        if (poured + predicted_lead) >= (target - 15.5):
            return np.array([-1.0], dtype=np.float32)
        elif tilt_deg < 52.0:
            tilt_error = (52.0 - tilt_deg) / 52.0
            action = np.clip(self.kp * tilt_error - self.kd * ang_vel, 0.0, 0.5)
            return np.array([action], dtype=np.float32)
        else:
            return np.array([0.0], dtype=np.float32)


class SB3PolicyWrapper:
    """
    Wrapper interface for Stable-Baselines3 policies (PPO, SAC).
    """

    def __init__(self, model, name: str = "RL-Model"):
        self.model = model
        self.name = name

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def reset(self):
        pass
