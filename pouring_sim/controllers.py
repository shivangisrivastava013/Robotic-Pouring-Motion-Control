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
        else:
            tilt_deg = info["tilt_angle_deg"]
            poured = info["poured_volume_ml"]
            target = info["target_volume_ml"]

        # If close to target volume, rotate back quickly
        if poured >= target * 0.90:
            return np.array([-0.8], dtype=np.float32)
        # If tilt below flow threshold (45 deg), tilt forward
        elif tilt_deg < 48.0:
            return np.array([0.5], dtype=np.float32)
        # Hold angle near threshold
        else:
            return np.array([0.0], dtype=np.float32)

    def reset(self):
        pass


class PIDController:
    """
    Feedback PID controller managing volume error and angular velocity.
    """

    def __init__(self, kp: float = 2.5, ki: float = 0.05, kd: float = 0.8, dt: float = 0.1):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.name = "PID"
        self.reset()

    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def select_action(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        if info is None:
            # Reconstruct approximately from normalized obs
            poured_ratio = obs[2]
            target_vol = obs[3] * 500.0
            error = max(0.0, 1.0 - poured_ratio)
            tilt_deg = obs[0] * 90.0
        else:
            target_vol = info["target_volume_ml"]
            poured_vol = info["poured_volume_ml"]
            error = (target_vol - poured_vol) / max(1.0, target_vol)
            tilt_deg = info["tilt_angle_deg"]

        if error <= 0.05 or (info and info["poured_volume_ml"] >= target_vol * 0.92):
            # Target reached: command strong upright rotation
            self.reset()
            return np.array([-0.9], dtype=np.float32)

        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Compute control output
        desired_tilt = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        desired_tilt_deg = min(75.0, max(0.0, desired_tilt * 60.0))

        # Action based on error between desired and current tilt
        tilt_error = desired_tilt_deg - tilt_deg
        action = np.clip(tilt_error / 15.0, -1.0, 1.0)
        return np.array([action], dtype=np.float32)


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
