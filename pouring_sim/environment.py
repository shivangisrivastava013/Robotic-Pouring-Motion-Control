import numpy as np
from typing import Tuple, Dict, Any


class RoboticPouringEnv:
    """
    Physics-based Reinforcement Learning Environment for Robotic Liquid Container Pouring.
    Observation: [current_tilt_angle_deg, angular_velocity, poured_volume_ml, target_volume_ml]
    Action: [angular_acceleration_rad_s2]
    """

    def __init__(self, target_volume_ml: float = 250.0, max_capacity_ml: float = 500.0):
        self.target_volume_ml = target_volume_ml
        self.max_capacity_ml = max_capacity_ml
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets environment state for new simulation episode.
        """
        self.tilt_angle = 0.0  # degrees
        self.angular_velocity = 0.0
        self.poured_volume = 0.0
        self.spilled_volume = 0.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.tilt_angle / 90.0,
            self.angular_velocity / 5.0,
            self.poured_volume / self.max_capacity_ml,
            self.target_volume_ml / self.max_capacity_ml
        ], dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one control step in simulation.
        Action: normalized angular acceleration (-1.0 to 1.0)
        """
        self.step_count += 1
        accel = np.clip(action, -1.0, 1.0) * 2.0  # deg/s2

        self.angular_velocity = np.clip(self.angular_velocity + accel * 0.1, -5.0, 5.0)
        self.tilt_angle = np.clip(self.tilt_angle + self.angular_velocity * 0.1, 0.0, 90.0)

        # Liquid flow physics approximation
        if self.tilt_angle > 45.0:
            flow_rate = (self.tilt_angle - 45.0) * 1.5  # ml/step
            if self.angular_velocity > 3.0:
                # High velocity causes spill
                spill = flow_rate * 0.2
                flow_rate *= 0.8
                self.spilled_volume += spill

            self.poured_volume = min(self.max_capacity_ml, self.poured_volume + flow_rate)

        # Reward calculation: minimize target volume error & spill penalty
        volume_error = abs(self.poured_volume - self.target_volume_ml)
        reward = - (volume_error / 10.0) - (self.spilled_volume * 2.0)

        done = self.step_count >= 50 or self.poured_volume >= self.target_volume_ml * 1.1

        info = {
            "poured_volume_ml": round(self.poured_volume, 1),
            "spilled_volume_ml": round(self.spilled_volume, 1),
            "tilt_angle_deg": round(self.tilt_angle, 1),
            "volume_accuracy_pct": round(max(0.0, 100.0 - (volume_error / self.target_volume_ml * 100.0)), 1)
        }

        return self._get_obs(), reward, done, info
