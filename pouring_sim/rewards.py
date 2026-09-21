class PouringRewardFunction:
    """
    Multi-component reward function for robotic precision pouring.
    Includes explicit penalty breakdown in info dictionary.
    """

    def __init__(
        self,
        weight_error: float = 1.0,
        weight_spill: float = 2.0,
        weight_smoothness: float = 0.05,
        weight_overshoot: float = 3.0,
        success_bonus: float = 10.0,
        success_threshold_ml: float = 10.0,
    ):
        self.weight_error = float(weight_error)
        self.weight_spill = float(weight_spill)
        self.weight_smoothness = float(weight_smoothness)
        self.weight_overshoot = float(weight_overshoot)
        self.success_bonus = float(success_bonus)
        self.success_threshold_ml = float(success_threshold_ml)

    def calculate(self, state: dict, action: float, is_terminal: bool) -> tuple[float, dict]:
        poured = state["poured_volume_ml"]
        target = state["target_volume_ml"]
        spill = state["cumulative_spill_ml"]
        ang_vel = state["angular_velocity_rad_s"]

        volume_error = abs(poured - target)
        overshoot = max(0.0, poured - (target + 15.0))

        volume_error_penalty = -self.weight_error * (volume_error / max(1.0, target))
        spill_penalty = -self.weight_spill * (spill / max(1.0, target))
        smoothness_penalty = -self.weight_smoothness * (abs(ang_vel) + abs(action))
        overshoot_penalty = -self.weight_overshoot * (overshoot / max(1.0, target))

        success = is_terminal and (volume_error <= 15.0) and (spill <= 20.0)
        success_bonus = self.success_bonus if success else 0.0

        total_reward = volume_error_penalty + spill_penalty + smoothness_penalty + overshoot_penalty + success_bonus

        reward_info = {
            "volume_error_penalty": float(volume_error_penalty),
            "spill_penalty": float(spill_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "overshoot_penalty": float(overshoot_penalty),
            "success_bonus": float(success_bonus),
            "is_success": bool(success),
        }

        return float(total_reward), reward_info
