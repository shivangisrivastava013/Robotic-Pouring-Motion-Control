from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pouring_sim.dynamics import PouringDynamics
from pouring_sim.rewards import PouringRewardFunction


class RoboticPouringEnv(gym.Env):
    """
    Gymnasium environment for precision robotic pouring motion control.
    Complies fully with standard Gymnasium API specifications.
    """

    metadata: ClassVar[dict] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None, target_volume: float = 200.0, max_steps: int = 100):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target_volume = float(target_volume)

        # Action Space: Angular acceleration in [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation Space: 7 normalized physical state features
        # [tilt_normalized, velocity_normalized, poured_ratio, target_ratio, source_ratio, spill_ratio, flow_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.dynamics = PouringDynamics(target_volume=self.target_volume)
        self.reward_fn = PouringRewardFunction()
        self.current_step = 0

    def _get_obs(self) -> np.ndarray:
        st = self.dynamics.get_state()
        obs = np.array(
            [
                st["tilt_angle_rad"] / self.dynamics.max_tilt_rad,
                st["angular_velocity_rad_s"] / np.radians(90.0),
                st["poured_volume_ml"] / max(1.0, st["target_volume_ml"]),
                st["target_volume_ml"] / 500.0,
                st["source_volume_ml"] / 500.0,
                st["cumulative_spill_ml"] / 500.0,
                st["flow_rate_ml_s"] / 100.0,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        target_vol = self.target_volume
        if options and "target_volume" in options:
            target_vol = float(options["target_volume"])
        elif seed is not None and options and options.get("randomize_target", False):
            rng = np.random.default_rng(seed)
            target_vol = float(rng.choice([100.0, 150.0, 200.0, 250.0, 300.0]))

        self.dynamics.reset(target_volume=target_vol)
        self.current_step = 0

        obs = self._get_obs()
        info = self.dynamics.get_state()
        info["step"] = 0
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if isinstance(action, (list, tuple, np.ndarray)):
            act_val = float(action[0])
        else:
            act_val = float(action)

        self.current_step += 1
        state = self.dynamics.step(act_val)

        # Check termination & truncation
        terminated = bool(
            state["poured_volume_ml"] >= state["target_volume_ml"] + 50.0
            or state["cumulative_spill_ml"] >= 100.0
            or state["source_volume_ml"] <= 0.0
        )
        truncated = bool(self.current_step >= self.max_steps)

        reward, reward_info = self.reward_fn.calculate(state, act_val, is_terminal=(terminated or truncated))

        info = {**state, **reward_info, "step": self.current_step}
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # Simple synthetic render canvas for video/GIF generation
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (400, 300), color=(240, 245, 250))
            draw = ImageDraw.Draw(img)

            st = self.dynamics.get_state()
            tilt_deg = st["tilt_angle_deg"]
            poured = st["poured_volume_ml"]
            target = st["target_volume_ml"]
            spill = st["cumulative_spill_ml"]

            # Draw target container
            draw.rectangle([250, 150, 350, 270], outline=(40, 40, 40), width=3)
            fill_height = min(110, int(110 * (poured / max(1.0, target))))
            if fill_height > 0:
                draw.rectangle([253, 267 - fill_height, 347, 267], fill=(50, 150, 240))

            # Draw source container rotated
            draw.text((20, 20), f"Tilt: {tilt_deg:.1f} deg", fill=(0, 0, 0))
            draw.text((20, 40), f"Poured: {poured:.1f} / {target:.1f} ml", fill=(0, 0, 0))
            draw.text((20, 60), f"Spill: {spill:.1f} ml", fill=(200, 0, 0) if spill > 0 else (0, 0, 0))

            return np.array(img)
        return None
