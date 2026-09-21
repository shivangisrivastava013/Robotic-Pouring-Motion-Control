import numpy as np


class PouringDynamics:
    """
    Physical dynamics model of container rotation and fluid volume flow.
    Operates strictly in physical SI / metric units:
    - Angles: Radians [rad]
    - Angular Velocity: [rad/s]
    - Angular Acceleration: [rad/s^2]
    - Volume: Milliliters [ml]
    - Flow Rate: [ml/s]
    """

    def __init__(
        self,
        target_volume: float = 200.0,
        initial_source_volume: float = 500.0,
        dt: float = 0.1,
        flow_threshold_rad: float = np.radians(45.0),
        max_tilt_rad: float = np.radians(90.0),
        max_angular_accel: float = np.radians(60.0),
        flow_coef: float = 80.0,  # ml / (rad * s)
        spill_coef: float = 10.0,  # ml / (rad/s * s)
    ):
        self.target_volume = float(target_volume)
        self.initial_source_volume = float(initial_source_volume)
        self.dt = float(dt)
        self.flow_threshold_rad = float(flow_threshold_rad)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_angular_accel = float(max_angular_accel)
        self.flow_coef = float(flow_coef)
        self.spill_coef = float(spill_coef)

        self.reset()

    def reset(self, target_volume: float | None = None, initial_source_volume: float | None = None):
        if target_volume is not None:
            self.target_volume = float(target_volume)
        if initial_source_volume is not None:
            self.initial_source_volume = float(initial_source_volume)

        self.tilt_angle = 0.0  # rad
        self.angular_velocity = 0.0  # rad/s
        self.poured_volume = 0.0  # ml
        self.source_volume = self.initial_source_volume  # ml
        self.cumulative_spill = 0.0  # ml
        self.flow_rate = 0.0  # ml/s

    def step(self, action: float):
        """
        Action: normalized angular acceleration in [-1.0, 1.0].
        """
        accel = np.clip(action, -1.0, 1.0) * self.max_angular_accel

        # Update rotational motion
        self.angular_velocity = np.clip(
            self.angular_velocity + accel * self.dt,
            -np.radians(90.0),
            np.radians(90.0),
        )
        self.tilt_angle = np.clip(
            self.tilt_angle + self.angular_velocity * self.dt,
            0.0,
            self.max_tilt_rad,
        )

        # Stop velocity if hitting boundaries
        if (
            self.tilt_angle == 0.0
            and self.angular_velocity < 0.0
            or self.tilt_angle == self.max_tilt_rad
            and self.angular_velocity > 0.0
        ):
            self.angular_velocity = 0.0

        # Calculate fluid flow
        effective_tilt = max(0.0, self.tilt_angle - self.flow_threshold_rad)

        if effective_tilt > 0.0 and self.source_volume > 0.0:
            # Controlled flow rate proportional to tilt past threshold
            raw_flow = self.flow_coef * effective_tilt
            actual_flow_delta = min(raw_flow * self.dt, self.source_volume)
            self.flow_rate = actual_flow_delta / self.dt
            self.poured_volume += actual_flow_delta
            self.source_volume -= actual_flow_delta

            # Spill calculation from excessive angular speed / splashing (>40 deg/s)
            if abs(self.angular_velocity) > np.radians(40.0):
                spill_delta = min(
                    self.spill_coef * abs(self.angular_velocity) * self.dt,
                    self.source_volume,
                )
                self.cumulative_spill += spill_delta
                self.source_volume -= spill_delta
        else:
            self.flow_rate = 0.0

        return self.get_state()

    def get_state(self):
        return {
            "tilt_angle_rad": self.tilt_angle,
            "tilt_angle_deg": np.degrees(self.tilt_angle),
            "angular_velocity_rad_s": self.angular_velocity,
            "poured_volume_ml": self.poured_volume,
            "target_volume_ml": self.target_volume,
            "source_volume_ml": self.source_volume,
            "cumulative_spill_ml": self.cumulative_spill,
            "flow_rate_ml_s": self.flow_rate,
        }
