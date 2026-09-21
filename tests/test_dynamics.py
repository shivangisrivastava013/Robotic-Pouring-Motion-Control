import numpy as np

from pouring_sim.dynamics import PouringDynamics


def test_initial_dynamics_state():
    dyn = PouringDynamics(target_volume=200.0, initial_source_volume=500.0)
    st = dyn.get_state()
    assert st["tilt_angle_rad"] == 0.0
    assert st["poured_volume_ml"] == 0.0
    assert st["source_volume_ml"] == 500.0
    assert st["cumulative_spill_ml"] == 0.0
    assert st["flow_rate_ml_s"] == 0.0


def test_no_flow_below_threshold():
    dyn = PouringDynamics(dt=0.1)
    # Action 0.1 tilts container slightly below 45 deg threshold
    for _ in range(3):
        dyn.step(0.1)

    st = dyn.get_state()
    assert st["tilt_angle_deg"] < 45.0
    assert st["poured_volume_ml"] == 0.0
    assert st["flow_rate_ml_s"] == 0.0


def test_flow_onset_above_threshold():
    dyn = PouringDynamics(dt=0.1)
    # Action 1.0 accelerates container past 45 deg threshold
    for _ in range(15):
        dyn.step(1.0)

    st = dyn.get_state()
    assert st["tilt_angle_deg"] > 45.0
    assert st["poured_volume_ml"] > 0.0
    assert st["flow_rate_ml_s"] > 0.0


def test_volume_conservation():
    dyn = PouringDynamics(dt=0.1)
    for _ in range(20):
        dyn.step(0.8)

    st = dyn.get_state()
    total_vol = st["poured_volume_ml"] + st["source_volume_ml"] + st["cumulative_spill_ml"]
    assert np.isclose(total_vol, 500.0)
