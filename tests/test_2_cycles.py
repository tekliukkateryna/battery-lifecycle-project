import numpy as np
import numpy as np
import pytest

from backend.physics.config import (
    BatteryGeometry,
    BatteryParameters,
    SimulationConfig,
)
from backend.physics.simulation import BatterySimulation


def test_charge_discharge_cycles():

    geo = BatteryGeometry()
    param = BatteryParameters()
    sim = SimulationConfig()

    model = BatterySimulation(
        geo,
        param,
        sim,
        enable_sei=True,
        enable_thermal=False,
    )

    data = model.run_cycles(
        n_cycles=2,
        charge_time=3600.0,
        discharge_time=3600.0,
        current_density=2.5,
    )

    # all data is finite
    assert np.all(np.isfinite(data["voltage_V"]))

    # Two complete cycles = 4 hours
    assert data["time_s"][-1] == pytest.approx(14400.0)

    # SEI must grow
    assert (
        data["sei_thickness_m"][-1]
        > param.L_sei_init
    )

    # Lithium loss must accumulate
    assert data["capacity_loss_Ah"][-1] > 0.0

    # Charge and discharge must both occur
    assert np.any(data["current_A_m2"] < 0)
    assert np.any(data["current_A_m2"] > 0)
