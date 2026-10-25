
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

    V_max = 4.3
    V_min = 3.0
    current_density = 0.5

    data = model.run_cycles(
        n_cycles=2,
        charge_time=7200.0,
        discharge_time=7200.0,
        current_density=current_density,
        V_max=V_max,
        V_min=V_min,
    )

    # 1. Numerical results must be finite
    assert np.all(np.isfinite(data["voltage_V"]))
    assert np.all(np.isfinite(data["sei_thickness_m"]))
    assert np.all(np.isfinite(data["capacity_loss_Ah"]))

    # 2. Total duration must not exceed 4 hours
    assert 0 < data["time_s"][-1] <= 28800.0 + 1e-6

    # 3. Time must be non-decreasing
    assert np.all(np.diff(data["time_s"]) >= -1e-9)

    # 4. Check voltage cutoffs in each phase
    phases = [
        ("charge", -current_density, V_max),
        ("discharge", current_density, V_min),
    ]

    for cycle in (1, 2):
        for phase, current, cutoff in phases:

            mask = (
                (data["cycle"] == cycle)
                & (data["phase"] == phase)
            )

            assert np.any(mask)

            times = data["time_s"][mask]
            voltages = data["voltage_V"][mask]
            currents = data["current_A_m2"][mask]

            duration = times[-1] - times[0]

            assert 0 < duration <= 7200.0 + 1e-6

            assert voltages[-1] == pytest.approx(
                cutoff, abs=1e-4
            )

            assert np.allclose(currents, current)

    # 5. SEI must grow
    assert (
        data["sei_thickness_m"][-1]
        > param.L_sei_init
    )

    # 6. Lithium loss must accumulate
    assert data["capacity_loss_Ah"][-1] > 0.0

    # 7. Check measured discharge capacities
    summary = data["cycle_summary"]

    assert len(summary["discharge_capacity_Ah"]) == 2
    assert np.all(summary["discharge_capacity_Ah"] > 0)

    assert summary["SOH_percent"][0] == pytest.approx(100.0)
