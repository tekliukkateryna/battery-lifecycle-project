#python -m pytest

import numpy as np
import pytest

from backend.physics.config import (
    BatteryGeometry,
    BatteryParameters,
    SimulationConfig,
)
from backend.physics.simulation import BatterySimulation


def calculate_lithium_balance(sol, model):
    """Calculate total lithium inventory balance [C]."""

    g = model.geo
    p = model.param
    s = model.sim

    def sphere_weights(R, N):
        faces = np.linspace(0.0, R, N + 1)
        return np.diff(faces**3) / R**3

    w_n = sphere_weights(g.R_n, g.N_r_n)
    w_p = sphere_weights(g.R_p, g.N_r_p)

    c_n_avg = w_n @ sol.y[model.s_n, :]
    c_p_avg = w_p @ sol.y[model.s_p, :]

    Q_n = (
        p.F * s.A_cell * g.L_n
        * g.eps_active_n * c_n_avg
    )

    Q_p = (
        p.F * s.A_cell * g.L_p
        * g.eps_active_p * c_p_avg
    )

    Q_loss = sol.y[model.i_Q, :]

    return Q_n, Q_p, Q_loss


@pytest.mark.parametrize(
    "current, enable_sei",
    [
        (0.25, False),
        (-0.25, False),
        (0.25, True),
        (-0.25, True),
    ]
)
def test_lithium_conservation(current, enable_sei):

    geo = BatteryGeometry()
    param = BatteryParameters()
    sim = SimulationConfig()

    sim.I_app = current


    model = BatterySimulation(
        geo,
        param,
        sim,
        enable_sei=enable_sei,
        enable_thermal=False,
    )

    sol = model.run(
        t_final=600.0,
        num_points=21,
    )

    Q_n, Q_p, Q_loss = calculate_lithium_balance(
        sol, model
    )

    Q_total = Q_n + Q_p + Q_loss
    error = Q_total - Q_total[0]

    # Conservation tolerance [C]
    assert np.max(np.abs(error)) < 1e-7

    # Check that lithium actually moves between electrodes
    assert abs(Q_n[-1] - Q_n[0]) > 1e-5

    # Check SEI inventory loss
    if enable_sei:
        assert Q_loss[-1] > Q_loss[0]
    else:
        assert np.all(Q_loss == 0.0)