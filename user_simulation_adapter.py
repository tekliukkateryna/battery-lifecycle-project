"""Connect the unmodified SPMe simulation_cccc/cccv files to the FastAPI pipeline.

This module runs ONLY full lifecycle simulations. It does not invent per-frame
SPMe telemetry, estimate SOC, or retrain ML.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from backend.physics.config import BatteryGeometry, BatteryParameters, SimulationConfig


def run_lifecycle(payload: dict[str, Any]) -> dict[str, Any]:
    p = dict(payload)
    if float(p.get("initial_soc_percent", 100.0)) != 100.0:
        raise ValueError(
            "This physical snapshot has fixed initial electrode stoichiometries; "
            "only initial_soc_percent=100 is supported. Displayed 3D HUD 65% is a mock value."
        )

    temp_c = float(p.get("temperature_C", 25.0))
    temperature_k = temp_c + 273.15
    n_cycles = int(p.get("n_cycles", 1))
    discharge_a = float(p.get("current", 5.0))
    charge_a = float(p.get("charge_current_A") or discharge_a)
    vmax = float(p.get("upper_cutoff_V", 4.2))
    vmin = float(p.get("lower_cutoff_V", 2.5))
    mode = p.get("cycling_mode", "CC")
    if mode not in ("CC", "CCCV"):
        raise ValueError("cycling_mode must be 'CC' or 'CCCV'")
    if n_cycles < 1 or min(charge_a, discharge_a) <= 0:
        raise ValueError("Cycles and currents must be positive")
    if vmin >= vmax:
        raise ValueError("lower_cutoff_V must be < upper_cutoff_V")

    geo, params, cfg = BatteryGeometry(), BatteryParameters(), SimulationConfig()
    cfg.T_init = temperature_k
    params.T_amb = temperature_k
    cfg.I_app = discharge_a
    cfg.I_charge = charge_a
    cfg.I_discharge = discharge_a
    cfg.upper_cut_off_V = vmax
    cfg.lower_cut_off_V = vmin

    if mode == "CCCV":
        from backend.physics.simulation_cccv import BatterySimulation
    else:
        from backend.physics.simulation_cccc import BatterySimulation

    model = BatterySimulation(
        geo, params, cfg, enable_sei=True,
        enable_thermal=bool(p.get("enable_thermal", True)),
    )
    settings = {
        "n_cycles": n_cycles,
        "I_charge": charge_a,
        "I_discharge": discharge_a,
        "V_max": vmax,
        "V_min": vmin,
        "charge_time": 3600.0 * float(p.get("max_charge_hours", 8.0)),
        "discharge_time": 3600.0 * float(p.get("max_discharge_hours", 8.0)),
        "points_per_phase": int(p.get("points_per_phase", 11)),
        "pre_discharge": True,
    }
    if mode == "CCCV":
        cutoff = float(p.get("cv_current_cutoff_A", 0.05))
        if cutoff >= charge_a:
            raise ValueError("cv_current_cutoff_A must be smaller than charging current")
        settings.update(
            enable_cv=True,
            cv_time=3600.0 * float(p.get("max_cv_hours", 4.0)),
            I_cv_cutoff=cutoff,
        )

    result = model.run_cycles(**settings)
    summaries = result["cycle_summary"]
    cycle = np.asarray(summaries["cycle"], dtype=int)
    cycles_dense = np.asarray(result["cycle"], dtype=int)
    phases = np.asarray(result["phase"])
    sei_m = np.asarray(result["sei_thickness_m"], dtype=float)
    q_loss = np.asarray(result["capacity_loss_Ah"], dtype=float)

    sei_by_cycle, q_by_cycle = [], []
    for c in cycle:
        locations = np.flatnonzero((cycles_dense == c) & (phases == "discharge"))
        if len(locations) == 0:
            raise RuntimeError(f"No discharge samples for completed cycle {c}")
        ix = int(locations[-1])
        sei_by_cycle.append(float(sei_m[ix] * 1e9))
        q_by_cycle.append(float(q_loss[ix]))

    return {
        "cycle": cycle.tolist(),
        "discharge_capacity_Ah": np.asarray(summaries["discharge_capacity_Ah"], dtype=float).tolist(),
        "SOH_percent": np.asarray(summaries["SOH_percent"], dtype=float).tolist(),
        "sei_thickness_nm": sei_by_cycle,
        "initial_sei_thickness_nm": float(params.L_sei_init * 1e9),
        "capacity_loss_ah": q_by_cycle,
    }
