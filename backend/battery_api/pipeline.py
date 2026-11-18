"""SPMe -> canonical cycle arrays -> Ridge correction -> hybrid JSON."""
from typing import Any
import json
from pathlib import Path
import numpy as np

from .ridge import predict_residual, model_path
from .simulator import run_physics


def _numbers(value: Any, label: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{label} must be a non-empty 1D array")
    if length is not None and len(result) != length:
        raise ValueError(f"{label} length {len(result)} != cycle length {length}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite numbers")
    return result


def _scalar(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def run_full_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    raw = run_physics(params)
    if "cycle_summary" in raw:
        raw = raw["cycle_summary"]
    if not isinstance(raw, dict):
        raise TypeError("cycle_summary must be a dict of same-length arrays")
    required = {"cycle", "discharge_capacity_Ah", "SOH_percent", "sei_thickness_nm", "initial_sei_thickness_nm"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"SPMe adapter output missing {sorted(missing)}. "
            "Map your BatterySimulation.run_cycles() keys in user_simulation_adapter.py."
        )

    cycle = _numbers(raw["cycle"], "cycle")
    length = len(cycle)
    if np.any(cycle <= 0) or np.any(cycle != np.floor(cycle)) or np.any(np.diff(cycle) <= 0):
        raise ValueError("cycle must contain strictly increasing positive integers")
    capacity = _numbers(raw["discharge_capacity_Ah"], "discharge_capacity_Ah", length)
    physical = _numbers(raw["SOH_percent"], "SOH_percent", length)
    sei = _numbers(raw["sei_thickness_nm"], "sei_thickness_nm", length)
    initial_sei = _scalar(raw["initial_sei_thickness_nm"], "initial_sei_thickness_nm")
    if initial_sei < 0 or np.any(sei < 0):
        raise ValueError("SEI thickness cannot be negative")
    sei_growth = sei - initial_sei

    loss = None
    if "capacity_loss_ah" in raw and raw["capacity_loss_ah"] is not None:
        loss = _numbers(raw["capacity_loss_ah"], "capacity_loss_ah", length)

    warnings = []
    if np.any(sei_growth < -1e-7):
        warnings.append("SEI thickness dropped below its initial value; check the physics and signs.")
    if np.any((physical < 0) | (physical > 100)):
        warnings.append("Physical SoH outside [0,100]%; values were preserved for diagnostics.")
    if int(cycle[-1]) != params["n_cycles"]:
        warnings.append(
            "Physics returned fewer cycles than requested (possibly voltage cutoff or solver stop)."
        )

    correction = None
    hybrid = None
    ml_status = "not_requested"
    if params["apply_ml"]:
        correction = predict_residual(cycle, params["temperature_C"])
        hybrid = physical + correction
        meta_file = model_path().with_suffix(".metadata.json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        trained_temps = meta.get("temperatures_C", [10, 25, 40])
        within_observed_range = (
            min(trained_temps) <= params["temperature_C"] <= max(trained_temps)
            and int(cycle[-1]) <= meta.get("cycle_max", 936)
        )
        if params["protocol_id"] == "imperial_expt5" and within_observed_range:
            ml_status = "declared_training_protocol"
            warnings.append(
                "ML applicability assumes current, SoC window, and cycling protocol truly match "
                "Imperial College Expt 5; the Ridge features do not include those inputs."
            )
        else:
            ml_status = "exploratory_extrapolation"
            warnings.append(
                "Ridge was trained for Imperial College Expt 5. Custom currents/SoC protocols "
                "or inputs outside the training temperature/cycle range are not validated; "
                "hybrid result is exploratory."
            )
        if np.any((hybrid < 0) | (hybrid > 100)):
            warnings.append("Hybrid SoH outside [0,100]%; prediction was not clipped.")

    # The simulator normalizes its first working-current discharge to 100%;
    # Imperial experimental RPT SoH may use a different capacity measurement.
    warnings.append("Simulator SoH is first working-current discharge normalized to 100%; "
                    "experimental RPT SoH may not be directly equivalent.")

    # For legacy app.js, SOH_percent is the displayed best available estimate.
    shown = hybrid if hybrid is not None else physical
    final_index = -1
    final = {
        "cycle": int(cycle[final_index]),
        "soh_physical_percent": float(physical[final_index]),
        "soh_hybrid_percent": float(hybrid[final_index]) if hybrid is not None else None,
        "ml_correction_pp": float(correction[final_index]) if correction is not None else None,
        "sei_thickness_nm": float(sei[final_index]),
        "sei_growth_nm": float(sei_growth[final_index]),
        "discharge_capacity_Ah": float(capacity[final_index]),
        "capacity_loss_ah": float(loss[final_index]) if loss is not None else None,
    }
    return {
        "cycle": cycle.astype(int).tolist(),
        "discharge_capacity_Ah": capacity.tolist(),
        "SOH_percent": shown.tolist(),
        "SOH_physical_percent": physical.tolist(),
        "SOH_hybrid_percent": hybrid.tolist() if hybrid is not None else None,
        "ml_correction_pp": correction.tolist() if correction is not None else None,
        "capacity_fade_percent": (100.0 - shown).tolist(),
        "sei_thickness_nm": sei.tolist(),
        "sei_growth_nm": sei_growth.tolist(),
        "initial_sei_thickness_nm": initial_sei,
        "capacity_loss_ah": loss.tolist() if loss is not None else None,
        "inference": {
            "final": final,
            "physics_source": "user SPMe adapter",
            "ml_method": "Ridge residual learning" if correction is not None else None,
            "ml_status": ml_status,
            "warnings": warnings,
        },
    }
