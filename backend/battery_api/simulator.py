"""Load real physics adapters; never invent SPMe data or silently fall back."""
import importlib
import os
from typing import Callable, Any


def load_callable(env_name: str) -> Callable[[dict], dict] | None:
    entry = os.getenv(env_name, "").strip()
    if not entry:
        return None
    module_name, separator, function_name = entry.partition(":")
    if not separator or not module_name or not function_name:
        raise RuntimeError(f"{env_name} must use the syntax 'module:function', got {entry!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{entry} is not callable")
    return fn


def run_physics(payload: dict[str, Any]) -> dict:
    fn = load_callable("BATT_SIM_ADAPTER")
    if fn is None:
        raise RuntimeError(
            "Real SPMe adapter missing. Configure BATT_SIM_ADAPTER="
            "user_simulation_adapter:run_lifecycle. No synthetic result returned."
        )
    result = fn(payload)
    if not isinstance(result, dict):
        raise TypeError("SPMe adapter must return a dict with the documented arrays")
    return result


def run_physics_step(payload: dict[str, Any]) -> dict:
    fn = load_callable("BATT_STEP_ADAPTER")
    if fn is None:
        raise NotImplementedError(
            "Stateful SPMe step telemetry not configured. "
            "Set BATT_STEP_ADAPTER when a real transient adapter is available."
        )
    result = fn(payload)
    expected = {"time_s", "voltage_V", "soc_percent", "sei_thickness_nm", "capacity_loss_ah", "current_A"}
    if not isinstance(result, dict) or expected - result.keys():
        raise ValueError(f"Step adapter must return {sorted(expected)}")
    return result
