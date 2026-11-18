
# Запуск: python -m backend.data.gen_data_3_comp.py

import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from backend.physics.config import (
    BatteryGeometry,
    BatteryParameters,
    SimulationConfig,
)
from backend.physics.simulation_cccc import BatterySimulation

from copy import deepcopy

N_CYCLES = 1000
CHECK_INTERVAL = 78

I_CHARGE = 1.5
I_DISCHARGE = 5.0
I_RPT = 0.5

CHARGE_TIME = 4 * 3600.0
DISCHARGE_TIME = 2 * 3600.0
RPT_DISCHARGE_TIME = 12 * 3600.0

T_RPT = 298.15
TEMPERATURES = [10, 25, 40]


RUN_RPT = True  # False = only fast 1C baseline


def get_capacity(data):
    return float(
        data["cycle_summary"]["discharge_capacity_Ah"][-1]
    )


def set_temperature(model, T_K):
    model.param.T_amb = float(T_K)
    model.sim.T_init = float(T_K)

    if model._last_state is not None:
        model._last_state[model.i_T] = float(T_K)


def virtual_rpt(ageing_model):
    """Measure C/10 capacity without changing the ageing model."""

    rpt_model = BatterySimulation(
        ageing_model.geo,
        deepcopy(ageing_model.param),
        deepcopy(ageing_model.sim),
        enable_sei=True,
        enable_thermal=True,
    )

    rpt_model._last_state = ageing_model._last_state.copy()
    rpt_model._last_time = ageing_model._last_time

    set_temperature(rpt_model, T_RPT)

    data = rpt_model.run_cycles(
        n_cycles=1,
        charge_time=CHARGE_TIME,
        discharge_time=RPT_DISCHARGE_TIME,
        pre_discharge=False,
        I_charge=I_CHARGE,
        I_discharge=I_RPT,
    )

    return get_capacity(data)


baseline_results = []
rpt_results = []

start_total = time.perf_counter()

for T_C in TEMPERATURES:

    start_time = time.perf_counter()
    T_K = T_C + 273.15

    geo = BatteryGeometry()
    param = BatteryParameters(T_amb=T_RPT)
    sim = SimulationConfig(T_init=T_RPT)

    model = BatterySimulation(
        geo, param, sim,
        enable_sei=True,
        enable_thermal=True,
    )

    Q0_RPT = None

    if RUN_RPT:
        # Initial C/10 measurement at 25°C
        initial_data = model.run_cycles(
            n_cycles=1,
            charge_time=CHARGE_TIME,
            discharge_time=RPT_DISCHARGE_TIME,
            pre_discharge=True,
            I_charge=I_CHARGE,
            I_discharge=I_RPT,
        )

        Q0_RPT = get_capacity(initial_data)

        rpt_results.append({
            "temperature_C": T_C,
            "cycle": 0,
            "capacity_Ah": Q0_RPT,
            "SOH_percent": 100.0,
        })

    set_temperature(model, T_K)

    completed_cycles = 0
    Q0_1C = None

    print(f"\nTemperature: {T_C}°C")

    while completed_cycles < N_CYCLES:

        block_size = min(
            CHECK_INTERVAL,
            N_CYCLES - completed_cycles
        )

        # Normal ageing, uninterrupted by RPT
        ageing_data = model.run_cycles(
            n_cycles=block_size,
            charge_time=CHARGE_TIME,
            discharge_time=DISCHARGE_TIME,
            pre_discharge=(not RUN_RPT and completed_cycles == 0),
            I_charge=I_CHARGE,
            I_discharge=I_DISCHARGE,
        )

        capacities = ageing_data[
            "cycle_summary"
        ]["discharge_capacity_Ah"]

        # Save the capacity of EVERY normal 1C cycle
        for i, Q in enumerate(capacities):

            cycle_number = completed_cycles + i + 1
            Q = float(Q)

            if Q0_1C is None:
                Q0_1C = Q

            baseline_results.append({
                "temperature_C": T_C,
                "cycle": cycle_number,
                "capacity_Ah": Q,
                "SOH_percent": 100.0 * Q / Q0_1C,
            })

        completed_cycles += block_size

        # Optional virtual RPT, done on a copy
        if RUN_RPT:

            Q_RPT = virtual_rpt(model)

            SOH_RPT = 100.0 * Q_RPT / Q0_RPT

            rpt_results.append({
                "temperature_C": T_C,
                "cycle": completed_cycles,
                "capacity_Ah": Q_RPT,
                "SOH_percent": SOH_RPT,
            })

            print(
                f"Cycle {completed_cycles}: "
                f"1C = {baseline_results[-1]['SOH_percent']:.4f}%, "
                f"C/10 = {SOH_RPT:.4f}%"
            )

    elapsed = time.perf_counter() - start_time
    print(f"Time: {elapsed / 60:.2f} min")


# Save baseline: 1000 points per temperature
df_baseline = pd.DataFrame(baseline_results)

output_dir = Path(__file__).parent

df_baseline.to_csv(
    output_dir / "model_soh_baseline.csv",
    index=False
)

# Save C/10 reference measurements separately
if RUN_RPT:
    df_rpt = pd.DataFrame(rpt_results)

    df_rpt.to_csv(
        output_dir / "model_soh_rpt_virtual.csv",
        index=False
    )


# Comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

for T_C in TEMPERATURES:

    subset = df_baseline[
        df_baseline["temperature_C"] == T_C
    ]

    line, = ax.plot(
        subset["cycle"],
        subset["SOH_percent"],
        label=f"{T_C}°C — 1C baseline",
    )

    if RUN_RPT:
        reference = df_rpt[
            df_rpt["temperature_C"] == T_C
        ]

        ax.plot(
            reference["cycle"],
            reference["SOH_percent"],
            linestyle="--",
            marker="o",
            markersize=4,
            color=line.get_color(),
            label=f"{T_C}°C — C/10 RPT",
        )

ax.set_xlabel("Cycle number")
ax.set_ylabel("Relative SoH [%]")
ax.set_title("SPMe + SEI: Baseline vs reference capacity")
ax.yaxis.set_major_formatter(
    ticker.FormatStrFormatter("%.3f")
)

ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

total = time.perf_counter() - start_total
print(f"Total simulation time: {total / 60:.2f} min")