# Запуск: python -m backend.data.gen_data_3_C10.py

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

# !!!!!!!!!!!!!!!!! For faster cycling use cccc simulation
# from backend.physics.simulation_ cccc import BatterySimulation
from backend.physics.simulation_cccv import BatterySimulation


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

#!!!!!!!!!!!!!!!! For fast 1C cycling, use the following parameters:
# I_RPT = 5
# RPT_DISCHARGE_TIME = 2 * 3600.0

def get_capacity(data):
    return float(
        data["cycle_summary"]["discharge_capacity_Ah"][-1]
    )


def set_temperature(model, T_K):
    # Instantaneous temperature equilibration approximation
    model.param.T_amb = float(T_K)
    model.sim.T_init = float(T_K)

    if model._last_state is not None:
        model._last_state[model.i_T] = float(T_K)


all_results = []
start_total = time.perf_counter()

fig, ax = plt.subplots(figsize=(9, 5))

for T_C in TEMPERATURES:

    start_time = time.perf_counter()
    T_K = T_C + 273.15

    geo = BatteryGeometry()
    param = BatteryParameters(T_amb=T_RPT)
    sim = SimulationConfig(T_init=T_RPT)

    model = BatterySimulation(
        geo,
        param,
        sim,
        enable_sei=True,
        enable_thermal=True,
    )

    # Initial RPT at 25°C
    reference_data = model.run_cycles(
        n_cycles=1,
        charge_time=CHARGE_TIME,
        discharge_time=RPT_DISCHARGE_TIME,
        pre_discharge=True,
        I_charge=I_CHARGE,
        I_discharge=I_RPT,
    )

    Q0 = get_capacity(reference_data)

    if not hasattr(model, "_last_state") or model._last_state is None:
        raise RuntimeError(
            "run_cycles() must preserve the final battery "
            "state in model._last_state"
        )

    print(f"\nTemperature: {T_C}°C")
    print(f"Initial capacity: {Q0:.6f} Ah")

    summary = [{
        "temperature_C": T_C,
        "cycle": 0,
        "capacity_Ah": Q0,
        "SOH_percent": 100.0,
    }]

    # Switch to ageing temperature
    set_temperature(model, T_K)

    completed_cycles = 0

    while completed_cycles < N_CYCLES:

        block_size = min(
            CHECK_INTERVAL,
            N_CYCLES - completed_cycles
        )

        # Ageing cycles at 10°C, 25°C or 40°C
        model.run_cycles(
            n_cycles=block_size,
            charge_time=CHARGE_TIME,
            discharge_time=DISCHARGE_TIME,
            pre_discharge=False,
            I_charge=I_CHARGE,
            I_discharge=I_DISCHARGE,
        )

        completed_cycles += block_size

        # Reference performance test at 25°C
        set_temperature(model, T_RPT)

        data = model.run_cycles(
            n_cycles=1,
            charge_time=CHARGE_TIME,
            discharge_time=RPT_DISCHARGE_TIME,
            pre_discharge=False,
            I_charge=I_CHARGE,
            I_discharge=I_RPT,
        )

        Q = get_capacity(data)
        SOH = Q / Q0 * 100.0

        summary.append({
            "temperature_C": T_C,
            "cycle": completed_cycles,
            "capacity_Ah": Q,
            "SOH_percent": SOH,
        })

        print(
            f"Cycle {completed_cycles}: "
            f"Q = {Q:.6f} Ah, "
            f"SoH = {SOH:.4f}%"
        )

        # Return to ageing temperature
        set_temperature(model, T_K)

    all_results.extend(summary)

    elapsed = time.perf_counter() - start_time
    print(f"Time at {T_C}°C: {elapsed / 60:.2f} min")

    ax.plot(
        [row["cycle"] for row in summary],
        [row["SOH_percent"] for row in summary],
        marker="o",
        label=f"{T_C}°C",
    )


# Save results for residual learning
df = pd.DataFrame(all_results)

output_path = Path(__file__).with_name(
    "model_soh_3temps.csv"
)

df.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Rows: {len(df)}")

total_time = time.perf_counter() - start_total
print(f"Total time: {total_time / 60:.2f} min")

ax.set_xlabel("Cycle number")
ax.set_ylabel("Relative SoH [%]")
ax.set_title("SPMe + SEI: Capacity retention")
ax.yaxis.set_major_formatter(
    ticker.FormatStrFormatter("%.3f")
)

ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
