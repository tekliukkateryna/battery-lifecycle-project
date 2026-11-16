# Запуск: python -m backend.data.gen_data.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

from backend.physics.config import (
    BatteryGeometry,
    BatteryParameters,
    SimulationConfig
)
from backend.physics.simulation import BatterySimulation


geo = BatteryGeometry()
param = BatteryParameters()
sim = SimulationConfig()


def create_model():
    return BatterySimulation(
        geo,
        param,
        sim,
        enable_sei=True,
        enable_thermal=True,
    )


def get_capacity(data):
    s = data["cycle_summary"]
    return float(s["discharge_capacity_Ah"][-1])


N_CYCLES = 1000
CHECK_INTERVAL = 78

I_CHARGE = 1.5
I_DISCHARGE = 5.0
I_RPT = 0.5

CHARGE_TIME = 4 * 3600.0
DISCHARGE_TIME = 2 * 3600.0
RPT_DISCHARGE_TIME = 12 * 3600.0

start_time = time.perf_counter()

model = create_model()

# Початкове контрольне вимірювання
reference_data = model.run_cycles(
    n_cycles=1,
    charge_time=CHARGE_TIME,
    discharge_time=RPT_DISCHARGE_TIME,
    pre_discharge=True,
    I_charge=I_CHARGE,
    I_discharge=I_RPT,
)

Q0 = get_capacity(reference_data)

print(f"Initial capacity: {Q0:.6f} Ah")

summary = [{
    "cycle": 0,
    "capacity_Ah": Q0,
    "SOH_percent": 100.0
}]

completed_cycles = 0

while completed_cycles < N_CYCLES:

    block_size = min(
        CHECK_INTERVAL,
        N_CYCLES - completed_cycles
    )

    # Старіння батареї
    model.run_cycles(
        n_cycles=block_size,
        charge_time=CHARGE_TIME,
        discharge_time=DISCHARGE_TIME,
        pre_discharge=False,
        I_charge=I_CHARGE,
        I_discharge=I_DISCHARGE,
    )

    completed_cycles += block_size

    # Контрольне вимірювання ємності
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
        "cycle": completed_cycles,
        "capacity_Ah": Q,
        "SOH_percent": SOH
    })

    print(
        f"Cycle {completed_cycles}: "
        f"Q = {Q:.6f} Ah, "
        f"SoH = {SOH:.4f}%"
    )

total_time_sec = time.perf_counter() - start_time

print(f"Total time: {total_time_sec / 60:.2f} min")

# Побудова графіка
cycles = [item["cycle"] for item in summary]
soh = [item["SOH_percent"] for item in summary]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(cycles, soh, marker="o")

ax.set_xlabel("Cycle number")
ax.set_ylabel("Relative SOH [%]")
ax.set_title("Capacity retention during cycling")

ax.yaxis.set_major_formatter(
    ticker.FormatStrFormatter("%.4f")
)

ax.grid(True)
plt.tight_layout()
plt.show()