import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from backend.physics.config import BatteryGeometry, BatteryParameters, SimulationConfig
from backend.physics.simulation import BatterySimulation

geo = BatteryGeometry()
param = BatteryParameters()
sim = SimulationConfig()

model = BatterySimulation(
    geo,
    param,
    sim,
    enable_sei=True,
    enable_thermal=True,
)
start_time = time.perf_counter()
# 1000 cycles
data = model.run_cycles(
    n_cycles = 1000,
    charge_time = 3 * 3600.0,
    discharge_time = 3 * 3600.0,
    pre_discharge=True,
)

summary = data["cycle_summary"]

total_time_sec = time.perf_counter() - start_time
print(total_time_sec / 60.0)

q = data["cycle_summary"]["discharge_capacity_Ah"]

print("Cycle 1:", q[0], "Ah")
print("Cycle 2:", q[1], "Ah")
print("Cycle 3:", q[2], "Ah")
print("Cycle 1000:", q[-1], "Ah")

print("Loss after cycle 2:",
      100 * (1 - q[-1] / q[1]), "%")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    summary["cycle"],
    summary["SOH_percent"]
)

ax.set_xlabel("Cycle number")
ax.set_ylabel("Relative SOH [%]")
ax.set_title("Capacity retention during cycling")

# Show four decimal places
ax.yaxis.set_major_formatter(
    ticker.FormatStrFormatter("%.4f")
)

ax.grid(True)
plt.show()