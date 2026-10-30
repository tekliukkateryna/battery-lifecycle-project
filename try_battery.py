import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    enable_thermal=True
)

# 1000 cycles
data = model.run_cycles(
    n_cycles = 1000,
    charge_time = 3 * 3600.0,
    discharge_time = 3 * 3600.0,
)

summary = data["cycle_summary"]

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