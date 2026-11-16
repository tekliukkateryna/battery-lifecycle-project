
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

folder = Path(__file__).parent

baseline = pd.read_csv(folder / "model_soh_baseline.csv")
rpt = pd.read_csv(folder / "model_soh_rpt_virtual.csv")

# Match temperatures and cycle numbers
df = rpt.merge(
    baseline,
    on=["temperature_C", "cycle"],
    suffixes=("_rpt", "_1c"),
    validate="one_to_one"
)

# Difference in percentage points
df["delta"] = (
    df["SOH_percent_rpt"] -
    df["SOH_percent_1c"]
)

print("\nComparison by temperature:")

for T in [10, 25, 40]:

    part = df[df["temperature_C"] == T]

    mae = np.mean(np.abs(part["delta"]))
    max_error = np.max(np.abs(part["delta"]))

    print(
        f"{T}°C: "
        f"MAE = {mae:.5f} pp, "
        f"Max = {max_error:.5f} pp"
    )

fig, ax = plt.subplots(figsize=(9, 5))

for T in [10, 25, 40]:

    part = df[df["temperature_C"] == T]

    ax.plot(
        part["cycle"],
        part["delta"],
        marker="o",
        label=f"{T}°C"
    )

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Cycle number")
ax.set_ylabel("C/10 SoH - 1C SoH [percentage points]")
ax.set_title("Difference between reference and baseline SoH")

ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
