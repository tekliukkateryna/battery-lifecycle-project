"""Prepare Imperial College London Experiment 5 data for residual learning.

Run:
# python -m backend.data.get_data_imperial

Required files:
    experiment_metadata.xlsx
    Summary Data/Performance Summary/
        Expt 5 ...

Produces:
    imperial_soh_cells.csv         individual experimental observations
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_SHEET = "expt 5"
EXPERIMENT_NUMBER = 5
CYCLE_COLUMN = "Ageing Cycles"
CAPACITY_COLUMN = "C/10 Capacity [mA h]"


def load_imperial_cells(data_dir: Path) -> pd.DataFrame:
    metadata_file = data_dir / "experiment_metadata.xlsx"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_excel(metadata_file, sheet_name=EXPERIMENT_SHEET)
    required_meta = {"Cell", "Temp"}
    missing = required_meta - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if metadata.empty:
        raise ValueError("No cells in experiment metadata, sheet 'expt 5'.")

    rows = []
    summary_dir = data_dir / "Summary Data" / "Performance Summary"

    for _, meta in metadata.iterrows():
        cell = str(meta["Cell"]).strip()
        if not cell or cell.lower() == "nan":
            raise ValueError("Empty Cell identifier in metadata")
        temperature = float(meta["Temp"])
        if not np.isfinite(temperature) or not temperature.is_integer():
            raise ValueError(f"Invalid temperature for cell {cell}: {meta['Temp']}")
        temperature = int(temperature)

        csv_path = summary_dir / (
            f"Expt {EXPERIMENT_NUMBER} - cell {cell} "
            f"({temperature}degC) - Processed Data.csv"
        )
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing file for cell {cell}:\n  {csv_path}\n"
                "Check the extracted 'Summary Data' folder and file names."
            )

        # The first CSV column is an index in the original ICL notebook.
        raw = pd.read_csv(csv_path, index_col=0)
        missing_cols = {CYCLE_COLUMN, CAPACITY_COLUMN} - set(raw.columns)
        if missing_cols:
            raise ValueError(
                f"Cell {cell}: missing columns {sorted(missing_cols)}. "
                f"Found: {list(raw.columns)}"
            )

        obs = pd.DataFrame({
            "rpt_index": np.arange(len(raw), dtype=int),
            "cycle": pd.to_numeric(raw[CYCLE_COLUMN], errors="coerce").to_numpy(),
            "capacity_Ah": (
                pd.to_numeric(raw[CAPACITY_COLUMN], errors="coerce").to_numpy()
                / 1000.0
            ),
        })
        obs = obs.replace([np.inf, -np.inf], np.nan)
        obs = obs.dropna(subset=["cycle", "capacity_Ah"])
        obs = obs.loc[(obs["cycle"] >= 0) & (obs["capacity_Ah"] > 0)].copy()
        obs = obs.sort_values(["cycle", "rpt_index"])

        if obs.empty:
            raise ValueError(f"Cell {cell}: no valid C/10 capacity observations")

        initial = obs.loc[np.isclose(obs["cycle"], 0.0, atol=1e-9)]
        if initial.empty:
            raise ValueError(
                f"Cell {cell}: no cycle-zero C/10 capacity found. "
                "Inspect the RPT baseline before computing SoH; "
                "do not silently normalize to another cycle."
            )
        if len(initial) > 1:
            print(f"Warning: cell {cell} has {len(initial)} records at cycle 0; "
                  "using the earliest as Q0 and keeping every record.")
        q0 = float(initial.iloc[0]["capacity_Ah"])

        obs["SOH_percent"] = obs["capacity_Ah"] / q0 * 100.0
        obs["cell_id"] = cell
        obs["temperature_C"] = temperature
        rows.append(obs)
        print(f"Cell {cell}: {temperature}°C | Q0={q0:.6f} Ah | "
              f"{len(obs)} valid RPT observations")

    cells = pd.concat(rows, ignore_index=True)
    return cells[[
        "cell_id", "temperature_C", "rpt_index", "cycle",
        "capacity_Ah", "SOH_percent",
    ]].sort_values(["temperature_C", "cell_id", "cycle", "rpt_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder with experiment_metadata.xlsx and Summary Data")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    print(f"Reading Imperial Experiment 5 from: {data_dir}\n")

    cells = load_imperial_cells(data_dir)
    cells_file = data_dir / "imperial_soh_cells.csv"
    cells.to_csv(cells_file, index=False)

    print(f"\nExtracted {len(cells)} observations across {cells['cell_id'].nunique()} cells.")
    print(f"Saved: {cells_file}")

if __name__ == "__main__":
    main()
