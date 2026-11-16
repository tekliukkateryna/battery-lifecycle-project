"""Prepare Imperial College London Experiment 5 data for residual learning.

Place this file in: battery-lifecycle-project/backend/data/
Run from the project root:

# python -m backend.data.prepare_imperial_residuals

Required files under backend/data/:
    experiment_metadata.xlsx
    model_soh_3temps.csv
    Summary Data/Performance Summary/
        Expt 5 - cell A (10degC) - Processed Data.csv
        ... (all cells listed in the 'expt 5' metadata sheet)

Produces (all under backend/data/):
    imperial_soh_cells.csv         individual experimental observations
    residual_learning_dataset.csv  matched experimental/model SoH + residuals
    imperial_vs_spme.png           comparison plot without averaging cells

The C/10 capacity is converted from mAh to Ah. Each cell's SoH is
normalized to its own cycle-zero RPT. Model values are interpolated
at the *actual* experimental cycle counts, without extrapolation.
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
MODEL_COLUMNS = {"temperature_C", "cycle", "capacity_Ah", "SOH_percent"}


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


def load_model(model_file: Path) -> pd.DataFrame:
    if not model_file.is_file():
        raise FileNotFoundError(f"Simulation CSV not found: {model_file}")
    model = pd.read_csv(model_file)
    missing = MODEL_COLUMNS - set(model.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")
    model = model.copy()
    for name in MODEL_COLUMNS:
        model[name] = pd.to_numeric(model[name], errors="coerce")
    if model[list(MODEL_COLUMNS)].isna().any().any():
        raise ValueError("Model CSV contains missing or nonnumeric fields")
    if not np.isfinite(model[list(MODEL_COLUMNS)].to_numpy()).all():
        raise ValueError("Model CSV contains non-finite values")
    if model.duplicated(["temperature_C", "cycle"]).any():
        raise ValueError("Model CSV has duplicate temperature/cycle pairs")
    return model.sort_values(["temperature_C", "cycle"])


def match_to_model(cells: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    matched = []
    for temperature, observations in cells.groupby("temperature_C"):
        predicted = model.loc[model["temperature_C"] == temperature]
        if predicted.empty:
            raise ValueError(f"No SPMe simulation for {temperature}°C")
        predicted = predicted.sort_values("cycle")
        x = predicted["cycle"].to_numpy(dtype=float)
        y = predicted["SOH_percent"].to_numpy(dtype=float)
        model_capacities = predicted["capacity_Ah"].to_numpy(dtype=float)

        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError(f"Invalid model cycle grid at {temperature}°C")

        within_range = observations["cycle"].between(x[0], x[-1])
        skipped = int((~within_range).sum())
        if skipped:
            print(f"Note: {skipped} observations at {temperature}°C lie outside "
                  f"the modeled [{x[0]:g}, {x[-1]:g}] cycle range; "
                  "they stay in imperial_soh_cells.csv, but not the residual CSV.")

        valid = observations.loc[within_range].copy()
        if valid.empty:
            continue
        cycles = valid["cycle"].to_numpy(dtype=float)
        valid["soh_exp"] = valid["SOH_percent"].to_numpy()
        valid["soh_model"] = np.interp(cycles, x, y)
        valid["capacity_model_Ah"] = np.interp(cycles, x, model_capacities)
        valid["residual"] = valid["soh_exp"] - valid["soh_model"]
        matched.append(valid)

    if not matched:
        raise ValueError("No overlapping experimental/model observations")

    result = pd.concat(matched, ignore_index=True)
    return result[[
        "cell_id", "temperature_C", "rpt_index", "cycle",
        "capacity_Ah", "capacity_model_Ah", "soh_exp", "soh_model", "residual",
    ]].sort_values(["temperature_C", "cell_id", "cycle", "rpt_index"])


def plot_comparison(cells: pd.DataFrame, model: pd.DataFrame, filename: Path,
                    show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {10: "tab:blue", 25: "tab:orange", 40: "tab:green"}
    for temperature in sorted(cells["temperature_C"].unique()):
        temp_cells = cells.loc[cells["temperature_C"] == temperature]
        color = colors.get(int(temperature), None)
        for cell, points in temp_cells.groupby("cell_id"):
            points = points.sort_values("cycle")
            ax.plot(points["cycle"], points["SOH_percent"], "o-",
                    color=color, alpha=0.55, linewidth=1.2, markersize=3,
                    label=f"Imperial {cell} ({temperature}°C)")

        physics = model.loc[model["temperature_C"] == temperature]
        if not physics.empty:
            ax.plot(physics["cycle"], physics["SOH_percent"], "--",
                    color=color, linewidth=2.5,
                    label=f"SPMe+SEI ({temperature}°C)")

    ax.set(xlabel="Ageing cycles", ylabel="Relative SoH [%]",
           title="Imperial College Experiment 5 vs SPMe+SEI (individual cells)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder with experiment_metadata.xlsx and Summary Data")
    parser.add_argument("--model-file", type=Path, default=None,
                        help="Model CSV (default: <data-dir>/model_soh_3temps.csv)")
    parser.add_argument("--show", action="store_true", help="Display the comparison plot")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    model_file = (args.model_file or data_dir / "model_soh_3temps.csv").resolve()
    print(f"Reading Imperial Experiment 5 from: {data_dir}")
    print(f"Reading SPMe predictions from: {model_file}\n")

    cells = load_imperial_cells(data_dir)
    model = load_model(model_file)
    residuals = match_to_model(cells, model)

    cells_file = data_dir / "imperial_soh_cells.csv"
    residual_file = data_dir / "residual_learning_dataset.csv"
    chart_file = data_dir / "imperial_vs_spme.png"

    cells.to_csv(cells_file, index=False)
    residuals.to_csv(residual_file, index=False)
    plot_comparison(cells, model, chart_file, show=args.show)

    print("\nDone — no battery averaging performed.")
    print(f"Individual observations: {len(cells)} | Cells: {cells['cell_id'].nunique()}")
    print(f"Matched residual observations: {len(residuals)}")
    print("Residuals use percentage points: experimental SoH - SPMe SoH")
    print(f"Saved: {cells_file}")
    print(f"Saved: {residual_file}")
    print(f"Saved: {chart_file}")
    print("\nObservations per cell:")
    print(residuals.groupby(["temperature_C", "cell_id"]).size().to_string())


if __name__ == "__main__":
    main()
