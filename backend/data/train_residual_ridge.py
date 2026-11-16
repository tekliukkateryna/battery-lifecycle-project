"""Train and validate Ridge residual learning for Imperial College Expt 5.

Run from project root:
    python -m backend.ML.Ringe_residual_learning.train_residual_ridge

Expected file in the same folder:
    residual_learning_dataset.csv

Outputs (same folder):
    residual_ridge_predictions.csv
    residual_ridge_metrics.csv
    residual_ridge_comparison.png
    residual_ridge_model.joblib

The evaluation holds out an entire cell each time (leave-one-cell-out).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import joblib

DATA_DIR = Path(__file__).resolve().parent
INPUT = DATA_DIR / "residual_learning_dataset.csv"
ALPHA = 0.01  # Chosen beforehand, not optimized on test batteries

REQUIRED = {"cell_id", "temperature_C", "cycle", "soh_exp", "soh_model"}


def make_features(df: pd.DataFrame) -> np.ndarray:
    n = df["cycle"].to_numpy(dtype=float) / 1000.0
    t = (df["temperature_C"].to_numpy(dtype=float) - 25.0) / 15.0
    if np.any(n < 0):
        raise ValueError("Negative cycle count")
    sn = np.sqrt(n)
    return np.column_stack((sn, n, t * sn, t * n, t**2 * sn, t**2 * n))


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT}. Put this script beside residual_learning_dataset.csv."
        )

    data = pd.read_csv(INPUT)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = data.copy()
    for col in ("temperature_C", "cycle", "soh_exp", "soh_model"):
        data[col] = pd.to_numeric(data[col], errors="raise")
    data["cell_id"] = data["cell_id"].astype(str)
    data = data.sort_values(["temperature_C", "cell_id", "cycle"]).reset_index(drop=True)
    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Input has missing values")
    if data.duplicated(["cell_id", "cycle"]).any():
        raise ValueError("Repeated cycle for a cell: inspect individual RPTs before training")
    if data.groupby("cell_id")["temperature_C"].nunique().gt(1).any():
        raise ValueError("A cell has multiple ageing temperatures")
    if data["cell_id"].nunique() < 3:
        raise ValueError("Need at least three cells for held-out-cell validation")

    data["residual"] = data["soh_exp"] - data["soh_model"]
    data["pred_residual"] = np.nan

    # Each fold's test measurements belong to a battery unseen during training.
    for cell in sorted(data["cell_id"].unique()):
        mask_test = data["cell_id"].eq(cell)
        train, test = data.loc[~mask_test], data.loc[mask_test]
        model = Ridge(alpha=ALPHA, fit_intercept=False)
        model.fit(make_features(train), train["residual"])
        data.loc[mask_test, "pred_residual"] = model.predict(make_features(test))

    data["soh_hybrid"] = data["soh_model"] + data["pred_residual"]
    data["error_physical_pp"] = data["soh_model"] - data["soh_exp"]
    data["error_hybrid_pp"] = data["soh_hybrid"] - data["soh_exp"]

    # Exclude the trivial N=0 points from validation statistics.
    scored = data.loc[data["cycle"] > 0]
    rows = []
    for cell, part in scored.groupby("cell_id", sort=True):
        rows.append({
            "cell_id": cell,
            "temperature_C": int(part["temperature_C"].iloc[0]),
            "n_test_points": len(part),
            "MAE_physical_pp": np.abs(part["error_physical_pp"]).mean(),
            "MAE_hybrid_pp": np.abs(part["error_hybrid_pp"]).mean(),
            "MaxAE_hybrid_pp": np.abs(part["error_hybrid_pp"]).max(),
        })
    metrics = pd.DataFrame(rows).sort_values(["temperature_C", "cell_id"])

    print("=== Leave-one-cell-out validation (cycle > 0) ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOverall held-out-cell MAE:")
    print(f"  Physical model: {np.abs(scored['error_physical_pp']).mean():.4f} pp")
    print(f"  Hybrid model:   {np.abs(scored['error_hybrid_pp']).mean():.4f} pp")
    print(f"  Max hybrid absolute error: {np.abs(scored['error_hybrid_pp']).max():.4f} pp")

    data.to_csv(DATA_DIR / "residual_ridge_predictions.csv", index=False)
    metrics.to_csv(DATA_DIR / "residual_ridge_metrics.csv", index=False)

    # Fit a deployment candidate only AFTER cross-validation, on all cells.
    # Do not treat this in-sample fit as the quality estimate.
    final_model = Ridge(alpha=ALPHA, fit_intercept=False)
    final_model.fit(make_features(data), data["residual"])
    joblib.dump(final_model, DATA_DIR / "residual_ridge_model.joblib")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)
    colors = {10: "tab:blue", 25: "tab:green", 40: "tab:red"}
    for ax, temp in zip(axs, (10, 25, 40)):
        subset = data.loc[data["temperature_C"].eq(temp)]
        for cell, cell_df in subset.groupby("cell_id"):
            ax.plot(cell_df["cycle"], cell_df["soh_exp"], "o-", markersize=3,
                    color=colors[temp], alpha=0.65, label=f"{cell} exp")
            ax.plot(cell_df["cycle"], cell_df["soh_hybrid"], "--", linewidth=1.4,
                    alpha=0.75, label=f"{cell} hybrid (held out)")
        physical = subset.sort_values("cycle").drop_duplicates("cycle")
        ax.plot(physical["cycle"], physical["soh_model"], color="black",
                linewidth=1.6, label="Physical model")
        ax.set_title(f"Ageing at {temp}°C")
        ax.set_xlabel("Ageing cycles")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=1)
    axs[0].set_ylabel("SoH (%)")
    fig.suptitle("Imperial College vs SPMe and out-of-cell Ridge correction")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "residual_ridge_comparison.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved outputs in {DATA_DIR}")
    print("Note: this tests new cells under the SAME three ageing protocols,")
    print("not transfer to different currents or SoC windows.")


if __name__ == "__main__":
    main()
