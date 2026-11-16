"""Plot residual-learning results for LG M50T (Imperial College, experiment 5).

Run from the project root after train_residual_ridge.py:
    python -m backend.data.plot_residual_results
    python -m backend.data.plot_residual_results --show

Reads in the directory containing this script:
    residual_ridge_predictions.csv

Writes the following PNGs next to the input CSV:
    ridge_soh_by_temperature.png    - experimental cells, physical, held-out-cell hybrid
    ridge_soh_by_cell.png           - one panel for each battery
    ridge_residuals_by_temperature.png - observed vs predicted ML residuals
    ridge_errors_by_cell.png        - MAE comparison across eight cells

The hybrid predictions are leave-one-cell-out *only if* the input CSV was
produced by train_residual_ridge.py without modifying its validation logic.
No re-training or scikit-learn is required to plot these results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / 'residual_ridge_predictions.csv'

REQUIRED = {
    'cell_id', 'temperature_C', 'cycle',
    'soh_exp', 'soh_model', 'soh_hybrid',
}


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f'Cannot find {path}. Run train_residual_ridge.py first, '
            'and place this plotting script in the same folder as the CSV.'
        )
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path.name}: {sorted(missing)}')

    df['cell_id'] = df['cell_id'].astype(str)
    for col in ('temperature_C', 'cycle', 'soh_exp', 'soh_model', 'soh_hybrid'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError('CSV has missing values in required columns')
    if not np.isfinite(df[['temperature_C','cycle','soh_exp','soh_model','soh_hybrid']]).all().all():
        raise ValueError('CSV contains infinite or invalid numeric values')
    if df.duplicated(['cell_id', 'cycle']).any():
        raise ValueError('Duplicate cycle numbers for one cell: check RPT rows')
    if df.groupby('cell_id')['temperature_C'].nunique().gt(1).any():
        raise ValueError('At least one cell has more than one ageing temperature')
    df['residual_observed'] = df['soh_exp'] - df['soh_model']
    df['residual_predicted'] = df['soh_hybrid'] - df['soh_model']
    df['absolute_error_physical'] = (df['soh_model'] - df['soh_exp']).abs()
    df['absolute_error_hybrid'] = (df['soh_hybrid'] - df['soh_exp']).abs()
    return df.sort_values(['temperature_C','cell_id','cycle']).reset_index(drop=True)


def save_figure(fig, path: Path, show: bool) -> None:
    fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}')
    if not show:
        plt.close(fig)


def plot_by_temperature(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    temperatures = sorted(df['temperature_C'].unique())
    fig, axs = plt.subplots(1, len(temperatures), figsize=(5.6 * len(temperatures), 5),
                            sharex=True, sharey=True, squeeze=False)
    axs = axs.ravel()
    for ax, temp in zip(axs, temperatures):
        group = df[df['temperature_C'] == temp]
        physical = group.drop_duplicates('cycle').sort_values('cycle')
        ax.plot(physical['cycle'], physical['soh_model'], 'k:', lw=2,
                label='SPMe + SEI')
        cells = sorted(group['cell_id'].unique())
        palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(cells))))
        for i, cell in enumerate(cells):
            cell_df = group[group['cell_id'] == cell].sort_values('cycle')
            color = palette[i]
            ax.plot(cell_df['cycle'], cell_df['soh_exp'], '-o', color=color,
                    ms=3.2, lw=1.5, alpha=.9, label=f'{cell} observed')
            ax.plot(cell_df['cycle'], cell_df['soh_hybrid'], '--', color=color,
                    lw=1.8, alpha=.95, label=f'{cell} Ridge (held-out)')
        ax.set_title(f'{temp:g}°C', fontweight='bold')
        ax.set_xlabel('Ageing cycles')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc='best')
    axs[0].set_ylabel('SoH (%)')
    fig.suptitle('LG M50T: experiment vs SPMe+SEI vs Ridge (held-out cells)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / 'ridge_soh_by_temperature.png', show)


def plot_by_cell(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    cells = sorted(df['cell_id'].unique())
    ncols = min(4, len(cells))
    nrows = (len(cells) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(4.3 * ncols, 3.6 * nrows),
                            sharex=True, sharey=True, squeeze=False)
    axes = axs.ravel()
    for ax, cell in zip(axes, cells):
        g = df[df['cell_id'] == cell].sort_values('cycle')
        temp = float(g['temperature_C'].iloc[0])
        ax.plot(g['cycle'], g['soh_exp'], 'o-', ms=3, lw=1.7,
                label='Measured SoH')
        ax.plot(g['cycle'], g['soh_model'], ':', lw=2,
                label='SPMe + SEI')
        ax.plot(g['cycle'], g['soh_hybrid'], '--', lw=1.9,
                label='Hybrid (held-out)')
        ax.set_title(f'Cell {cell}  |  {temp:g}°C')
        ax.grid(alpha=.25)
        ax.set_xlabel('Cycles')
        ax.set_ylabel('SoH (%)')
    for ax in axes[len(cells):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(.5, -0.008), frameon=False)
    fig.suptitle('Prediction quality by individual battery (LOCO validation)',
                 fontsize=14, y=1.01)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, output_dir / 'ridge_soh_by_cell.png', show)


def plot_residuals(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    temperatures = sorted(df['temperature_C'].unique())
    fig, axs = plt.subplots(1, len(temperatures), figsize=(5.6 * len(temperatures), 4.8),
                            sharex=True, sharey=True, squeeze=False)
    for ax, temp in zip(axs.ravel(), temperatures):
        group = df[df['temperature_C'] == temp]
        cells = sorted(group['cell_id'].unique())
        palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(cells))))
        for i, cell in enumerate(cells):
            g = group[group['cell_id'] == cell].sort_values('cycle')
            ax.plot(g['cycle'], g['residual_observed'], 'o-', color=palette[i],
                    lw=1.3, ms=3, label=f'{cell} measured residual')
            ax.plot(g['cycle'], g['residual_predicted'], '--', color=palette[i],
                    lw=1.7, label=f'{cell} predicted residual')
        ax.axhline(0, c='black', lw=.7, alpha=.4)
        ax.set_title(f'{temp:g}°C')
        ax.set_xlabel('Ageing cycles')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axs.ravel()[0].set_ylabel('Residual = SoH exp - SoH model (percentage points)')
    fig.suptitle('Residual learning: measured and out-of-cell Ridge predictions',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / 'ridge_residuals_by_temperature.png', show)


def plot_mae(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    # Exclude cycle 0, whose 100%-100% residual is trivial.
    evaluated = df[df['cycle'] > 0]
    if evaluated.empty:
        raise ValueError('No cycles > 0: cannot evaluate nontrivial accuracy')
    metrics = evaluated.groupby(['temperature_C','cell_id'], sort=True).agg(
        MAE_physical_pp=('absolute_error_physical', 'mean'),
        MAE_hybrid_pp=('absolute_error_hybrid', 'mean'),
        n_points=('cycle','size'),
    ).reset_index()
    labels = [f'{r.cell_id}\n{r.temperature_C:g}°C' for r in metrics.itertuples()]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5))
    width = .38
    ax.bar(x - width / 2, metrics['MAE_physical_pp'], width, label='SPMe + SEI')
    ax.bar(x + width / 2, metrics['MAE_hybrid_pp'], width, label='Hybrid (held-out)')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Mean absolute error (percentage points)')
    ax.set_title('Validation error by battery (excluding cycle 0)')
    ax.grid(axis='y', alpha=.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / 'ridge_errors_by_cell.png', show)

    physical_mae = evaluated['absolute_error_physical'].mean()
    hybrid_mae = evaluated['absolute_error_hybrid'].mean()
    max_hybrid_error = evaluated['absolute_error_hybrid'].max()
    print(f'Physical MAE = {physical_mae:.4f} pp')
    print(f'Hybrid MAE   = {hybrid_mae:.4f} pp')
    print(f'Max hybrid absolute error = {max_hybrid_error:.4f} pp')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot LG M50T SPMe+SEI and Ridge predictions')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Path to residual_ridge_predictions.csv')
    parser.add_argument('--show', action='store_true', help='Open figures interactively after saving')
    args = parser.parse_args()
    path = args.input.expanduser().resolve()
    df = load_predictions(path)
    out_dir = path.parent
    print(f'Loaded {len(df)} observations, {df.cell_id.nunique()} cells, '
          f'{df.temperature_C.nunique()} temperatures')
    plot_by_temperature(df, out_dir, args.show)
    plot_by_cell(df, out_dir, args.show)
    plot_residuals(df, out_dir, args.show)
    plot_mae(df, out_dir, args.show)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
