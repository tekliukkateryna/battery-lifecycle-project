import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# You can leave blank if this notebook is saved in root directory of data
root_filepath = 'backend/data'
if len(root_filepath) > 0:
    root_filepath += '/'
main_data_path = 'Summary Data'

# Read metadata from excel file
meta_data_dict = {}
expt_list = ['expt 5']
for expt in expt_list:
    meta_data = pd.read_excel(f'{root_filepath}experiment_metadata.xlsx', sheet_name=f'{expt}')
    meta_data.set_index('Cell', inplace=True, drop=False)
    meta_data_dict[expt] = meta_data

# Define colours for plotting data later.  
colour_temp_dict = {10:'C0', 25:'C1', 40:'C3'}
data_dict = {}

# Directory containing Summary Data
base_path = Path(root_filepath) if root_filepath else Path.cwd()
summary_path = base_path / "Summary Data"

# Read only selected experiments
for expt in meta_data_dict.keys():

    main_data = {}
    age_set_data = {}

    # Read all cells from the experiment
    for cell in meta_data_dict[expt].index.values:

        temp_val = meta_data_dict[expt].loc[cell, 'Temp']
        cell_ID = meta_data_dict[expt].loc[cell, 'Cell']

        # Performance Summary: capacity, SoH, LLI, LAM
        main_file = (
            summary_path
            / "Performance Summary"
            / f"Expt {expt[5:]} - cell {cell_ID} ({temp_val}degC) - Processed Data.csv"
        )

        main_data[cell_ID] = pd.read_csv(
            main_file, index_col=0
        )

        # Ageing Sets Summary
        ageing_file = (
            summary_path
            / "Ageing Sets Summary"
            / "Summary per Set"
            / f"Expt {expt[5:]} - cell {cell_ID} - set_data.csv"
        )

        age_set_data[cell_ID] = pd.read_csv(
            ageing_file, index_col=0
        )

    data_dict[expt] = {
        'Main': main_data,
        'Ageing': age_set_data
    }

print("Loaded experiments:", list(data_dict.keys()))
print("Loaded cells:", list(data_dict['expt 5']['Main'].keys()))

''' expt = 'expt 5'

fig_ageing, ax = plt.subplots(
    2, figsize=(10, 8), sharex=True
)

# Variables to plot
x_variable = 'Ageing Cycles'
y_variable_1 = 'C/10 Capacity [mA h]'
y_variable_2 = '0.1s Resistance [Ohms]'

# Temperature colors
colors = {
    10: 'blue',
    25: 'green',
    40: 'red'
}

for cell in meta_data_dict[expt].index.values:

    temp_val = int(meta_data_dict[expt].loc[cell, 'Temp'])

    df = data_dict[expt]['Main'][cell]

    # Capacity degradation
    ax[0].plot(
        df[x_variable],
        df[y_variable_1],
        color=colors[temp_val],
        marker='o',
        markersize=4,
        label=f'Cell {cell} ({temp_val}°C)'
    )

    # Resistance: only even-numbered RPTs
    df_slice = df.iloc[::2]

    ax[1].plot(
        df_slice[x_variable],
        df_slice[y_variable_2],
        color=colors[temp_val],
        marker='o',
        markersize=4
    )

ax[0].set_ylabel('Capacity [mAh]')
ax[1].set_ylabel('Resistance [Ohm]')
ax[1].set_xlabel('Ageing Cycles')

ax[0].set_title(
    'LG M50T - Experimental Degradation (Experiment 5)'
)

for axis in ax:
    axis.grid(True, alpha=0.3)

ax[0].legend(fontsize=8, ncol=2)

fig_ageing.tight_layout()
plt.show() '''

expt = 'expt 5'
x_variable = 'Ageing Cycles'

dma_variables = [
    'SoH',
    'LAM PE',
    'LAM NE',
    'LAM NE_Gr',
    'LAM NE_Si',
    'LLI'
]

# Check which columns actually exist
first_df = next(iter(data_dict[expt]['Main'].values()))

available = [
    col for col in dma_variables
    if col in first_df.columns
]

print("Available:", available)
print("Missing:", set(dma_variables) - set(available))

if not available:
    raise ValueError("No matching DMA columns found")

colors = {10: 'blue', 25: 'green', 40: 'red'}

fig, axes = plt.subplots(
    len(available),
    1,
    figsize=(10, 2.5 * len(available)),
    sharex=True,
    squeeze=False
)

axes = axes.flatten()

for cell, df in data_dict[expt]['Main'].items():

    temp = int(meta_data_dict[expt].loc[cell, 'Temp'])

    for i, variable in enumerate(available):
        if variable not in df.columns:
            continue

        axes[i].plot(
            df[x_variable],
            df[variable],
            color=colors[temp],
            marker='o',
            markersize=3,
            label=f'Cell {cell}'
        )

        axes[i].set_ylabel(variable)
        axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Ageing Cycles')
axes[0].set_title('LG M50T - Degradation Mode Analysis')

fig.tight_layout()
plt.show()
