# Physics-Based Battery Lifecycle Simulation with ML Residual Correction (LG M50T)

**A simulator for the LG M50T lithium-ion cell, combining electrochemical modelling, SEI ageing, simple machine-learning correction, and an interactive web interface.**

The project aims to predict how a cell's **capacity and state of health (SoH)** evolve during repeated charge–discharge cycles. It combines a simplified physics-based model with an optional data-driven correction rather than relying on a purely empirical prediction.

The project is prototype, thus the physical parameter set and ML correction have limited validation.

## 1. Project overview

BATT-SIM has three main components:

1. **Physics-based simulation:** SPMe electrochemistry, a lumped thermal model, and SEI-driven degradation.
2. **Optional ML residual correction:** Ridge regression adjusts the physics-based SoH prediction using experimental ageing data.
3. **Web interface:** interactive battery visualisation, simulation settings, and lifecycle plots.

```text
             Web interface (HTML / JavaScript / Three.js)
                              |
                              v
                        FastAPI backend
                              |
                              v
                  Lifecycle simulation adapter
                              |
                              v
                      SPMe + Thermal + SEI
                      (SciPy BDF / Numba)
                              |
                    Physics-based SoH
                              |
                        + optional
                       Ridge residual
                              |
                              v
                 Results, curves, 3D interface
```

**Note:** Full lifecycle curves come from the Python physics simulator. The interactive, frame-by-frame 3D demo are illustrative. 

## 2. Quick start

Run the following commands **from the repository root** using **Windows PowerShell**.

### Install dependencies

```powershell
python -m pip install -r requirements_backend.txt
```
Additional for some experiments with data: matplotlib openpyxl

### Start the backend

In the **first terminal**:

```powershell
$env:BATT_SIM_ADAPTER = "user_simulation_adapter:run_lifecycle"
python -m uvicorn backend.battery_api.server:app --host 127.0.0.1 --port 8000
```

Alternatively, run `./start_backend.ps1` from PowerShell (or `.\start_backend.ps1` on Windows).

- API health check: http://127.0.0.1:8000/api/health
- Interactive API documentation: http://127.0.0.1:8000/docs


### Start the frontend

In a **second terminal**:

```powershell
cd frontend
python -m http.server 5500
```

Open **http://127.0.0.1:5500** in your browser.

You can also start the frontend from the repository root with `.\start_frontend.ps1`.

> The frontend loads Three.js resources from a CDN, so internet access may be needed for its 3D scene. The API and frontend must run in separate terminals.

## 3. Physics-based model

The physical model is designed as a compromise between **physical interpretability, numerical accuracy, and computational cost**. It captures important electrochemical mechanisms without the expense of a full porous-electrode model at every spatial scale.

### Single Particle Model with Electrolyte (SPMe)

The electrochemical equations are based on the **asymptotic SPMe formulation discussed by Marquis et al. (2019)**. The model includes:

- Spherical solid-phase lithium diffusion in a representative particle of each electrode.
- One-dimensional electrolyte concentration dynamics across the negative electrode, separator, and positive electrode.
- Open-circuit voltage (OCV), reaction overpotential, electrolyte concentration effects, and resistive voltage losses.
- Charge/discharge cutoff conditions and CC–CC or CC–CV cycling.

The diffusion equations are discretised using a **finite-volume method**. Time integration uses SciPy's implicit **BDF** solver, with sparse-Jacobian information and **Numba** acceleration for selected numerical kernels.

### Thermal model

The optional **lumped thermal model** tracks an average cell temperature, balancing irreversible heat generation against heat loss to the environment. Temperature-dependent transport parameters are represented using Arrhenius-type relations.

In the current implementation, the reversible-heat coefficient `dUeq_dT` is set to zero; consequently, reversible entropic heating is not explicitly represented.

### SEI growth and capacity loss

The **solid electrolyte interphase (SEI)** is modelled as a growing surface layer that consumes cyclable lithium and contributes to resistance and capacity loss.

Two SEI approaches have been explored in this project:

| Formulation | Status | Notes |
| --- | --- | --- |
| **Solvent-diffusion-limited SEI** | **Primary implementation** | Used by the current lifecycle simulator; selected because its parameterisation is better supported for this project. |
| **Reaction–diffusion SEI** | Separate experimental Git branch (`reaction-diffusion-sei`) | Implemented as an alternative, but its parameters are largely empirical and require additional calibration data. It is **not** a selectable mode in the current API. |

The active diffusion-limited implementation uses a temperature-dependent solvent diffusivity and an SEI flux inversely proportional to film thickness.

The growth rate is then related to the SEI formation flux and film material parameters. The model also tracks lithium-inventory loss. This is a simplified ageing mechanism: it does not capture every source of real battery degradation, such as lithium plating or mechanical particle cracking - that mechanisms are up to possible future updates. SEI growth was chosen as the primary degradation mechanism because it is a contributor to battery ageing across a wide range of operating conditions, whereas the relative importance of other degradation mechanisms depends more strongly on different factors.

### Model parameters and scientific basis

The project draws on several sources, including:

- **Marquis et al. (2019):** theoretical/asymptotic basis for SPMe equations.
- **Chen et al. (2020):** LG M50-type electrode OCV functions and electrochemical parameterisation.
- **O'Kane et al. (2022):** temperature/activation-energy parameter references used in developing the model.

Actual parameter values are defined in `backend/physics/config.py`. The current configuration also contains assumptions and provisional values, especially for SEI. These sources therefore describe the model's scientific basis, not proof of complete cell-specific validation.

### Cycling and output

Supported lifecycle protocols:

- **CC–CC:** constant-current charging and constant-current discharging.
- **CC–CV:** constant-current charging followed by a constant-voltage hold, then constant-current discharging.

Outputs include discharge capacity, physical SoH, SEI thickness/growth, lithium-loss estimates, and optional hybrid SoH.

**SoH convention:** the simulator normalises working-current discharge capacity to the **first completed discharge**. This is not automatically the same as an experimental reference performance test (RPT) SoH, which may use a different current or reference capacity.

## 4. Machine learning: Ridge residual learning

The second component compensates for systematic differences between physical predictions and experimental observations. Rather than replacing SPMe, a **Ridge regression** model learns its residual error:

```text
residual = SoH_experimental - SoH_physics
SoH_hybrid = SoH_physics + predicted_residual
```

Residuals are expressed in **percentage points of SoH**.

### Experimental dataset

The ML workflow uses the **Imperial College London battery ageing dataset**, specifically **Experiment 5**, identified in the project as a **0–100% SoC cycling-window experiment**. The training workflow covers eight cells (A–H) and three temperatures: **10 °C, 25 °C, and 40 °C**.
(It is easy to notice that there is only a few batteries, due to lack of open sourse experimental data for specific battery it is one of reasonable options)

The dataset-processing code reads C/10 capacity measurements from reference performance tests and normalises each cell to its cycle-zero capacity. Physical-model SoH values are matched to the experimental cycle counts before computing residuals.

### Training and validation

- **Algorithm:** Ridge regression with engineered features based on cycle count and temperature.
- **Training:** performed offline; the web API loads a saved `.joblib` artifact for prediction.
- **Validation:** leave-one-cell-out evaluation, so each held-out battery is tested without using its measurements for that fold's training.
- **Prediction:** the API can return both the physical SoH curve and its optional residual-corrected version.

To retrain when prepared data are available:

```powershell
python -m backend.ML.Ringe_residual_learning.train_residual_ridge
```

The directory is currently named `Ringe_residual_learning` in the codebase; the spelling is retained in commands to avoid breaking imports and paths. Sorry for mistake, we will change it when we have time.

To reproduce the full data pipeline when the original dataset and compatible physics-model results are available:

```powershell
python -m backend.data.get_data_imperial
python -m backend.data.prepare_imperial_residuals
python -m backend.ML.Ringe_residual_learning.train_residual_ridge
```

The preparation step expects `backend/data/model_soh_3temps.csv` in addition to the experimental files. If physics parameters change, the baseline results and residual targets should be regenerated before retraining.

### ML limitations

The current Ridge model uses **cycle count and temperature**, but does not directly include current, SoC window, or the full charging protocol as input features. It therefore should **not** be interpreted as a universally validated correction for arbitrary cycling regimes. Extrapolation to new currents, temperatures, cycling windows, or long lifetimes requires additional data and evaluation.

The reason for choosing residual learning is pragmatic: it keeps the physics solver and statistical correction separate, making each component easier to test and interpret.
In the future, after possible physics expantion to account for lithium loss and particle cracking we might add more tightly connected ML, that might inprove predictions and physics accuracy on unknown condidtions. However that is a task for the next stage of the project.

## 5. Web interface and API

The frontend uses **HTML, CSS, JavaScript, and Three.js** for visual interaction and **FastAPI** to initiate and retrieve lifecycle simulations.

A simulation can be configured with parameters such as cycle count, temperature, charge/discharge current, CC or CC–CV mode, thermal model, and optional ML correction. Long calculations run as **asynchronous jobs** so the browser can poll for progress instead of blocking on a single HTTP request.

Read more detailed README in frontend folder/

### Main endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Check physics/ML integration readiness. |
| `POST /api/simulate/lifecycle/jobs` | Start an asynchronous lifecycle job. |
| `GET /api/simulate/lifecycle/jobs/{job_id}` | Retrieve job status and results. |
| `POST /api/simulate/lifecycle` | Run synchronously (short runs / compatibility). |
| `POST /api/simulate/step` | Real transient-step API; needs an additional step adapter. |

Example request to `POST /api/simulate/lifecycle/jobs`:

```json
{
  "n_cycles": 10,
  "current": 5.0,
  "charge_current_A": 1.5,
  "temperature_C": 25,
  "cycling_mode": "CC",
  "enable_thermal": true,
  "initial_soc_percent": 100,
  "apply_ml": false
}
```

A successful submission returns a `job_id`; retrieve the outcome using the corresponding `GET` endpoint.

**Current initial-state limitation:** the lifecycle adapter uses fixed initial electrode stoichiometries. It presently accepts **only `initial_soc_percent = 100`** rather than a freely configurable physical initial SoC.

## 6. Repository structure

```text
battery-lifecycle-project/
├── backend/
│   ├── physics/
│   │   ├── config.py                  # Geometry, parameters, initial conditions
│   │   ├── physics.py                 # SPMe, thermal, voltage, SEI components
│   │   ├── simulation_cccc.py         # CC–CC cycles
│   │   └── simulation_cccv.py         # CC–CV cycles
│   ├── battery_api/
│   │   ├── server.py                  # FastAPI routes
│   │   ├── pipeline.py                # Physics + optional ML pipeline
│   │   ├── jobs.py                    # Background job manager
│   │   └── ridge.py                   # Ridge inference
│   ├── ML/Ringe_residual_learning/
│   │   └── train_residual_ridge.py    # Training and held-out-cell validation
│   └── data/                          # Experimental-data processing / benchmarks
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── README.md                      # Detailed frontend controls
├── user_simulation_adapter.py         # Connects lifecycle physics to the API
├── try_battery.py                    # Standalone physics demonstration
├── requirements_backend.txt
├── start_backend.ps1
├── start_frontend.ps1
├── smoke_test_backend.py
└── tests/
```

## 7. Tests and standalone runs

Run the API integration smoke test:

```powershell
python smoke_test_backend.py
```

Run the available physics tests:

```powershell
python -m pytest tests/
```

Run the standalone physics script:

```powershell
python try_battery.py
```

## 8. Limitations and future development

BATT-SIM deliberately uses reduced-order models. Important remaining tasks include:

- *Validation and calibration of physical and SEI parameters against independent LG M50T measurements* - really important.
- Testing predictions across different charge/discharge protocols and SoC windows.
- Expanding the ML dataset and assessing generalisation beyond Imperial College Experiment 5.
- Improving solver performance, including investigation of an **analytical sparse Jacobian**, without weakening tolerances or changing physical assumptions.
- Adding a real stateful physics adapter for the live 3D simulation, rather than illustrative demo telemetry.
- Extending the degradation model if reliable parameters become available for additional mechanisms.
- *Adding other ML models with tighter integration with physics* - especially intersting task

## 9. Technology stack and references

**Numerical/scientific:** Python, NumPy, SciPy, Numba, pandas, scikit-learn (Ridge), joblib, Matplotlib.
**Backend:** FastAPI, Pydantic, Uvicorn.  
**Frontend:** JavaScript, HTML, CSS, Three.js.  
**Tests:** pytest, HTTPX/FastAPI test utilities.

**Scientific starting points:**

- Marquis et al. (2019) — asymptotic derivation and analysis of SPMe.
- Chen et al. (2020) — LG M50 electrochemical parameter set / OCV functions.
- O'Kane et al. (2022) — battery degradation and temperature-dependent parameter references.
- Imperial College London battery ageing dataset — Experiment 5, used for ML residual analysis.

The most important literature used for project:
https://doi.org/10.48550/arXiv.2206.05078
https://doi.org/10.1016/j.jpowsour.2024.234185
https://doi.org/10.48550/arXiv.1905.12553
https://doi.org/10.1088/2516-1083/ac7d31
https://doi.org/10.1149/1945-7111/ab9050
https://doi.org/10.48550/arXiv.2112.02037
https://doi.org/10.1149/1945-7111/ad8548
