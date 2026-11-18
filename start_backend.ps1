# Run in PowerShell from the root of battery-lifecycle-project
$env:BATT_SIM_ADAPTER = "user_simulation_adapter:run_lifecycle"
$env:BATT_RIDGE_MODEL_PATH = "backend/ML/Ringe_residual_learning/residual_ridge_model.joblib"
python -m uvicorn backend.battery_api.server:app --host 127.0.0.1 --port 8000
