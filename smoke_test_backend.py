"""Integration smoke test; run from the repository root with `python smoke_test_backend.py`."""
from __future__ import annotations
import os
import time
from fastapi.testclient import TestClient

os.environ.setdefault('BATT_SIM_ADAPTER', 'user_simulation_adapter:run_lifecycle')
os.environ.setdefault('BATT_RIDGE_MODEL_PATH', 'backend/ML/Ringe_residual_learning/residual_ridge_model.joblib')

from backend.battery_api.server import app


def main():
    with TestClient(app) as client:
        r = client.get('/api/health')
        assert r.status_code == 200, r.text
        health = r.json()
        assert health['simulation_ready'] is True, health
        assert health['ml_ready'] is True, health
        print('HEALTH OK:', health, flush=True)

        request = dict(n_cycles=2, current=5., charge_current_A=1.5,
                       temperature_C=25., cycling_mode='CC', apply_ml=True)
        res = client.post('/api/simulate/lifecycle/jobs', json=request)
        assert res.status_code == 202, res.text
        job_id = res.json()['job_id']
        timeout = time.monotonic() + 60
        while time.monotonic() < timeout:
            status = client.get('/api/simulate/lifecycle/jobs/' + job_id)
            assert status.status_code == 200, status.text
            payload = status.json()
            if payload['status'] == 'completed':
                output = payload['result']
                assert len(output['cycle']) == 2, output
                for k in ('SOH_physical_percent', 'SOH_hybrid_percent', 'sei_thickness_nm', 'sei_growth_nm', 'capacity_loss_ah'):
                    assert len(output[k]) == 2, k
                print('ASYNC JOB OK:', output['inference']['final'], flush=True)
                break
            if payload['status'] in ('failed','cancelled'):
                raise AssertionError(payload)
            time.sleep(0.2)
        else:
            raise TimeoutError('Job did not finish within smoke test budget')

        bad = client.post('/api/simulate/lifecycle/jobs',json=dict(n_cycles=-1,current=5))
        assert bad.status_code == 422, bad.text
        print('VALIDATION OK', flush=True)
        missing = client.get('/api/simulate/lifecycle/jobs/not-a-real-job')
        assert missing.status_code == 404, missing.text
        print('NOT FOUND OK', flush=True)


if __name__ == '__main__':
    main()
