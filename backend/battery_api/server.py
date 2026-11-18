"""FastAPI for existing colleague frontend and real SPMe / Ridge model."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

from .jobs import JobStore
from .pipeline import run_full_pipeline
from .ridge import model_path
from .schemas import LifecycleRequest, StepRequest
from .simulator import load_callable, run_physics_step


JOBS = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    JOBS.shutdown()


app = FastAPI(title="LG M50T SPMe + SEI + Ridge Research API", version="0.1.0", lifespan=lifespan)
origins = os.getenv("BATT_CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=[x.strip() for x in origins if x.strip()],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"]
)


def readiness():
    try:
        sim = load_callable("BATT_SIM_ADAPTER") is not None
        step = load_callable("BATT_STEP_ADAPTER") is not None
        error = None
    except Exception as exc:
        sim, step, error = False, False, f"{type(exc).__name__}: {exc}"
    return sim, step, error


@app.get("/api/health")
def health():
    sim, step, error = readiness()
    return {
        "status": "ok" if sim else "integration_required",
        "simulation_ready": sim,
        "step_ready": step,
        "ml_ready": model_path().is_file(),
        "error": error,
    }


def require_ready(payload: LifecycleRequest):
    sim, _, error = readiness()
    if not sim:
        raise HTTPException(503, detail=error or "Configure BATT_SIM_ADAPTER to run the real SPMe model")
    if payload.apply_ml and not model_path().is_file():
        raise HTTPException(503, detail=f"Trained Ridge model missing: {model_path()}. Disable apply_ml to run SPMe only.")


@app.post("/api/simulate/lifecycle/jobs", status_code=202)
def start_lifecycle_job(request: LifecycleRequest):
    require_ready(request)
    try:
        job_id = JOBS.submit(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(429, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/simulate/lifecycle/jobs/{job_id}"}


@app.get("/api/simulate/lifecycle/jobs/{job_id}")
def get_lifecycle_job(job_id: str):
    found = JOBS.status(job_id)
    if found is None:
        raise HTTPException(404, detail="Unknown job or server was restarted")
    return found


@app.post("/api/simulate/lifecycle")
def lifecycle_compatibility(request: LifecycleRequest):
    """Compatible with the original frontend; long runs can hit browser timeouts."""
    require_ready(request)
    try:
        return run_full_pipeline(request.model_dump())
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/simulate/step")
def step(request: StepRequest):
    try:
        return run_physics_step(request.model_dump())
    except NotImplementedError as exc:
        raise HTTPException(501, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}") from exc
