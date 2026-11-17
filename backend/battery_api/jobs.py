"""Single-worker local research jobs. For deployment use persistent queue/storage."""
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from .pipeline import run_full_pipeline


@dataclass
class Job:
    future: Future


class JobStore:
    def __init__(self):
        
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def submit(self, payload: dict) -> str:
        with self._lock:
            outstanding = sum(not j.future.done() for j in self._jobs.values())
            if outstanding >= 3:
                raise RuntimeError("Three calculations are already queued or running")
            job_id = str(uuid4())
            self._jobs[job_id] = Job(future=self._pool.submit(run_full_pipeline, payload))
            return job_id

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        future = job.future
        if future.cancelled():
            return {"job_id": job_id, "status": "cancelled", "result": None}
        if not future.done():
            return {
                "job_id": job_id,
                "status": "running" if future.running() else "queued",
                "result": None,
            }
        try:
            result = future.result()
        except Exception as exc:
            return {"job_id": job_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "result": None}
        return {"job_id": job_id, "status": "completed", "result": result}

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
