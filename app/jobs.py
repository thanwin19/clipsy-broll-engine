"""Background job runner (thread pool) with progress, crash recovery and signed callbacks."""
import hashlib
import hmac
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from app import db
from app.config import get_settings
from app.errors import ApiError
from app.logs import correlation_id_var, get_logger, job_id_var
from app.pipeline import engine  # imported here (main thread): concurrent first imports in worker threads can deadlock

log = get_logger("jobs")
_pool: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def pool() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=get_settings().worker_threads, thread_name_prefix="job")
    return _pool


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _callback(job_id: str, url: str):
    job = db.get_job(job_id)
    payload = json.dumps(job_view(job), separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json",
               "X-IncuBrix-Signature": sign(payload, get_settings().callback_secret),
               "X-Correlation-Id": job.correlation_id or ""}
    for attempt in range(3):
        try:
            httpx.post(url, content=payload, headers=headers, timeout=10).raise_for_status()
            log.info("callback delivered")
            return
        except Exception as e:
            log.warning("callback attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)


def _execute(job_id: str):
    job = db.get_job(job_id)
    job_id_var.set(job_id)
    correlation_id_var.set(job.correlation_id)
    req = json.loads(job.request_json)
    db.update_job(job_id, status="running", progress=1, stage="starting")
    log.info("job started")
    last = {"p": -1}

    def progress(p: int, stage: str):
        p = max(1, min(99, int(p)))
        if p != last["p"]:
            last["p"] = p
            db.update_job(job_id, progress=p, stage=stage)

    try:
        if job.kind == "swap":
            result = engine.swap_scene(job.parent_job_id, job_id, int(req["scene_index"]), req, progress)
        else:
            result = engine.run_job(job_id, req, progress)
        db.update_job(job_id, status="succeeded", progress=100, stage="done", result_json=result)
        log.info("job succeeded")
    except ApiError as e:
        db.update_job(job_id, status="failed", stage="failed", error_json=e.to_dict()["error"])
        log.warning("job failed: %s %s", e.code, e.message)
    except Exception as e:
        log.exception("job crashed")
        db.update_job(job_id, status="failed", stage="failed", error_json={
            "code": "INTERNAL_ERROR", "message": str(e)[:500], "retryable": True, "details": {"type": type(e).__name__}})
    if req.get("callback_url"):
        _callback(job_id, req["callback_url"])


def submit(job_id: str):
    pool().submit(_execute, job_id)


def recover():
    """Requeue jobs left queued/running by a restart."""
    for status in ("running", "queued"):
        for job in db.jobs_with_status(status):
            db.update_job(job.id, status="queued", stage="requeued after restart")
            submit(job.id)


def job_view(job) -> dict:
    out = {"job_id": job.id, "kind": job.kind, "status": job.status, "progress": job.progress, "stage": job.stage,
           "created_at": job.created_at.isoformat() if job.created_at else None,
           "updated_at": job.updated_at.isoformat() if job.updated_at else None,
           "result": None, "error": None}
    if job.parent_job_id:
        out["parent_job_id"] = job.parent_job_id
    if job.result_json:
        out["result"] = json.loads(job.result_json)
    if job.error_json:
        out["error"] = json.loads(job.error_json)
    return out
