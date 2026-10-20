"""HTTP API. All routes under /api/v1 plus /health, /ready, the demo UI (/) and local file links (/files)."""
import hashlib
import json
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from app import db, jobs
from app.config import get_settings
from app.errors import ERROR_CODES, ApiError, install_handlers
from app.logs import correlation_id_var, get_logger, setup_logging
from app.pipeline import media
from app.pipeline.narration import voice_catalog
from app.pipeline.voicestyle import catalog as style_catalog
from app.pipeline.interpret import looks_structured
from app.pipeline.planner import script_to_scenes
from app.storage import get_storage

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger("api")

app = FastAPI(title="IncuBrix Script/Voiceover to B-roll Video Engine", version=settings.app_version,
              description="Turns a voiceover, script or IncuBrix scene list into finished, licence-clean B-roll videos.")
install_handlers(app)
STATIC = Path(__file__).parent / "static"
AUDIO_EXT = {".mp3": "mp3", ".wav": "wav", ".m4a": "m4a"}


# ------------------------------------------------------------------ models
class Captions(BaseModel):
    burn_in: bool = True
    srt: bool = True


class SceneIn(BaseModel):
    text: str
    visual_hint: str | None = None


class BrollRequest(BaseModel):
    audio_url: HttpUrl | None = Field(None, description="mp3, wav or m4a, up to 10 minutes")
    script: str | None = Field(None, description="Plain-text script; narration is generated with TTS. Text in "
                                                "[square brackets] is a shot description for the words after it and is not spoken.")
    scenes: list[SceneIn] | None = Field(None, description="IncuBrix scene list: text plus optional visual hint")
    voice: str = Field("en-IN-female", description="Text input only: en-IN-female, en-US-male, hi-IN-female ... or a "
                                                    "specific voice from GET /api/v1/voices, e.g. gemini:Charon, kokoro:af_bella")
    voice_speed: float = Field(1.0, ge=0.75, le=1.25, description="Speaking speed for generated narration (0.75-1.25)")
    voice_style: Literal["natural", "soft", "cute", "warm", "deep", "calm", "energetic", "storyteller", "news", "auto"] = Field(
        "natural", description="Voice character: picks a fitting voice + speed + audio shaping. 'auto' chooses from the script")
    clean_audio: bool = Field(True, description="Uploaded voiceovers: noise reduction + loudness normalisation")
    aspect_ratios: list[Literal["16:9", "9:16", "1:1"]] = Field(default_factory=lambda: ["16:9"], min_length=1)
    style: str = "bright, modern, real-world footage"
    avoid: list[str] = Field(default_factory=list)
    captions: Captions = Field(default_factory=Captions)
    music: bool = True
    language: str | None = Field(None, description="Optional ISO code hint for the voiceover, e.g. en, hi, ta")
    callback_url: HttpUrl | None = Field(None, description="POSTed when the job ends; signed with X-IncuBrix-Signature")


class SwapRequest(BaseModel):
    candidate_id: str | None = Field(None, description="One of the scene's alternatives in timeline.json")
    query: str | None = Field(None, description="Or a new search query for this scene")


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


# ------------------------------------------------------------------ middleware
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def context(request: Request, call_next):
    cid = request.headers.get("X-Correlation-Id") or uuid.uuid4().hex
    correlation_id_var.set(cid)
    start = time.perf_counter()
    if request.url.path.startswith("/api/v1") and request.method == "POST":
        ip = request.client.host if request.client else "?"
        q, now = _hits[ip], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= settings.rate_limit_per_minute:
            r = JSONResponse(ApiError("RATE_LIMITED").to_dict(), status_code=429)
            r.headers["X-Correlation-Id"] = cid
            return r
        q.append(now)
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = cid
    if not request.url.path.startswith(("/files", "/static")):
        log.info("request", extra={"extra_fields": {"method": request.method, "path": request.url.path,
                                                     "status": response.status_code,
                                                     "ms": round((time.perf_counter() - start) * 1000, 1)}})
    return response


def auth(authorization: str | None = Header(None)):
    if settings.api_token and authorization != f"Bearer {settings.api_token}":
        raise ApiError("UNAUTHORIZED")


# ------------------------------------------------------------------ helpers
def _validate(req: dict, has_upload: bool):
    has_text = bool((req.get("script") or "").strip()) or bool(req.get("scenes"))
    if not (req.get("audio_url") or has_upload or req.get("script") is not None or req.get("scenes") is not None):
        raise ApiError("NO_INPUT")
    if not (req.get("audio_url") or has_upload) and not has_text:
        raise ApiError("EMPTY_SCRIPT")
    if req.get("scenes") is not None and not any((s.get("text") or "").strip() for s in req["scenes"]):
        raise ApiError("EMPTY_SCRIPT", "Every scene text is empty")
    if len(req.get("script") or "") > settings.max_script_chars:
        raise ApiError("SCRIPT_TOO_LONG", details={"max_chars": settings.max_script_chars})


def _check_audio_file(path: Path, filename: str):
    ext = Path(filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise ApiError("UNSUPPORTED_FILE_TYPE", details={"filename": filename, "allowed": list(AUDIO_EXT)})
    try:
        info = media.probe(path)
    except ApiError:
        raise ApiError("AUDIO_UNREADABLE", details={"filename": filename})
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise ApiError("UNSUPPORTED_FILE_TYPE", "File has no audio stream", {"filename": filename})
    dur = float(info.get("format", {}).get("duration") or 0)
    if dur > settings.max_audio_seconds:
        raise ApiError("AUDIO_TOO_LONG", details={"seconds": round(dur, 1), "max": settings.max_audio_seconds})


def _accept(req: dict, idem: str | None, body_hash: str, kind="broll", parent=None):
    job, created = db.create_job(req, idem, body_hash, correlation_id_var.get(), kind, parent)
    if not created and job.request_hash != body_hash:
        raise ApiError("IDEMPOTENCY_CONFLICT", details={"job_id": job.id})
    if created:
        jobs.submit(job.id)
    body = {"job_id": job.id, "status": job.status, "status_url": f"/api/v1/jobs/{job.id}"}
    return JSONResponse(body, status_code=202, headers={"Location": body["status_url"],
                                                        "Idempotent-Replay": "false" if created else "true"})


MULTIPART_DOC = {"requestBody": {"required": True, "content": {
    "application/json": {"schema": {"$ref": "#/components/schemas/BrollRequest"}},
    "multipart/form-data": {"schema": {"type": "object", "properties": {
        "file": {"type": "string", "format": "binary", "description": "Voiceover (mp3, wav, m4a)"},
        "request": {"type": "string", "description": "JSON BrollRequest (without audio_url)"}}}}}}}


# ------------------------------------------------------------------ routes
@app.post("/api/v1/broll/jobs", status_code=202, response_model=JobAccepted, dependencies=[Depends(auth)],
          openapi_extra=MULTIPART_DOC, tags=["broll"],
          responses={409: {"description": "IDEMPOTENCY_CONFLICT"}, 413: {}, 415: {}, 422: {}, 429: {}})
async def create_broll_job(request: Request, idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    """Create a B-roll job (202 + job_id). Send JSON, or multipart with `file` + `request` JSON."""
    ctype = request.headers.get("content-type", "")
    upload_path = None
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        raw = form.get("request") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise ApiError("VALIDATION_ERROR", "`request` field must be JSON")
        f = form.get("file")
        hasher = hashlib.sha256(raw.encode())
        if f is not None and getattr(f, "filename", None):
            content = await f.read()
            if len(content) > settings.max_upload_mb * 1024 * 1024:
                raise ApiError("FILE_TOO_LARGE", details={"max_mb": settings.max_upload_mb})
            hasher.update(content)
            ext = Path(f.filename).suffix.lower()
            up = settings.data_dir / "uploads"
            up.mkdir(parents=True, exist_ok=True)
            upload_path = up / f"{hasher.hexdigest()[:24]}{ext}"
            upload_path.write_bytes(content)
            _check_audio_file(upload_path, f.filename)
        body_hash = hasher.hexdigest()
    else:
        try:
            data = await request.json()
        except Exception:
            raise ApiError("VALIDATION_ERROR", "Body must be JSON or multipart/form-data")
        body_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    try:
        req = BrollRequest(**data).model_dump(mode="json")
    except Exception as e:
        errs = getattr(e, "errors", lambda: [])()
        if any("aspect_ratios" in str(x.get("loc")) for x in errs):
            raise ApiError("INVALID_ASPECT_RATIO", details={"allowed": list(["16:9", "9:16", "1:1"])})
        raise ApiError("VALIDATION_ERROR", details={"errors": [{"loc": list(x.get("loc", [])), "msg": x.get("msg")} for x in errs]})
    if req.get("audio_url"):
        ext = Path(str(req["audio_url"]).split("?")[0]).suffix.lower()
        if ext and ext not in AUDIO_EXT:
            raise ApiError("UNSUPPORTED_FILE_TYPE", details={"url": req["audio_url"], "allowed": list(AUDIO_EXT)})
    if upload_path:
        req["audio_path"] = str(upload_path)
    if req.get("script") and not req.get("scenes") and not looks_structured(req["script"]):
        shots = script_to_scenes(req["script"])
        if shots is not None:
            if not shots:
                raise ApiError("EMPTY_SCRIPT", "Only shot descriptions in [brackets] were given; add words to speak")
            req["scenes"], req["script"] = shots, None
    _validate(req, upload_path is not None)
    v = req.get("voice") or ""
    if ":" in v and v not in {x["id"] for x in voice_catalog()}:
        raise ApiError("VALIDATION_ERROR", f"Unknown voice '{v}'", {"see": "/api/v1/voices"})
    req["aspect_ratios"] = list(dict.fromkeys(req["aspect_ratios"]))
    return _accept(req, idempotency_key, body_hash)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(auth)], tags=["jobs"])
def get_job(job_id: str):
    """Job status (queued, running, succeeded, failed), progress 0-100 and result links."""
    job = db.get_job(job_id)
    if not job:
        raise ApiError("JOB_NOT_FOUND", details={"job_id": job_id})
    return jobs.job_view(job)


@app.post("/api/v1/broll/jobs/{job_id}/scenes/{scene_index}/swap", status_code=202, response_model=JobAccepted,
          dependencies=[Depends(auth)], tags=["broll"])
def swap_scene(job_id: str, scene_index: int, body: SwapRequest,
               idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    """Replace one scene's clip; only that scene's segments are re-rendered. Returns a new job."""
    parent = db.get_job(job_id)
    if not parent or parent.kind != "broll":
        raise ApiError("JOB_NOT_FOUND", details={"job_id": job_id})
    if parent.status != "succeeded":
        raise ApiError("JOB_NOT_READY", details={"status": parent.status})
    req = body.model_dump() | {"scene_index": scene_index}
    with db.session() as s:
        from sqlalchemy import func, select
        n = s.scalar(select(func.count()).select_from(db.Job).where(db.Job.parent_job_id == job_id)) or 0
    req["_version"] = n + 2
    h = hashlib.sha256(json.dumps({**body.model_dump(), "scene": scene_index, "parent": job_id}, sort_keys=True).encode()).hexdigest()
    return _accept(req, idempotency_key, h, kind="swap", parent=job_id)


@app.get("/api/v1/voices", tags=["meta"])
def voices():
    """Voices for text input. Gemini voices use the free cloud quota; Kokoro voices run locally without limits."""
    return {"default": "en-IN-female", "generic": ["en-IN-female", "en-IN-male", "en-US-female", "en-US-male",
                                                    "en-GB-female", "en-GB-male", "hi-IN-female", "hi-IN-male", "ta-IN-female"],
            "styles": style_catalog(), "voices": voice_catalog()}


@app.get("/api/v1/errors", tags=["meta"])
def error_codes():
    """All error codes this service returns."""
    return {code: {"http_status": st, "retryable": r, "description": d} for code, (st, r, d) in ERROR_CODES.items()}


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@app.get("/ready", tags=["meta"])
def ready():
    checks = {"database": db.db_ok(), "ffmpeg": media.ffmpeg_ok()}
    try:
        checks["storage"] = get_storage().ok()
    except Exception:
        checks["storage"] = False
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@app.get("/files/{key:path}", include_in_schema=False)
def files(key: str):
    root = (settings.data_dir / "outputs").resolve()
    p = (root / key).resolve()
    if not str(p).startswith(str(root)) or not p.exists():
        raise ApiError("NOT_FOUND")
    return FileResponse(p)


@app.get("/", include_in_schema=False)
def ui():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.on_event("startup")
def _startup():
    import importlib
    # import heavy native libraries once in the main thread; first imports racing in worker threads can deadlock
    for mod in ("numpy", "cv2", "PIL.Image", "torch", "open_clip", "faster_whisper"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            log.warning("preload of %s failed: %s", mod, e)
    db.init_db()
    if settings.preload_models:
        import threading

        def warm():
            try:
                from app.pipeline.narration import get_asr
                from app.pipeline.rerank import get_scorer
                asr, scorer = get_asr(), get_scorer()
                if hasattr(asr, "_load"):
                    asr._load()
                    get_asr(for_alignment=True)._load()
                if hasattr(scorer, "_load"):
                    scorer._load()
                log.info("models preloaded")
            except Exception as e:
                log.warning("model preload failed: %s", e)
        threading.Thread(target=warm, daemon=True).start()
    jobs.recover()
    log.info("service started")
