# clipsy

**Your story. The right shot.**

clipsy turns a **voiceover, a script, or an IncuBrix scene list** into finished, licence-clean B-roll videos,
scene by scene: footage that matches the *meaning* of each line, timed to the narration, with captions,
background music and a full record of where every clip came from.

```
POST /api/v1/broll/jobs  ->  video_16x9.mp4, video_9x16.mp4, video_1x1.mp4, captions.srt, timeline.json, manifest.json
```

The web app (served at `/`) has an editorial landing page, **AI Studio** (script or voiceover in, live progress,
the finished video, storyboard and AI Director), a multitrack **Editor** (preview, B-roll / captions / voiceover /
music tracks, inspector with alternatives and clip swap), **Export**, **Projects** and **Settings**.

## What makes it different

| Idea | What it does |
|---|---|
| **Visual Beat Planner** | Scenes are cut on sentence/phrase boundaries from word timestamps (the voice sets the timing). An LLM then writes a *visual plan* per scene: type (literal, metaphor, abstract, statistic, named entity, sensitive), word sense ("cloud computing" → data centers, not sky), concrete search queries and negative terms. |
| **Look-then-pick** | Candidate clips from several sources are *looked at*: CLIP scores each thumbnail against the scene plan, minus a penalty for negative/avoid concepts. |
| **Global assignment** | Best (scene, clip) pairs are taken first across the whole video; a clip is never used twice. |
| **Smart source routing** | Sources are chosen per scene category (space → NASA first, landmarks → Wikimedia, everyday → Pixabay/Coverr/Pexels). Statistics become animated stat cards. AI images (FLUX.1-schnell) compete with stock and win only by a measurable margin. |
| **Safety Guard** | Avoid list enforced on tags and via CLIP negatives; OpenCV face detection when faces are to be avoided; no AI for named people/brands/sensitive scenes; sensitive topics get calm footage and sombre/soft music. |
| **Subject-aware reframing** | 9:16 / 1:1 crops are centred on the visual subject (spectral-residual saliency), not the frame centre. |
| **Self-QA gate** | Every output is checked for black frames, frozen frames, A/V drift, codec, fps and pixel format; failing scenes are auto-replaced once. |
| **Swap a clip** | Each scene is its own normalised segment; swapping one scene re-renders only that segment and re-joins the rest without re-encoding. |
| **Independent judge** | A different model (Gemini vision) looks at a frame of every finished scene and rates it 1–5 against the narration; stored per scene (`judge`) and summarised in the manifest. |
| **2.5D parallax** | Still images (AI images, Openverse photos) get a depth-based camera move from Depth-Anything-V2-Small, so they read as real footage. |
| **Voices** | Pick any of ~56 voices (`GET /api/v1/voices`: Gemini cloud voices or unlimited local Kokoro voices), set `voice_speed` 0.75–1.25, and uploaded voiceovers are cleaned (noise reduction + −16 LUFS loudness). |
| **Fast rendering** | Each clip is decoded once for all aspect ratios, segments render in parallel, the GPU encoder is used when present (`VIDEO_ENCODER=h264_amf`/`h264_nvenc`/`h264_qsv`), and audio is mixed once and stream-copied. |
| **Prompt reader** | Paste a whole brief: only the voiceover is spoken, visual prompts become shot descriptions, on-screen text becomes titles; titles, timings and notes are dropped. |
| **Languages** | English, Hindi and Tamil voiceovers; scripts can be narrated in Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi (translated, footage follows the original meaning). |
| **Creative controls** | Voice styles (soft, cute, warm, deep, calm, energetic, storyteller, news, auto), target duration, background music mood (lo-fi, calm, upbeat, cinematic, corporate, none), caption colour / size / font / position / box with live preview. |
| **Resilience** | Per-job circuit breaker: a source that fails is skipped and the job finishes with other sources, then cards. |

## Architecture

```
            ┌──────────── FastAPI (/api/v1, /health, /ready, UI at /) ────────────┐
 client ──► │ validate · Idempotency-Key · X-Correlation-Id · rate limit · auth    │
            └───────────────┬───────────────────────────────────────┬────────────┘
                            │ 202 job_id                            │ GET /api/v1/jobs/{id}
                            ▼                                       │
                   jobs table (SQLite / Postgres) ◄─────────────────┘
                            │
                   worker thread pool ─► pipeline (app/pipeline/engine.py)
   1 ingest ─ 2 narration (faster-whisper ASR | Gemini TTS) ─ 3 Beat Planner (Gemini | heuristic)
   4 search (Pixabay · Coverr · Pexels · NASA · Wikimedia · Internet Archive · Openverse images, circuit breaker) ─ 5 CLIP rerank + Safety Guard
   6 route + global assignment (stock | AI image | stat/text card) ─ 7 render segments (ffmpeg: normalise,
   colour, reframe, captions) ─ 8 music (Openverse CC0/CC-BY | procedural) + ducking ─ 9 Self-QA
                            │
                            ▼
              S3-compatible storage (MinIO) → presigned download links (+ signed callback)
```

Every external service sits behind a small adapter selected in configuration, each with a **mock**
(`ASR_PROVIDER=mock`, `TTS_PROVIDER=mock`, `LLM_PROVIDER=mock`, `RERANK_PROVIDER=mock`,
`FOOTAGE_PROVIDERS=mock`, `MUSIC_PROVIDER=mock`), so all tests run with no keys and no network.

## Setup (under 10 commands)

```bash
git clone <repo> && cd broll-engine
cp .env.example .env            # add your free keys (Gemini, Pixabay, Coverr, optional Pexels/Cloudflare)
docker compose up --build       # API + Postgres + MinIO; open http://localhost:8000
```

Without Docker (Windows/macOS/Linux, needs Python 3.11 and ffmpeg):

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/pip on Linux/macOS)
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

## Tests (one command, no API keys)

```bash
docker compose run --rm tests        # or locally: python -m pytest -q
```

Unit tests (segmentation, coverage, captions, frame-exact timing, planner, errors, HMAC), integration
tests (API contract, idempotency, error codes, OpenAPI file == running service) and end-to-end tests
(script → video, scene list → video → swap a clip, audio upload, silent audio, provider outage).

## Batch runner

```bash
python -m cli.batch_runner testpack/inputs --api http://localhost:8000 --out testpack/outputs --concurrency 2
```

Accepts a folder or file: `*.mp3|wav|m4a` (uploaded), `*.txt` (script), `*.json` (request body, optional
`"file"` for an audio upload), `*.jsonl` (one request per line). Writes every output plus `summary.csv`.

## API

See `openapi.yaml` (OpenAPI 3.1, generated from the service with `python -m cli.export_openapi`, and
checked against the running service by a test) or `/docs` on a running server.

```bash
curl -X POST localhost:8000/api/v1/broll/jobs -H "Content-Type: application/json" \
     -H "Idempotency-Key: demo-1" -d '{"script":"Saving money starts with a plan.","aspect_ratios":["16:9","9:16"]}'
curl localhost:8000/api/v1/jobs/<job_id>
curl -X POST localhost:8000/api/v1/broll/jobs -F file=@voiceover.mp3 -F 'request={"aspect_ratios":["9:16"]}'
curl -X POST localhost:8000/api/v1/broll/jobs/<job_id>/scenes/2/swap -H "Content-Type: application/json" -d '{"query":"city at night"}'
```

Callbacks: set `callback_url`; the final job JSON is POSTed with header
`X-IncuBrix-Signature: sha256=<HMAC-SHA256(body, CALLBACK_SECRET)>`.

### Error codes

All errors are `{"error": {"code", "message", "retryable", "details"}}`. Also served live at `GET /api/v1/errors`.

| Code | HTTP | Retryable | Meaning |
|---|---|---|---|
| VALIDATION_ERROR | 422 | no | Request body or parameters are invalid |
| NO_INPUT | 422 | no | None of audio_url, upload, script or scenes was given |
| EMPTY_SCRIPT | 422 | no | The script (or every scene text) is empty |
| SCRIPT_TOO_LONG | 422 | no | Script exceeds MAX_SCRIPT_CHARS |
| UNSUPPORTED_FILE_TYPE | 415 | no | Audio must be mp3, wav or m4a |
| FILE_TOO_LARGE | 413 | no | Upload exceeds MAX_UPLOAD_MB |
| AUDIO_TOO_LONG | 422 | no | Audio longer than MAX_AUDIO_SECONDS (10 min) |
| AUDIO_UNREADABLE | 422 | no | Corrupt / undecodable audio |
| NO_SPEECH_DETECTED | 422 | no | Silent or music-only audio |
| INPUT_DOWNLOAD_FAILED | 502 | yes | audio_url could not be downloaded |
| INVALID_ASPECT_RATIO | 422 | no | Ratios must be 16:9, 9:16, 1:1 |
| IDEMPOTENCY_CONFLICT | 409 | no | Idempotency-Key reused with a different body |
| JOB_NOT_FOUND | 404 | no | No job with this id |
| SCENE_NOT_FOUND | 404 | no | No scene with this index |
| JOB_NOT_READY | 409 | yes | Job not finished yet (swap) |
| UNAUTHORIZED | 401 | no | Missing/wrong bearer token |
| RATE_LIMITED | 429 | yes | Too many requests |
| TTS_FAILED | 502 | yes | Narration could not be generated |
| ALL_PROVIDERS_FAILED | 502 | yes | No footage and fallbacks failed |
| RENDER_FAILED | 500 | yes | ffmpeg failed |
| NOT_FOUND | 404 | no | Route not found |
| INTERNAL_ERROR | 500 | yes | Unexpected error |

Non-fatal problems are reported as **warnings** in the job result and `manifest.json`
(e.g. `PROVIDER_DOWN`, `PLANNER_FALLBACK`, `LOW_RELEVANCE`, `CLIP_DOWNLOAD_FAILED`, `QA_AUTOFIX`, `MUSIC_SOURCE_DOWN`).

## Source-selection rule (measurable)

Per scene: (1) statistics → animated stat card. (2) Stock candidates from category-suited sources are
scored with CLIP (thumbnail vs. visual plan + narration; minus penalty when a negative/avoid concept scores
higher). Unsafe candidates are removed. (3) Global assignment takes the highest (scene, clip) pairs first,
each clip once, if score ≥ **τ = `STOCK_MIN_SCORE` (0.24)**. (4) If a scene's best stock score < τ + 0.03 and
AI is allowed, an AI image is generated; it wins only if its CLIP score ≥ stock + **`AI_MARGIN` (0.02)**.
(5) Otherwise the best unused clip within 0.03 of τ (with a `LOW_RELEVANCE` warning), else a text card.
The reason for every choice is stored per scene in `timeline.json`.

## Settings

Every setting is an environment variable (see `.env.example` for the full list with defaults):
service (`DATA_DIR`, `DATABASE_URL`, `PUBLIC_BASE_URL`, `API_TOKEN`, `RATE_LIMIT_PER_MINUTE`, `WORKER_THREADS`,
`LOG_LEVEL`), storage (`STORAGE_BACKEND`, `S3_*`, `LINK_TTL_SECONDS`), limits (`MAX_AUDIO_SECONDS`,
`MAX_UPLOAD_MB`, `MAX_SCRIPT_CHARS`), adapters (`ASR_PROVIDER`, `WHISPER_MODEL`, `TTS_PROVIDER`,
`GEMINI_TTS_MODEL`, `LLM_PROVIDER`, `GEMINI_MODEL`, `RERANK_PROVIDER`, `CLIP_MODEL`, `CLIP_PRETRAINED`,
`FOOTAGE_PROVIDERS`, `AI_IMAGE_PROVIDER`, `CLOUDFLARE_MODEL`, `MUSIC_PROVIDER`, `PROVIDER_TIMEOUT_SECONDS`,
`LLM_TIMEOUT_SECONDS`, `DISABLED_PROVIDERS`), keys (`GEMINI_API_KEY`, `PIXABAY_API_KEY`, `COVERR_API_KEY`,
`PEXELS_API_KEY`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CALLBACK_SECRET`), selection rule
(`STOCK_MIN_SCORE`, `AI_MARGIN`, `CANDIDATES_PER_QUERY`, `MIN_SCENE_SECONDS`, `MAX_SCENE_SECONDS`) and render
(`FFMPEG_BIN`, `FFPROBE_BIN`, `FPS`, `VIDEO_PRESET`, `VIDEO_CRF`, `MUSIC_VOLUME_DB`, `SEED`).

`DISABLED_PROVIDERS=pixabay` switches a source off at runtime (used to test outages).

## Logs and health

Structured JSON logs on stdout with `timestamp`, `level`, `job_id` and `correlation_id`
(from the `X-Correlation-Id` request header, echoed in the response). `GET /health` (liveness) and
`GET /ready` (database, storage, ffmpeg).

## Files

`app/main.py` API · `app/jobs.py` background jobs + callbacks · `app/db.py` jobs/idempotency ·
`app/storage.py` S3/local · `app/pipeline/*` engine (narration, planner, sources, rerank, render, cards,
music, qa) · `app/static/index.html` demo UI · `cli/` batch runner + OpenAPI export · `tests/`.

See `REPORT.md`, `SOURCES.md` and `AI_USE.md`.
