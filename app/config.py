"""All settings come from environment variables (or .env). No hard-coded paths or keys."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # service
    app_name: str = "incubrix-broll-engine"
    app_version: str = "1.0.0"
    data_dir: Path = Path("./data")
    database_url: str = ""  # default: sqlite file inside data_dir
    public_base_url: str = "http://localhost:8000"
    api_token: str = ""  # if set, required as Bearer token on /api/v1 routes
    rate_limit_per_minute: int = 30
    worker_threads: int = 2
    log_level: str = "INFO"
    preload_models: bool = True  # load Whisper + CLIP at startup (faster first job)

    # storage: "local" or "s3" (MinIO / any S3-compatible store)
    storage_backend: str = "local"
    s3_endpoint_url: str = "http://minio:9000"
    s3_public_endpoint_url: str = ""  # endpoint used inside presigned links (defaults to s3_endpoint_url)
    s3_bucket: str = "broll-outputs"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"
    link_ttl_seconds: int = 7 * 24 * 3600

    # input limits
    max_audio_seconds: float = 600.0
    max_upload_mb: int = 100
    max_script_chars: int = 12000

    # adapters (each has a "mock" implementation so tests need no keys)
    asr_provider: str = "whisper"          # whisper | mock
    whisper_model: str = "base"
    whisper_align_model: str = "tiny"     # used to time TTS of a known script (words come from the script)
    tts_provider: str = "gemini,kokoro,espeak"   # tried in order: gemini | kokoro | espeak | mock
    gemini_tts_model: str = "gemini-3.8-flash-tts,gemini-3.8-flash-lite-tts,gemini-2.5-flash-preview-tts,gemini-3.1-flash-tts-preview"  # tried in order (free tier: ~10 requests/day each)
    llm_provider: str = "gemini"           # gemini | mock
    gemini_model: str = "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-flash-lite-latest"  # tried in order
    rerank_provider: str = "clip"          # clip | mock
    interpret_provider: str = "gemini"     # gemini | rules  (reads prompt documents: voiceover vs shots vs notes)
    interpret_model: str = "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.5-flash"
    judge_provider: str = "gemini"         # gemini | none  (independent VLM relevance judge)
    judge_model: str = "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.5-flash"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    footage_providers: str = "pixabay,coverr,pexels,nasa,wikimedia,internet_archive,openverse_images"
    ai_image_provider: str = "cloudflare"  # cloudflare | none | mock
    cloudflare_model: str = "@cf/black-forest-labs/flux-1-schnell"
    music_provider: str = "openverse"      # openverse | none | mock
    provider_timeout_seconds: float = 20.0
    llm_timeout_seconds: float = 10.0
    disabled_providers: str = ""           # comma list, used to simulate an outage

    # keys
    gemini_api_key: str = ""
    pixabay_api_key: str = ""
    coverr_api_key: str = ""
    pexels_api_key: str = ""
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""
    freesound_api_key: str = ""
    callback_secret: str = "change-me"

    # selection rule (documented in README / REPORT)
    stock_min_score: float = 0.24        # tau: stock clip used only if CLIP score >= tau
    ai_margin: float = 0.02              # AI image must beat best stock by this margin
    candidates_per_query: int = 6
    queries_per_scene: int = 3           # search queries per scene (2 is faster but measurably less relevant)
    max_scored_candidates: int = 12      # CLIP looks at most at this many per scene (text pre-filter)
    min_scene_seconds: float = 2.0
    max_scene_seconds: float = 8.0

    # render
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    fps: int = 30
    video_encoder: str = "libx264"   # libx264 (CPU) | h264_amf | h264_nvenc | h264_qsv (GPU video engines)
    video_preset: str = "veryfast"
    render_parallel: int = 6         # scenes rendered at once
    multi_ratio_render: bool = True  # decode each clip once and split into all aspect ratios
    hw_decode: bool = True           # let ffmpeg use the GPU video decoder when available
    parallax: bool = True            # 2.5D depth parallax for still images (Depth-Anything-V2-Small)
    ffmpeg_threads: int = 2          # threads per ffmpeg process
    video_crf: int = 23
    music_volume_db: float = -20.0
    seed: int = 1234

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'jobs.db').as_posix()}"

    def provider_list(self) -> list[str]:
        return [p.strip() for p in self.footage_providers.split(",") if p.strip()]

    def is_disabled(self, name: str) -> bool:
        return name in {p.strip() for p in self.disabled_providers.split(",") if p.strip()}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
