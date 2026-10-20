"""Tests run fully offline with mock adapters: no API keys, no network, no model downloads."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="broll_test_")
os.environ.update({
    "DATA_DIR": _tmp, "ASR_PROVIDER": "mock", "TTS_PROVIDER": "mock", "LLM_PROVIDER": "mock",
    "RERANK_PROVIDER": "mock", "FOOTAGE_PROVIDERS": "mock", "AI_IMAGE_PROVIDER": "none",
    "MUSIC_PROVIDER": os.environ.get("MUSIC_PROVIDER", "mock"), "STOCK_MIN_SCORE": "0.18", "API_TOKEN": "",
    "RATE_LIMIT_PER_MINUTE": "1000", "GEMINI_API_KEY": "", "PIXABAY_API_KEY": "", "COVERR_API_KEY": "",
    "CLOUDFLARE_API_TOKEN": "", "DATABASE_URL": "", "WORKER_THREADS": "2",
})
os.environ.setdefault("STORAGE_BACKEND", "local")
