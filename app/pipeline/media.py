"""Thin ffmpeg / ffprobe helpers."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from app.errors import ApiError


def _bin(name: str) -> str:
    from app.config import get_settings
    configured = os.environ.get(f"{name.upper()}_BIN") or getattr(get_settings(), f"{name}_bin")
    return shutil.which(configured) or configured


FFMPEG = _bin("ffmpeg")
FFPROBE = _bin("ffprobe")


def run(cmd: list, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise ApiError("RENDER_FAILED", f"{Path(str(cmd[0])).name} failed",
                       {"stderr_tail": proc.stderr[-1200:]})
    return proc


def ffmpeg(*args, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return run([FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *args], cwd=cwd, timeout=timeout)


def probe(path: Path) -> dict:
    proc = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if proc.returncode != 0:
        raise ApiError("AUDIO_UNREADABLE", "File cannot be decoded", {"stderr_tail": proc.stderr[-400:]})
    return json.loads(proc.stdout or "{}")


def duration(path: Path) -> float:
    info = probe(path)
    d = info.get("format", {}).get("duration")
    if d is None:
        ds = [float(s["duration"]) for s in info.get("streams", []) if s.get("duration")]
        d = max(ds) if ds else 0
    return float(d)


def has_audio(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path).get("streams", []))


def video_stream(path: Path) -> dict | None:
    return next((s for s in probe(path).get("streams", []) if s.get("codec_type") == "video"), None)


def to_wav(src: Path, dest: Path, rate: int = 16000, af: str | None = None) -> Path:
    ffmpeg("-i", src, "-vn", *(["-af", af] if af else []), "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", dest)
    return dest


# voiceover clean-up: remove rumble, reduce steady background noise, then broadcast loudness (-16 LUFS)
CLEAN_FILTER = "highpass=f=70,afftdn=nf=-25:tn=1,loudnorm=I=-16:TP=-1.5:LRA=11"
LEVEL_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


def mean_volume_db(path: Path) -> float:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    m = re.search(r"mean_volume:\s*(-?[\d\.]+|-inf) dB", proc.stderr)
    if not m or m.group(1) == "-inf":
        return -100.0
    return float(m.group(1))


def ffmpeg_ok() -> bool:
    try:
        subprocess.run([FFMPEG, "-version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False
