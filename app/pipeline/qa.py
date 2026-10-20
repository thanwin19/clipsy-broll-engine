"""Self-QA gate: every output file is checked for black frames, frozen frames, A/V drift and format."""
import re
import subprocess
from pathlib import Path

from app.pipeline import media


def check_video(path: Path, expected_seconds: float, fps: int = 30) -> dict:
    # detection on a 480-px-wide copy: black/frozen frames are just as visible, decoding+filtering is much cheaper
    proc = subprocess.run([media.FFMPEG, "-hide_banner", "-nostdin", "-hwaccel", "auto", "-i", str(path), "-vf",
                           "scale=480:-2:flags=fast_bilinear,blackdetect=d=0.5:pix_th=0.08,freezedetect=n=-55dB:d=2.5",
                           "-an", "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    log = proc.stderr
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d\.]+) black_end:([\d\.]+)", log)]
    freezes = [float(x) for x in re.findall(r"freeze_start: ([\d\.]+)", log)]
    info = media.probe(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    vd, ad = float(v.get("duration", 0)), float(a.get("duration", 0))
    num, den = (v.get("r_frame_rate") or "0/1").split("/")
    result = {
        "file": path.name,
        "video_codec": v.get("codec_name"), "audio_codec": a.get("codec_name"),
        "width": v.get("width"), "height": v.get("height"), "pix_fmt": v.get("pix_fmt"),
        "fps": round(float(num) / float(den or 1), 3),
        "video_seconds": round(vd, 3), "audio_seconds": round(ad, 3),
        "av_drift_seconds": round(abs(vd - ad), 3),
        "narration_seconds": round(expected_seconds, 3),
        "black_segments": black, "freeze_starts": freezes,
    }
    result["passed"] = (result["video_codec"] == "h264" and result["audio_codec"] == "aac" and result["fps"] == fps
                        and result["pix_fmt"] == "yuv420p" and result["av_drift_seconds"] <= 0.1 and not black
                        and not freezes and abs(vd - expected_seconds) <= 0.1)
    return result
