"""Independent relevance judge: a vision-language model (Gemini, not CLIP) rates every finished scene 1-5.

One frame from the middle of each rendered scene is sent with its narration line; all scenes of a video go
in one request, so a whole video costs one free-tier call. The score is stored per scene in timeline.json
(`judge`), averaged in manifest.json, and compared with human ratings by testpack/make/metrics.py.
"""
import base64
import json
from pathlib import Path

import httpx

from app.config import get_settings
from app.logs import get_logger
from app.pipeline import media

log = get_logger("judge")

PROMPT = """You are an independent reviewer of B-roll videos. Each numbered image is a frame from one scene,
shown while the narrator says the line given with it. Rate how well the footage illustrates the meaning of
the line, as a professional video editor would:
5 = clearly shows what the line means; 4 = good, fitting B-roll; 3 = loosely related but acceptable;
2 = weak or generic, barely related; 1 = unrelated, misleading or inappropriate.
Animated text/number cards that show the statistic or words of the line count as 4.
Return ONLY JSON: [{"i": <scene number>, "score": <1-5>, "why": "<max 12 words>"}]"""


def scene_frame(segment: Path, seconds: float, dest: Path) -> Path:
    media.ffmpeg("-ss", f"{max(0.0, seconds / 2):.2f}", "-i", segment, "-frames:v", "1", "-vf", "scale=512:-2",
                 "-q:v", "4", dest)
    return dest


def judge_scenes(scenes, segment_paths: dict[int, Path], work: Path) -> tuple[dict[int, dict], str | None, str | None]:
    """Returns ({scene_index: {"score", "why"}}, model_used, warning)."""
    s = get_settings()
    if s.judge_provider != "gemini" or not s.gemini_api_key:
        return {}, None, None
    (work / "judge").mkdir(exist_ok=True)
    results, used, warning = {}, None, None
    batch = 20  # scenes per request
    for start in range(0, len(scenes), batch):
        part = scenes[start:start + batch]
        parts = [{"text": PROMPT}]
        for sc in part:
            seg = segment_paths.get(sc.index)
            if not seg or not seg.exists():
                continue
            img = scene_frame(seg, sc.duration, work / "judge" / f"scene_{sc.index:03d}.jpg")
            parts.append({"text": f"Scene {sc.index}. Narration: \"{sc.text}\""})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img.read_bytes()).decode()}})
        body = {"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
        models = [m.strip() for m in s.judge_model.split(",") if m.strip()]
        for attempt, model in enumerate(models + models[:1]):   # last try: first model again after a pause
            try:
                if attempt == len(models):
                    import time
                    time.sleep(4)   # per-minute free-tier limits recover quickly
                r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                               headers={"x-goog-api-key": s.gemini_api_key}, json=body, timeout=60)
                if r.status_code in (404, 429, 503):
                    continue
                r.raise_for_status()
                text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought"))
                for item in json.loads(text[text.find("["): text.rfind("]") + 1]):
                    score = int(item.get("score", 0))
                    if 1 <= score <= 5:
                        results[int(item["i"])] = {"score": score, "why": str(item.get("why", ""))[:120]}
                used = model
                break
            except Exception as e:
                log.warning("judge model %s failed: %s", model, e)
        else:
            warning = "JUDGE_UNAVAILABLE: the independent VLM judge could not score some scenes (quota or error)"
    return results, used, warning
