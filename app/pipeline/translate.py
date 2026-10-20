"""Narration language: speak an English script in Tamil (or Hindi, Telugu, ...).

The script's scene texts are translated together in one Gemini call (scene order and count preserved), so
footage planning still sees the original meaning (kept as the scene's visual hint) while the narrator and
captions use the target language.
"""
import json

import httpx

from app.config import get_settings
from app.errors import ApiError
from app.logs import get_logger

log = get_logger("translate")

LANGUAGES = {"en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
             "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu",
             "es": "Spanish", "fr": "French", "de": "German"}

PROMPT = """Translate each numbered narration line into natural, spoken {lang} for a video voiceover.
Keep the meaning and tone, keep numbers as digits (e.g. 40%), keep brand and person names as they are.
Do not add or drop lines. Return ONLY JSON: {{"lines": ["...", "..."]}} with exactly {n} items.

{lines}"""


def translate_lines(lines: list[str], target: str) -> tuple[list[str], str]:
    """Returns (translated lines, model used). Raises ApiError if no model could translate."""
    s = get_settings()
    lang = LANGUAGES.get(target)
    if not lang:
        raise ApiError("VALIDATION_ERROR", f"Unsupported narration_language '{target}'", {"supported": sorted(LANGUAGES)})
    if not s.gemini_api_key:
        raise ApiError("TRANSLATION_FAILED", "Translation needs GEMINI_API_KEY")
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    body = {"contents": [{"parts": [{"text": PROMPT.format(lang=lang, n=len(lines), lines=numbered)}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    last = None
    for model in [m.strip() for m in s.interpret_model.split(",") if m.strip()]:
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": s.gemini_api_key}, json=body, timeout=60)
            if r.status_code in (404, 429, 503):
                last = f"{model}: HTTP {r.status_code}"
                continue
            r.raise_for_status()
            text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought"))
            out = json.loads(text[text.find("{"): text.rfind("}") + 1]).get("lines") or []
            out = [str(x).strip() for x in out]
            if len(out) == len(lines) and all(out):
                return out, model
            last = f"{model}: returned {len(out)} of {len(lines)} lines"
        except Exception as e:
            last = f"{model}: {type(e).__name__}"
    raise ApiError("TRANSLATION_FAILED", f"Could not translate the script into {lang}", {"last_error": last})
