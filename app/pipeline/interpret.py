"""Smart prompt reader: turns a free-form video prompt / script document into scenes.

People paste whole briefs: titles, notes, timings, "Visual Prompt:", "Text Overlay:", "Voiceover:" ...
Only the voiceover must be spoken. This module separates, per scene:
  spoken   - the exact words the narrator says
  visual   - the shot description (used as the scene's visual hint)
  overlay  - on-screen text (shown as a title, not spoken)
and drops everything else (titles, notes, durations, music/SFX cues, platform notes).

Primary: an LLM (Gemini lite, fast). Fallback: a rule-based reader for labelled formats.
"""
import json
import re

import httpx

from app.config import get_settings
from app.logs import get_logger

log = get_logger("interpret")

STRUCTURE = re.compile(
    r"(?im)^\s*(#{1,6}\s|---+\s*$|\*\*?\s*scene\b|scene\s*\d+\b|shot\s*\d+\b)"
    r"|\b(visual(\s*prompt)?|shot\s*description|camera|b-?roll|footage|voice\s*-?\s*over|v\.?o\.?|narrat(or|ion)|"
    r"text\s*overlay|on-?\s*screen(\s*text)?|caption|music|sfx|sound\s*effects?|transition)\b\s*(\([^)]*\))?\s*\**\s*:")


def looks_structured(script: str) -> bool:
    """True when the text is a prompt document (labels, headings, scene blocks), not plain narration."""
    return bool(STRUCTURE.search(script or ""))


PROMPT = """You convert a video brief into scenes for a B-roll video generator.
Read the whole document. For each scene/section output:
- "spoken": the EXACT words the narrator/voiceover says (copy them verbatim, without quote marks, labels,
  markdown or stage directions such as "(fast, energetic)"). Empty if that section has no voiceover.
- "visual": the shot description (what the camera should show), cleaned of markdown. Empty if none.
- "overlay": on-screen text / text overlay / title card text for that scene, short. Empty if none.
Ignore and DROP: document titles, introductions about the script, timings like (0:00-0:15), music, SFX,
transitions, platform notes, hashtags, production notes.
If the whole document contains NO spoken narration at all, write a short natural voiceover for each scene
(1-2 sentences, matching its visual) and set "narration_written": true. Otherwise never invent narration.
Return ONLY JSON: {"scenes": [{"spoken": str, "visual": str, "overlay": str}], "narration_written": bool,
"ignored": [short list of what was dropped]}

DOCUMENT:
\"\"\"%s\"\"\""""


def _clean(t: str) -> str:
    t = re.sub(r"(?m)^\s*>\s?", "", t or "")      # markdown quote marker only at line start ("A > B" kept)
    t = re.sub(r"[*_`#\[\]]+", " ", t)
    t = re.sub(r"^\s*[-•]\s*", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" \"“”'")


def interpret_llm(script: str) -> dict:
    s = get_settings()
    body = {"contents": [{"parts": [{"text": PROMPT % script[:20000]}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
    last = None
    for model in [m.strip() for m in s.interpret_model.split(",") if m.strip()]:
        try:
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": s.gemini_api_key}, json=body, timeout=40)
            if r.status_code in (404, 429, 503):
                last = RuntimeError(f"{model}: HTTP {r.status_code}")
                continue
            r.raise_for_status()
            text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"] if not p.get("thought"))
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            data["model"] = model
            return data
        except Exception as e:
            last = e
    raise RuntimeError(f"prompt reader LLM failed: {last}")


LABEL = re.compile(r"(?im)^[\s*\-•]*\**\s*(?P<label>visual(?:\s*prompt)?|shot(?:\s*description)?|camera|b-?roll|footage|"
                   r"voice\s*-?\s*over|v\.?o\.?|narrat(?:or|ion)|script|dialogue|text\s*overlay|on-?\s*screen(?:\s*text)?|"
                   r"caption|title\s*card|music|sfx|sound\s*effects?|transition|duration|notes?)\b[^:\n]*:\**\s*")
SPLIT = re.compile(r"(?im)^\s*(?:#{1,6}\s.*|---+\s*|\**\s*scene\s*\d+.*|shot\s*\d+.*)$")


def interpret_rules(script: str) -> dict:
    """Fallback for labelled briefs when the LLM is unavailable."""
    blocks = [b for b in SPLIT.split(script) if b and b.strip()]
    scenes = []
    for b in blocks:
        parts = {"spoken": "", "visual": "", "overlay": ""}
        matches = list(LABEL.finditer(b))
        for i, m in enumerate(matches):
            content = b[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(b)]
            label = m.group("label").lower()
            if re.match(r"voice|v\.?o|narrat|script|dialogue", label):
                parts["spoken"] += " " + _clean(re.sub(r"^\s*\([^)]*\)\s*", "", content))
            elif re.match(r"visual|shot|camera|b-?roll|footage", label):
                parts["visual"] += " " + _clean(content)
            elif re.match(r"text|on-?\s*screen|caption|title", label):
                parts["overlay"] += " " + _clean(content)
        parts = {k: v.strip() for k, v in parts.items()}
        if any(parts.values()):
            scenes.append(parts)
    return {"scenes": scenes, "narration_written": False, "ignored": ["titles, notes, timings (rule-based reader)"],
            "model": "rules"}


def interpret(script: str) -> tuple[list[dict] | None, dict, list[str]]:
    """Returns (scenes as [{"text", "visual_hint", "overlay"}] or None for plain scripts, info, warnings)."""
    s = get_settings()
    if not looks_structured(script):
        return None, {}, []
    warnings = []
    data = None
    if s.interpret_provider == "gemini" and s.gemini_api_key:
        try:
            data = interpret_llm(script)
        except Exception as e:
            log.warning("LLM prompt reader failed: %s", e)
            warnings.append("PROMPT_READER_FALLBACK: LLM unavailable; rule-based reader used")
    if data is None:
        if not LABEL.search(script):   # no "Voiceover:/Visual:" labels: let the plain/[bracket] script reader handle it
            return None, {}, warnings
        data = interpret_rules(script)
    scenes = []
    for sc in data.get("scenes") or []:
        spoken = _clean(sc.get("spoken", ""))
        if not spoken:
            continue
        scenes.append({"text": spoken, "visual_hint": _clean(sc.get("visual", ""))[:400] or None,
                       "overlay": _clean(sc.get("overlay", ""))[:120] or None})
    info = {"reader": data.get("model"), "scenes_found": len(scenes),
            "narration_written": bool(data.get("narration_written")), "ignored": (data.get("ignored") or [])[:10]}
    if info["narration_written"]:
        warnings.append("NARRATION_WRITTEN: the prompt had no voiceover, so short narration was written for its shots")
    return scenes, info, warnings
