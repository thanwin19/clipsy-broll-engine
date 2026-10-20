"""Visual Beat Planner.

1. Scene boundaries come from the narration's word timestamps (sentence / phrase boundaries),
   so the voice sets the timing and every second is covered.
2. An LLM reads every scene in context and writes a *visual plan*: scene type, concrete visual,
   search queries (word sense resolved), negative terms, safety flags and an AI prompt.
"""
import json
import re
from dataclasses import dataclass, field

import httpx

from app.config import get_settings
from app.logs import get_logger
from app.pipeline.narration import Word

log = get_logger("planner")

SCENE_TYPES = ["literal", "metaphor", "abstract", "statistic", "named_entity", "sensitive"]
CATEGORIES = ["everyday", "business", "technology", "nature", "space_science", "place", "history", "people_free", "food", "health"]


@dataclass
class Scene:
    index: int
    text: str
    start: float
    end: float
    hint: str | None = None
    overlay: str | None = None      # on-screen text from the prompt (shown, not spoken)
    type: str = "literal"
    category: str = "everyday"
    visual: str = ""
    queries: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    stat: dict | None = None
    ai_prompt: str = ""
    ai_allowed: bool = True
    sensitive: bool = False
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Plan:
    scenes: list[Scene]
    mood: str = "calm"
    topic: str = ""
    avoid_terms: list[str] = field(default_factory=list)
    planner: str = "mock"
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------- segmentation
_END = re.compile(r"[.!?।…]+[\"')\]]*$")
_SOFT = re.compile(r"[,;:—–-]$")
_CONJ = {"and", "but", "so", "because", "which", "while", "when", "then", "or", "yet"}


def _split_long(words: list[Word], max_s: float, min_s: float) -> list[list[Word]]:
    if not words or words[-1].end - words[0].start <= max_s or len(words) < 4:
        return [words]
    mid_t = (words[0].start + words[-1].end) / 2
    best, best_cost = None, 1e9
    for i in range(1, len(words) - 1):
        left = words[i - 1].end - words[0].start
        right = words[-1].end - words[i].start
        if left < min_s or right < min_s or i < 3 or len(words) - i < 3:
            continue
        soft = _SOFT.search(words[i - 1].text) is not None
        conj = words[i].text.lower().strip(",.") in _CONJ
        cost = abs(words[i].start - mid_t) - (2.5 if soft else 0) - (1.2 if conj else 0)
        if cost < best_cost:
            best, best_cost = i, cost
    if best is None:
        return [words]
    return _split_long(words[:best], max_s, min_s) + _split_long(words[best:], max_s, min_s)


def segment(words: list[Word], total: float, min_s: float, max_s: float) -> list[list[Word]]:
    sentences, cur = [], []
    for w in words:
        cur.append(w)
        if _END.search(w.text):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    pieces = [p for snt in sentences for p in _split_long(snt, max_s, min_s)]
    # merge pieces shorter than the minimum scene length into a neighbour (no flashing clips)
    merged: list[list[Word]] = []
    for p in pieces:
        if merged and (p[-1].end - p[0].start < min_s or merged[-1][-1].end - merged[-1][0].start < min_s) \
                and (p[-1].end - merged[-1][0].start) <= max_s * 1.6:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    return merged


def scenes_from_words(words: list[Word], total: float, min_s: float, max_s: float,
                      groups: list[list[Word]] | None = None, hints: list[str | None] | None = None) -> list[Scene]:
    groups = groups or segment(words, total, min_s, max_s)
    scenes = []
    for i, g in enumerate(groups):
        start = 0.0 if i == 0 else round(g[0].start, 3)
        h = hints[i] if hints else None
        hint, overlay = h if isinstance(h, tuple) else (h, None)
        scenes.append(Scene(index=i, text=" ".join(w.text for w in g).strip(), start=start, end=0.0,
                            hint=hint, overlay=overlay, words=g))
    for i, sc in enumerate(scenes):  # contiguous: each scene ends where the next starts; last ends at audio end
        sc.end = scenes[i + 1].start if i + 1 < len(scenes) else round(total, 3)
    return scenes


# ------------------------------------------------------------- LLM annotation
PROMPT = """You are the visual director for a B-roll video. For each narration scene, decide what FOOTAGE
should be shown. Footage comes from stock libraries (search by English keywords), simple animated
stat cards, or AI-generated still images.

Rules:
- Resolve word sense from context ("bank" = river bank vs finance; "cloud" computing = data centers, not sky;
  "python" programming = code on screen, not a snake; "value" in money saving = coins in a jar, savings).
- Queries: 3-5 short ENGLISH stock-footage search queries (2-4 words each), most concrete first, things a
  camera can film. Even when the narration is Hindi/Tamil/other, queries must be English.
- negative: words for footage that would be wrong/misleading for this scene (the other word sense, etc).
- type: one of literal, metaphor, abstract, statistic, named_entity, sensitive.
  * statistic = the line states a number/percentage/growth; give stat {{"value": "40%", "label": "short label"}}.
  * named_entity = mentions a real person, company or brand; list them in entities; never depict their face
    or logo; queries should show neutral context (e.g. "electric car charging" not the person).
  * sensitive = death, disaster, violence, illness, grief; queries must be calm and respectful
    (e.g. "candles memorial", "rain on window", "empty street"), no graphic content.
- category: one of {categories}.
- ai_prompt: a prompt for a photorealistic still image that would illustrate the scene. NO people's faces,
  NO text, NO logos, NO real persons. Set ai_allowed=false for named_entity and sensitive scenes.
- visual: one sentence describing the ideal shot (used to score candidate clips).
- The user asked to AVOID: {avoid}. Never propose such footage; add related terms to avoid_terms
  (e.g. alcohol -> beer, wine, whisky, bar, cocktail).
- Style requested: {style}.
- mood: one of upbeat, calm, corporate, inspiring, sombre, playful (for background music).

Full narration for context:
\"\"\"{transcript}\"\"\"

Scenes (index, text, optional user visual hint):
{scenes}

Return ONLY JSON:
{{"topic": str, "mood": str, "avoid_terms": [str],
  "scenes": [{{"i": int, "type": str, "category": str, "visual": str, "queries": [str], "negative": [str],
              "entities": [str], "stat": {{"value": str, "label": str}} | null, "ai_prompt": str,
              "ai_allowed": bool, "sensitive": bool}}]}}
"""


class GeminiPlanner:
    name = "gemini"

    def __init__(self):
        s = get_settings()
        self.key, self.model = s.gemini_api_key, s.gemini_model

    def annotate(self, scenes: list[Scene], transcript: str, style: str, avoid: list[str]) -> dict:
        listing = "\n".join(f'{s.index}. "{s.text}"' + (f'  [hint: {s.hint}]' if s.hint else "") for s in scenes)
        prompt = PROMPT.format(categories=", ".join(CATEGORIES), avoid=", ".join(avoid) or "nothing",
                               style=style or "real-world footage", transcript=transcript[:12000], scenes=listing)
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}
        last = None
        for model in [m.strip() for m in self.model.split(",") if m.strip()]:
            gen = dict(body["generationConfig"])
            if model.startswith("gemini-3"):
                gen["thinkingConfig"] = {"thinkingLevel": "low"}
            for _ in range(1):  # one try per model, then the next (lite) model: overloaded models fail fast
                try:
                    r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                   headers={"x-goog-api-key": self.key}, json={**body, "generationConfig": gen},
                                   timeout=get_settings().llm_timeout_seconds + 0.6 * len(scenes))  # long scripts need longer answers
                    if r.status_code in (404, 400):
                        last = RuntimeError(f"{model}: HTTP {r.status_code}")
                        break
                    r.raise_for_status()
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    self.used_model = model
                    return json.loads(text[text.find("{"): text.rfind("}") + 1])
                except Exception as e:
                    last = e
        raise RuntimeError(f"gemini planner failed: {last}")


STOP = set("""a an the and or but so of to in on at for with from by is are was were be been being this that these
those it its as into over under than then there their they we you your our i he she his her them us very just
not no can will would could should may might must have has had do does did about more most much many such
every each all any some what which who whom how why when where while also only even really make makes made""".split())

DOUBLE = {"bank": ("bank building finance", "river"), "cloud": ("data center servers", "sky"),
          "python": ("programming code screen", "snake"), "mouse": ("computer mouse desk", "rodent"),
          "value": ("coins savings jar", "")}


class MockPlanner:
    """Keyword heuristics; used in tests and as fallback when the LLM is unavailable."""
    name = "heuristic"

    def annotate(self, scenes: list[Scene], transcript: str, style: str, avoid: list[str]) -> dict:
        out = []
        for s in scenes:
            toks = [t.strip(".,!?;:\"'()").lower() for t in s.text.split()]
            content = [t for t in toks if t and t not in STOP and len(t) > 3 and not t.isdigit()]
            queries, negative = [], []
            for t in content:
                if t in DOUBLE:
                    queries.append(DOUBLE[t][0])
                    if DOUBLE[t][1]:
                        negative.append(DOUBLE[t][1])
            if s.hint:
                queries.insert(0, s.hint)
            queries += [" ".join(content[i:i + 2]) for i in range(0, min(len(content), 6), 2)]
            stat = None
            m = re.search(r"(\d+(?:\.\d+)?\s?%|\$?\d[\d,\.]*\s?(?:million|billion|crore|lakh)?)", s.text)
            typ = "statistic" if m and re.search(r"\d", m.group(0)) else "literal"
            if typ == "statistic":
                stat = {"value": m.group(0).strip(), "label": " ".join(content[:4])}
            out.append({"i": s.index, "type": typ, "category": "everyday", "visual": s.hint or " ".join(content[:6]),
                        "queries": [q for q in dict.fromkeys(queries) if q][:4] or ["city timelapse"],
                        "negative": negative, "entities": [], "stat": stat,
                        "ai_prompt": "photorealistic " + " ".join(content[:6]), "ai_allowed": True, "sensitive": False})
        return {"topic": "", "mood": "calm", "avoid_terms": list(avoid), "scenes": out}


def plan_scenes(scenes: list[Scene], transcript: str, style: str, avoid: list[str]) -> Plan:
    s = get_settings()
    warnings = []
    planner = GeminiPlanner() if s.llm_provider == "gemini" and s.gemini_api_key else MockPlanner()
    try:
        data = planner.annotate(scenes, transcript, style, avoid)
    except Exception as e:
        log.warning("LLM planner failed, using heuristic planner: %s", e)
        warnings.append(f"PLANNER_FALLBACK: LLM planner unavailable ({type(e).__name__}); keyword heuristics used")
        planner = MockPlanner()
        data = planner.annotate(scenes, transcript, style, avoid)
    by_i = {d.get("i"): d for d in data.get("scenes", [])}
    fallback = MockPlanner().annotate(scenes, transcript, style, avoid)["scenes"]
    for sc in scenes:
        d = by_i.get(sc.index) or fallback[sc.index]
        sc.type = d.get("type") if d.get("type") in SCENE_TYPES else "literal"
        sc.category = d.get("category") or "everyday"
        sc.visual = d.get("visual") or sc.text
        sc.queries = [q for q in (d.get("queries") or []) if isinstance(q, str) and q.strip()][:5] or fallback[sc.index]["queries"]
        sc.negative = d.get("negative") or []
        sc.entities = d.get("entities") or []
        sc.stat = d.get("stat") if sc.type == "statistic" else None
        sc.sensitive = bool(d.get("sensitive")) or sc.type == "sensitive"
        sc.ai_allowed = bool(d.get("ai_allowed", True)) and sc.type not in ("named_entity", "sensitive") and not sc.sensitive
        sc.ai_prompt = d.get("ai_prompt") or sc.visual
    avoid_terms = list(dict.fromkeys([*avoid, *(data.get("avoid_terms") or [])]))
    mood = data.get("mood") or "calm"
    if any(sc.sensitive for sc in scenes):
        mood = "sombre"
    return Plan(scenes=scenes, mood=mood, topic=data.get("topic", ""), avoid_terms=avoid_terms,
                planner=f"{planner.name}" + (f"/{getattr(planner, 'used_model', '')}" if planner.name == "gemini" else ""), warnings=warnings)


SHOT = re.compile(r"\[([^\[\]]*)\]")


def script_to_scenes(script: str) -> list[dict] | None:
    """One script box, with shot descriptions inline: text in [square brackets] is a shot description
    for the words that follow it and is never spoken.  "[city skyline at dawn] Our city wakes up."
    Header lines such as "Scene 2 (5-10 sec):" are dropped from the narration."""
    if "[" not in script:
        return None
    script = re.sub(r"(?im)^\s*scene\s*\d+\s*(\([^)]*\))?\s*[:.-]?\s*$", "", script)
    parts = SHOT.split(script)              # text, hint, text, hint, text ...
    scenes, hint = [], None
    for i, part in enumerate(parts):
        if i % 2 == 1:
            hint = re.sub(r"(?i)^\s*(shot|visual|b-?roll)\s*:\s*", "", part).strip() or None
            continue
        text = " ".join(part.split())
        if text:
            scenes.append({"text": text, "visual_hint": hint})
            hint = None
        elif hint and scenes and not scenes[-1]["visual_hint"]:
            pass
    return scenes or []
