"""Voice styles: soft, cute, warm, deep, calm, energetic, storyteller, news, plus "auto".

A style is three things layered on the TTS:
  1. the voice whose character fits (Google publishes a character per Gemini voice, e.g. Achernar "soft",
     Leda "youthful", Algenib "gravelly"; Kokoro voices chosen by ear),
  2. a speaking speed,
  3. studio-style shaping with ffmpeg: pitch (rubberband, formants preserved), EQ warmth/brightness,
     compression and a touch of room for storytelling.
"auto" picks a style from the script's content (sad news -> calm, promo -> energetic, story -> storyteller ...).
"""
import re

# style -> gemini (female, male), kokoro (female, male), speed, audio filter chain (before loudness levelling)
STYLES: dict[str, dict] = {
    "natural": {"label": "Natural", "gemini": (None, None), "kokoro": (None, None), "speed": 1.0, "af": ""},
    "soft": {"label": "Soft & gentle", "gemini": ("Achernar", "Enceladus"), "kokoro": ("af_nicole", "am_echo"),
             "speed": 0.95, "af": "lowpass=f=9000,deesser=i=0.4,acompressor=threshold=-26dB:ratio=2:attack=25:release=300"},
    "cute": {"label": "Cute & bright", "gemini": ("Leda", "Puck"), "kokoro": ("af_sky", "am_puck"),
             "speed": 1.05, "af": "rubberband=pitch=1.10:formant=preserved,treble=g=2:f=4000"},
    "warm": {"label": "Warm & friendly", "gemini": ("Sulafat", "Achird"), "kokoro": ("af_heart", "am_michael"),
             "speed": 0.98, "af": "bass=g=3:f=160,treble=g=-1:f=6000"},
    "deep": {"label": "Deep & cinematic", "gemini": ("Gacrux", "Algenib"), "kokoro": ("af_alloy", "am_onyx"),
             "speed": 0.95, "af": "rubberband=pitch=0.92:formant=preserved,bass=g=4:f=120"},
    "calm": {"label": "Calm & respectful", "gemini": ("Vindemiatrix", "Schedar"), "kokoro": ("af_river", "am_adam"),
             "speed": 0.90, "af": "acompressor=threshold=-24dB:ratio=2:attack=30:release=400"},
    "energetic": {"label": "Energetic & upbeat", "gemini": ("Laomedeia", "Fenrir"), "kokoro": ("af_bella", "am_fenrir"),
                  "speed": 1.12, "af": "treble=g=2:f=5000,acompressor=threshold=-20dB:ratio=3:attack=5:release=120"},
    "storyteller": {"label": "Storyteller", "gemini": ("Despina", "Algieba"), "kokoro": ("bf_emma", "bm_fable"),
                    "speed": 0.93, "af": "bass=g=2:f=180,aecho=0.8:0.85:35:0.10"},
    "news": {"label": "News anchor", "gemini": ("Kore", "Charon"), "kokoro": ("af_sarah", "bm_george"),
             "speed": 1.03, "af": "equalizer=f=3000:t=q:w=1:g=3,acompressor=threshold=-20dB:ratio=3:attack=10:release=150"},
}

_AUTO = [
    ("calm", r"\b(died|death|dead|killed|flood(s|ed)?|disaster|tragedy|tragic|grief|victims?|war|earthquake|mourn\w*|"
             r"funeral|cancer|loss|lost their)\b"),
    ("energetic", r"\b(buy|sale|offer|discount|introducing|meet the|get yours|limited|launch\w*|order now|shop|"
                  r"brand new|amazing|deal)\b|!.*!"),
    ("storyteller", r"\b(once upon|years ago|one day|every morning|long ago|legend|story|grandfather|grandmother|"
                    r"he said|she said)\b"),
    ("news", r"\b(today|announced|according to|reported|government|officials?|minister|quarter|analysts?|breaking)\b"),
    ("warm", r"\b(how to|step \w+|first,|tutorial|recipe|tip(s)?|hack(s)?|guide)\b"),
]


def auto_style(text: str) -> str:
    t = (text or "").lower()
    scores = {name: len(re.findall(rx, t)) for name, rx in _AUTO}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "natural"


def resolve(style: str | None, voice: str, text: str) -> tuple[str, str, float, str]:
    """-> (style used, voice id for TTS, speed factor, audio filter). An explicitly named voice is kept;
    the style then only adds its speed and shaping."""
    style = (style or "natural").lower()
    if style == "auto":
        style = auto_style(text)
    cfg = STYLES.get(style, STYLES["natural"])
    gender = "male" if ("male" in voice and "female" not in voice) else "female"
    idx = 1 if gender == "male" else 0
    out_voice = voice
    if ":" not in voice and cfg["gemini"][idx]:          # generic voice like en-IN-female -> the style's voice
        out_voice = f"gemini:{cfg['gemini'][idx]}|kokoro:{cfg['kokoro'][idx]}"
    return style, out_voice, cfg["speed"], cfg["af"]


def catalog() -> list[dict]:
    return [{"id": k, "label": v["label"], "speed": v["speed"],
             "gemini_voices": [x for x in v["gemini"] if x], "kokoro_voices": [x for x in v["kokoro"] if x]}
            for k, v in STYLES.items()] + [{"id": "auto", "label": "Auto (picked from the script)", "speed": None}]
