"""Narration: speech-to-text (word timestamps) and text-to-speech. Adapters + mocks."""
import base64
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import get_settings
from app.errors import ApiError
from app.pipeline import media


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Narration:
    audio_path: Path            # 48 kHz wav used for the final mix
    words: list[Word]
    duration: float
    language: str = "en"
    transcript: str = ""
    providers: dict = field(default_factory=dict)


# ---------------------------------------------------------------- ASR
class WhisperASR:
    name = "faster-whisper"
    _models: dict = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().whisper_model
        self.version = f"faster-whisper/{self.model_name}"

    def _load(self):
        with WhisperASR._lock:
            if self.model_name not in WhisperASR._models:
                from faster_whisper import WhisperModel
                WhisperASR._models[self.model_name] = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8",
                    download_root=str(get_settings().data_dir / "models"))
        return WhisperASR._models[self.model_name]

    def transcribe(self, wav: Path, language: str | None = None) -> tuple[list[Word], str]:
        model = self._load()
        with WhisperASR._lock:  # one transcription at a time keeps memory bounded on small servers
            segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True,
                                              language=language, beam_size=1)
            words = []
            for seg in segments:
                if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
                    continue
                for w in seg.words or []:
                    t = w.word.strip()
                    if t:
                        words.append(Word(t, float(w.start), float(w.end)))
        return words, info.language


class MockASR:
    """Deterministic: reads a sidecar .txt next to the audio (tests) or uses a fixed sentence."""
    name = "mock-asr"
    version = "mock/1"

    def transcribe(self, wav: Path, language: str | None = None) -> tuple[list[Word], str]:
        side = Path(str(wav).rsplit(".", 1)[0] + ".txt")
        src = getattr(self, "source_text", None) or (side.read_text() if side.exists() else
                                                     "Saving money starts with a simple plan. Track every expense.")
        dur = media.duration(wav)
        if media.mean_volume_db(wav) < -60:
            return [], "en"
        return even_words(src, 0.0, dur), "en"


def even_words(text: str, start: float, end: float) -> list[Word]:
    toks = text.split()
    if not toks:
        return []
    step = (end - start) / len(toks)
    return [Word(t, start + i * step, start + (i + 1) * step - 0.02) for i, t in enumerate(toks)]


def align_script(script: str, asr_words: list[Word]) -> list[Word]:
    """Give the exact script words the ASR timings (captions show what was written, not mis-hearings).
    Matched tokens take the ASR time; unmatched runs are spread over the gap between their neighbours."""
    import difflib
    toks = script.split()
    if not toks or not asr_words:
        return asr_words
    norm = lambda t: re.sub(r"[^\w]", "", t.lower())
    sm = difflib.SequenceMatcher(a=[norm(t) for t in toks], b=[norm(w.text) for w in asr_words], autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(toks)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            times[a + k] = (asr_words[b + k].start, asr_words[b + k].end)
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag in ("replace", "insert") and a2 > a1:
            if b2 > b1:
                start, end = asr_words[b1].start, asr_words[b2 - 1].end
            else:
                start = times[a1 - 1][1] if a1 > 0 and times[a1 - 1] else asr_words[0].start
                nxt = next((times[j] for j in range(a2, len(toks)) if times[j]), None)
                end = nxt[0] if nxt else asr_words[-1].end
            step = max(end - start, 0.05 * (a2 - a1)) / (a2 - a1)
            for k in range(a1, a2):
                times[k] = (start + (k - a1) * step, start + (k - a1 + 1) * step)
    out, last = [], 0.0
    for t, tm in zip(toks, times):
        st, en = tm if tm else (last, last + 0.2)
        st = max(st, last)
        out.append(Word(t, st, max(en, st + 0.01)))
        last = out[-1].end
    return out


def get_asr(for_alignment: bool = False):
    """for_alignment: the words are known (TTS of a script), only timings are needed -> smaller, faster model."""
    s = get_settings()
    if s.asr_provider == "mock":
        return MockASR()
    return WhisperASR(s.whisper_align_model if for_alignment else None)


# ---------------------------------------------------------------- TTS
VOICES = {"female": "Kore", "male": "Puck"}
ACCENTS = {"en-IN": "Indian English", "en-US": "American English", "en-GB": "British English",
           "hi-IN": "Hindi", "ta-IN": "Tamil"}


class GeminiTTS:
    name = "gemini-tts"

    def __init__(self):
        s = get_settings()
        self.model, self.key, self.timeout = s.gemini_tts_model, s.gemini_api_key, 120
        self.version = self.model

    def _chunk(self, text: str, limit: int = 4000) -> list[str]:
        sents = re.split(r"(?<=[.!?।])\s+", text.strip())
        chunks, cur = [], ""
        for snt in sents:
            if len(cur) + len(snt) > limit and cur:
                chunks.append(cur)
                cur = ""
            cur = (cur + " " + snt).strip()
        if cur:
            chunks.append(cur)
        return chunks

    def synthesize(self, text: str, voice: str, out_wav: Path) -> Path:
        if not self.key:
            raise ApiError("TTS_FAILED", "GEMINI_API_KEY is not set")
        locale = "-".join(voice.split("-")[:2]) if voice else "en-IN"
        gender = "male" if "male" in voice and "female" not in voice else "female"
        accent = ACCENTS.get(locale, "neutral English")
        import hashlib
        cache_dir = get_settings().data_dir / "tts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        models = [m.strip() for m in self.model.split(",") if m.strip()]
        pcm = b""
        for chunk in self._chunk(text):
            key = hashlib.sha256(f"{voice}|{chunk}".encode()).hexdigest()[:32]
            cached = cache_dir / f"{key}.pcm"
            if cached.exists():                       # same text + voice never spends quota twice
                pcm += cached.read_bytes()
                continue
            # plain text only: newer TTS models read any style instruction aloud
            prompt = chunk
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES[gender]}}}}}
            errors = []
            for attempt, model in enumerate(models + models[:2]):
                try:
                    r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                   headers={"x-goog-api-key": self.key}, json=body, timeout=self.timeout)
                    if r.status_code in (404, 429):       # model retired / daily free quota used -> next model
                        errors.append(f"{model}: HTTP {r.status_code}")
                        continue
                    r.raise_for_status()
                    data = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                    audio = base64.b64decode(data)
                    cached.write_bytes(audio)
                    pcm += audio
                    self.version = model
                    break
                except Exception as e:
                    errors.append(f"{model}: {type(e).__name__}")
            else:
                raise ApiError("TTS_FAILED", "All TTS models failed or their free daily quota is used up; "
                               "try again later or send a voiceover file", {"attempts": errors[-6:]})
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        return out_wav


class MockTTS:
    """Generates a soft tone of realistic length (0.38 s per word) so the full pipeline runs with no keys."""
    name = "mock-tts"
    version = "mock/1"

    def synthesize(self, text: str, voice: str, out_wav: Path) -> Path:
        dur = max(1.5, len(text.split()) * 0.38)
        media.ffmpeg("-f", "lavfi", "-i", f"sine=frequency=220:duration={dur:.2f}", "-af", "volume=0.2",
                     "-ac", "1", "-ar", "24000", out_wav)
        return out_wav


class KokoroTTS:
    """Local, unlimited fallback: Kokoro-82M (Apache-2.0 weights) via kokoro-onnx, CPU only."""
    name = "kokoro-82m"
    version = "kokoro-v1.0-onnx"
    _model = None
    _lock = threading.Lock()
    # locale+gender -> (voice, language). Kokoro has no Indian-English voice; British voices are used for en-IN.
    VOICE_MAP = {("en-US", "female"): ("af_heart", "en-us"), ("en-US", "male"): ("am_michael", "en-us"),
                 ("en-GB", "female"): ("bf_emma", "en-gb"), ("en-GB", "male"): ("bm_george", "en-gb"),
                 ("en-IN", "female"): ("bf_emma", "en-gb"), ("en-IN", "male"): ("bm_george", "en-gb"),
                 ("hi-IN", "female"): ("hf_alpha", "hi"), ("hi-IN", "male"): ("hm_omega", "hi")}

    def _load(self):
        with KokoroTTS._lock:
            if KokoroTTS._model is None:
                from kokoro_onnx import Kokoro
                d = get_settings().data_dir / "models" / "kokoro"
                if not (d / "kokoro-v1.0.onnx").exists():
                    d.mkdir(parents=True, exist_ok=True)
                    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
                    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
                        with httpx.stream("GET", base + f, follow_redirects=True, timeout=600) as r, open(d / f, "wb") as out:
                            r.raise_for_status()
                            for chunk in r.iter_bytes(1 << 20):
                                out.write(chunk)
                KokoroTTS._model = Kokoro(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
        return KokoroTTS._model

    def synthesize(self, text: str, voice: str, out_wav: Path) -> Path:
        import numpy as np
        import soundfile as sf
        locale = "-".join((voice or "en-IN-female").split("-")[:2])
        gender = "male" if "male" in voice and "female" not in voice else "female"
        if (locale, gender) not in self.VOICE_MAP:
            raise ApiError("TTS_FAILED", f"Local TTS has no voice for {locale}; send a voiceover file instead",
                           {"voice": voice, "supported": sorted({k[0] for k in self.VOICE_MAP})})
        kv, lang = self.VOICE_MAP[(locale, gender)]
        model = self._load()
        parts = []
        with KokoroTTS._lock:
            for chunk in re.split(r"(?<=[.!?।])\s+", text.strip()):
                if chunk.strip():
                    audio, sr = model.create(chunk, voice=kv, speed=1.0, lang=lang)
                    parts += [audio, np.zeros(int(sr * 0.18), dtype=audio.dtype)]
        sf.write(str(out_wav), np.concatenate(parts), sr)
        self.version = f"kokoro-v1.0-onnx/{kv}"
        return out_wav


class ChainTTS:
    """Tries each TTS in order (e.g. Gemini, then local Kokoro when the free quota is used up)."""

    def __init__(self, providers: list):
        self.providers = providers
        self.name, self.version, self.warnings = providers[0].name, providers[0].version, []

    def synthesize(self, text: str, voice: str, out_wav: Path) -> Path:
        errors = []
        for i, p in enumerate(self.providers):
            try:
                out = p.synthesize(text, voice, out_wav)
                self.name, self.version = p.name, p.version
                if i > 0:
                    self.warnings.append(f"TTS_FALLBACK: {self.providers[0].name} unavailable "
                                         f"({errors[0][:120]}); narration generated locally with {p.name}")
                return out
            except ApiError as e:
                errors.append(f"{p.name}: {e.message}")
        raise ApiError("TTS_FAILED", "No TTS provider could generate the narration", {"attempts": errors})


def get_tts():
    s = get_settings()
    if s.tts_provider == "mock":
        return MockTTS()
    table = {"gemini": GeminiTTS, "kokoro": KokoroTTS}
    chain = [table[n.strip()]() for n in s.tts_provider.split(",") if n.strip() in table]
    return ChainTTS(chain or [GeminiTTS()])
