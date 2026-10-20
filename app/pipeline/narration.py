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
    _model = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_name = get_settings().whisper_model
        self.version = f"faster-whisper/{self.model_name}"

    def _load(self):
        with WhisperASR._lock:
            if WhisperASR._model is None:
                from faster_whisper import WhisperModel
                WhisperASR._model = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                                 download_root=str(get_settings().data_dir / "models"))
        return WhisperASR._model

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


def get_asr():
    return MockASR() if get_settings().asr_provider == "mock" else WhisperASR()


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

    def _chunk(self, text: str, limit: int = 1500) -> list[str]:
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
        pcm = b""
        for chunk in self._chunk(text):
            prompt = f"Read this aloud as a clear, warm documentary narrator with a natural {accent} accent:\n\n{chunk}"
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICES[gender]}}}}}
            last = None
            for model in [m.strip() for m in self.model.split(",") if m.strip()] * 2:
                try:
                    r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                   headers={"x-goog-api-key": self.key}, json=body, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                    pcm += base64.b64decode(data)
                    self.version = model
                    last = None
                    break
                except Exception as e:  # retry transient failures
                    last = e
            if last:
                raise ApiError("TTS_FAILED", "Gemini TTS failed", {"reason": str(last)[:300]})
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


def get_tts():
    return MockTTS() if get_settings().tts_provider == "mock" else GeminiTTS()
