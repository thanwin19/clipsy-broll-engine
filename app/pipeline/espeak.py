"""eSpeak NG text-to-speech (offline, unlimited) for languages the other engines cannot voice locally:
Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, ... Robotic, but never runs out.
Uses the eSpeak NG library shipped with espeakng-loader (already a Kokoro dependency) through ctypes.
"""
import ctypes
import threading
import wave
from pathlib import Path

_lock = threading.Lock()
_lib = None
_rate = 22050
_buf: list[bytes] = []

# locale prefix -> eSpeak NG voice
VOICES = {"ta": "ta", "te": "te", "kn": "kn", "ml": "ml", "bn": "bn", "mr": "mr", "gu": "gu", "pa": "pa",
          "hi": "hi", "ur": "ur", "or": "or", "as": "as", "ne": "ne", "si": "si", "en": "en-us"}

CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)


@CALLBACK
def _on_audio(wav, n, events):
    if n > 0 and wav:
        _buf.append(ctypes.string_at(wav, n * 2))
    return 0


def _load():
    global _lib, _rate
    if _lib is None:
        import espeakng_loader
        lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
        data = Path(espeakng_loader.get_data_path())
        lib.espeak_Initialize.restype = ctypes.c_int
        _rate = lib.espeak_Initialize(2, 0, str(data.parent).encode(), 0)    # 2 = AUDIO_OUTPUT_SYNCHRONOUS
        if _rate <= 0:
            _rate = lib.espeak_Initialize(2, 0, str(data).encode(), 0)
        if _rate <= 0:
            raise RuntimeError("eSpeak NG failed to initialise")
        lib.espeak_SetSynthCallback(_on_audio)
        _lib = lib
    return _lib


def supports(locale: str) -> bool:
    return locale.split("-")[0].lower() in VOICES


def synthesize(text: str, locale: str, out_wav: Path, words_per_minute: int = 150) -> Path:
    voice = VOICES.get(locale.split("-")[0].lower())
    if not voice:
        raise ValueError(f"eSpeak NG has no voice for {locale}")
    with _lock:   # the eSpeak NG library is not thread-safe
        lib = _load()
        if lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError(f"eSpeak NG voice '{voice}' unavailable")
        lib.espeak_SetParameter(1, words_per_minute, 0)                     # 1 = espeakRATE
        _buf.clear()
        data = text.encode("utf-8")
        lib.espeak_Synth(ctypes.c_char_p(data), ctypes.c_size_t(len(data) + 1), 0, 0, 0, 1, None, None)  # 1 = UTF8
        lib.espeak_Synchronize()
        pcm = b"".join(_buf)
        _buf.clear()
    if not pcm:
        raise RuntimeError("eSpeak NG produced no audio")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_rate)
        w.writeframes(pcm)
    return out_wav
