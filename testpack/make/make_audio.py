"""Builds the audio test inputs (synthetic TTS voices only - no real person's voice):
   python -m testpack.make.make_audio   (run from the repo root, needs GEMINI_API_KEY)
   python -m testpack.make.make_audio local   (only the local Kokoro voiceovers: 5-minute English, Hindi)"""
import json
import sys
from pathlib import Path

from app.pipeline import media
from app.pipeline.music import procedural_bed
from app.pipeline.narration import GeminiTTS, KokoroTTS

OUT = Path("testpack/inputs")
VOICES = {
    "audio_01_us_tutorial": ("en-US-male", "To change a bicycle tyre, first release the brake and remove the wheel. "
                             "Use two levers to lift the tyre off the rim. Pull out the old tube, check the tyre for "
                             "sharp objects, then fit the new tube and pump it up slowly."),
    "audio_02_gb_travel": ("en-GB-female", "The old harbour town sits at the end of a winding coastal road. Fishing "
                           "boats bob in the water, and narrow lanes lead up to a lighthouse on the cliff. In the "
                           "evening, the whole bay turns orange as the sun goes down."),
    "audio_03_in_startup": ("en-IN-male", "Starting a business is hard. You will build a product, talk to customers "
                            "and fix problems every single day. But when the first customer pays you, every late "
                            "night feels worth it."),
    "audio_04_tamil_nature": ("ta-IN-female", "மழைக்காலத்தில் மலைகள் பச்சையாக மாறுகின்றன. அருவிகள் வேகமாக ஓடுகின்றன. "
                              "பறவைகள் மரங்களில் பாடுகின்றன."),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tts = GeminiTTS()
    for name, (voice, text) in VOICES.items():
        mp3 = OUT / f"{name}.mp3"
        if mp3.exists():
            continue
        wav = tts.synthesize(text, voice, OUT / f"{name}.wav")
        media.ffmpeg("-i", wav, "-c:a", "libmp3lame", "-b:a", "128k", mp3)
        wav.unlink()
        print("made", mp3)
    # m4a variant of one voiceover (format coverage)
    if not (OUT / "audio_05_in_startup.m4a").exists():
        media.ffmpeg("-i", OUT / "audio_03_in_startup.mp3", "-c:a", "aac", "-b:a", "128k", OUT / "audio_05_in_startup.m4a")
    # hard cases: silent audio and music-only audio
    media.ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "12", "-c:a", "libmp3lame", OUT / "hc_silent_audio.mp3")
    bed = procedural_bed(OUT / "music_tmp.wav", 20, "calm")
    media.ffmpeg("-i", bed.path, "-c:a", "libmp3lame", "-b:a", "128k", OUT / "hc_music_only.mp3")
    bed.path.unlink()
    print("done")


def local():
    """Voiceovers made with the local Kokoro-82M model (Apache-2.0, synthetic voices): an uploaded 5-minute
    voiceover (hard case) and a Hindi voiceover. Texts are the same as the matching script inputs."""
    tts = KokoroTTS()
    jobs = {"audio_07_hindi_tutorial": ("hi-IN-female", "09_hindi_tutorial.json"),
            "audio_08_long_5min_agriculture": ("en-US-female", "10_long_5min_agriculture.json")}
    for name, (voice, src) in jobs.items():
        mp3 = OUT / f"{name}.mp3"
        if mp3.exists():
            continue
        text = json.loads((OUT / src).read_text(encoding="utf-8"))["script"]
        wav = tts.synthesize(text, voice, OUT / f"{name}.wav")
        # Kokoro reads the long script in ~3.4 min; slow it (pitch kept) to the 5 minutes the hard case asks for
        tempo = ["-af", "atempo=0.68"] if "5min" in name else []
        media.ffmpeg("-i", wav, *tempo, "-c:a", "libmp3lame", "-b:a", "96k", mp3)
        wav.unlink()
        print("made", mp3)


if __name__ == "__main__":
    local() if sys.argv[1:] == ["local"] else main()
