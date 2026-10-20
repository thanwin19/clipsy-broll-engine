# AI coding tools used

| Tool | Used for |
|---|---|
| antigravity,chatgtp,gemini,claude | Planning the architecture with me from the brief; writing most of the code (API, job runner, pipeline modules, ffmpeg filter graphs, UI), tests, Dockerfile/compose and first drafts of README / SOURCES / REPORT; debugging (Gemini model retirement, slow downloads, scene-split bug) from logs. |

How I worked with it:
- I chose the overall approach (meaning-first planning, CLIP re-ranking, global assignment, per-scene
  segments for swap-a-clip, stat cards, Self-QA) and the free/commercially-usable provider set.
- Every generated change was run: the full test suite (mock providers) and real end-to-end jobs with
  the live providers. Failures found this way (e.g. `gemini-2.5-flash` no longer available for new
  keys, Pixabay 4K downloads too slow, a scene cut in the middle of a sentence) were fixed and re-run.
- I reviewed the code and can explain and change every part of it.

AI models used *inside* the product (Gemini planner, TTS and judge; faster-whisper; OpenCLIP; Kokoro-82M; eSpeak NG; Depth-Anything-V2-Small; FLUX, not enabled on the demo) are listed in `SOURCES.md`.
