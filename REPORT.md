# REPORT: Script/Voiceover to B-roll Video Engine

## 1. Problem and approach

The job is to cover narration with footage that matches its *meaning*, timed to the voice, licence-clean
and safe. Keyword search fails on exactly the cases the brief lists ("value" → piles of cash, "cloud
computing" → sky). So the design separates **understanding**, **finding**, **looking** and **choosing**.

| Approach considered | Decision |
|---|---|
| Keyword extraction → stock search → first result | Rejected: the double-meaning failures the brief describes |
| LLM picks clips from titles/tags only | Rejected: tags are noisy, and the LLM never sees the footage |
| **LLM visual plan → multi-source search → CLIP looks at every candidate → global assignment** | **Chosen.** The LLM resolves meaning and writes concrete queries; CLIP checks the actual frames; assignment avoids repeats |
| Text-to-video generation for every scene | Rejected: no free commercial option on CPU, slow, and real footage looks better. AI *images* compete instead |
| One ffmpeg pass for the whole video | Rejected: swapping one clip would re-render everything. **Per-scene normalised segments** are joined with stream copy |

Pipeline: ingest → narration (faster-whisper word timestamps, or Gemini TTS / local Kokoro TTS then
Whisper alignment) → scene cuts on sentence/phrase boundaries → **Beat Planner** (Gemini: type, word
sense, queries, negatives, safety flags, mood) → category-routed search (Pixabay, Coverr, Pexels, NASA,
Wikimedia) with a circuit breaker → **CLIP rerank** with negative/avoid penalties and Safety Guard
(face detection, AI ban for people/brands/sensitive) → **global assignment** and the source router (stock /
AI image / stat card) → partial downloads → per-scene render (1080p, 30 fps, yuv420p, colour/brightness
normalisation, saliency-centred reframing, captions) → music (Openverse CC0/CC-BY or procedural) with
ducking → **Self-QA** (black/frozen frames, drift, format; auto-replace once) → S3-compatible storage (SeaweedFS) + manifest.

## 2. Measurable source-selection rule

Statistics → animated stat card. Otherwise each candidate is scored with CLIP (cosine of thumbnail vs.
visual plan, narration and top queries, minus a penalty when a negative/avoid concept scores higher). Pairs
are assigned globally, highest first, each clip once, if score ≥ τ = 0.24. A scene whose best stock score is
below τ + 0.03 may get an AI image, which wins only if its score ≥ stock + 0.02. Otherwise the best unused
clip within 0.03 of τ is used (with a `LOW_RELEVANCE` warning), else a text card. Each scene's reason is in
`timeline.json`.

## 3. What works (with evidence)

See `testpack/METRICS.md` (generated from recorded outputs by `testpack/make/metrics.py`) and
`testpack/INDEX.md`.

- **Inputs:** mp3/wav/m4a upload or URL (tested with synthetic US/UK/Indian-English and Tamil voiceovers);
  script; script with inline `[shot descriptions]`; IncuBrix scene lists.
- **Timing:** scenes are contiguous from 0 to the narration end; frame counts use cumulative rounding,
  so video length equals narration length (QA reports drift ≤ 0.1 s on every output).
- **Licences:** every clip records source, licence, licence URL, attribution, page link and `ai_generated`.
  Only commercial-use sources are queried; Wikimedia NC/ND/SA and Coverr premium items are filtered out.
- **No repeats:** the repeat-clip rate is 0 across all produced videos (global assignment).
- **Resilience:** provider failures (Coverr rate-limiting, Gemini 503/429) were hit for real during testing;
  jobs finished via other sources/models with `PROVIDER_DOWN`, `PLANNER_FALLBACK` and `TTS_FALLBACK`
  warnings. `DISABLED_PROVIDERS` switches a source off at runtime; an e2e test covers a total outage.
- **Hard cases:** wrong file type (415), corrupt audio, empty script, bad ratio, silent audio and music-only
  audio (`NO_SPEECH_DETECTED`), broken URL (`INPUT_DOWNLOAD_FAILED`) all return clear JSON errors.
  Content hard cases (abstract/stats, named people and brands, double meanings, sad news, 9:16 edge
  subject, avoid faces/alcohol) produce videos; see Hard cases in the form.
- **Swap a clip:** re-renders only the affected scene's segments and re-joins the rest without re-encoding
  (e2e test asserts other scenes unchanged and the swap is faster than the full job).

## 4. What does not work, or is limited (honest list)

- **Free-tier limits drive reliability.** Gemini TTS allows about 10 requests/day per model. During testing
  the quota ran out, and 7 hard-case jobs in run 1 failed with `TTS_FAILED` (kept in
  `testpack/outputs_run1`). Fixed by chaining four TTS models, caching narration, and a local Kokoro-82M
  fallback. Kokoro has no Indian-English voice (British voices are used) and no Tamil voice: Tamil
  *voiceovers* work through Whisper, but Tamil *text* input needs Gemini quota.
- **Bugs found by the test pack** (all fixed, failures kept as evidence): OpenCV 5 removed
  `CascadeClassifier`; the OpenCV face detector is not thread-safe (2 jobs failed in run 2); a TTS model
  read the style prompt aloud; an empty caption file broke a render; a syntax error in a lazily imported
  module left jobs stuck "starting" (imports now happen at startup).
- **AI images** (FLUX via Cloudflare) are implemented but were not enabled during these runs (account ID
  not configured), so the produced videos use stock and cards only.
- **Speed on a laptop CPU:** turnaround is several minutes per minute of video, and slower with parallel jobs
  (the 5-minute test took 26 min while other jobs ran). Partial downloads, search caching and faster LLM
  fallback cut a 18 s test job from 271 s to 123 s.
- **Live demo** runs on the author's laptop behind a Cloudflare quick tunnel (free). Hugging Face Docker
  Spaces now need a paid plan. The demo is only up while the laptop is on.
- **Relevance numbers** here are CLIP scores, not human ratings. Human 1–5 ratings are in the form.
- **Outstanding level is not claimed.** There is no independent VLM agreement study, and Hindi/Tamil
  quality has not been rated by native speakers.

## 5. Numbers

See `testpack/METRICS.md` for per-case values: turnaround, scenes, QA, sources, warnings.
Cost: **$0.00 per minute** (all providers on free tiers, open models on the host CPU).
Hardware: Windows 11 laptop, CPU only.

## 6. What I would do next

1. A small GPU box: faster-whisper and CLIP on GPU, and an open image-to-video model (e.g. Wan 2.1,
   Apache-2.0) for short generated shots.
2. Score 3 frames per clip from a downloaded proxy instead of 1 thumbnail; add a temporal-consistency
   term so consecutive scenes match in palette.
3. Paid TTS/LLM tiers behind the same adapters for predictable quotas; a Redis queue with separate
   worker containers to scale jobs horizontally.
4. An independent VLM (e.g. Qwen2.5-VL, Apache-2.0) judge and a human-rating study to calibrate τ.
