# REPORT: clipsy, a script/voiceover to B-roll video engine

## 1. Problem and approach

The job is to cover narration with footage that matches its *meaning*, timed to the voice, licence-clean
and safe. Keyword search fails on exactly the cases the brief lists ("value" → piles of cash, "cloud
computing" → sky). So the design separates **understanding**, **finding**, **looking** and **choosing**.

| Approach considered | Decision |
|---|---|
| Keyword extraction → stock search → first result | Rejected: the double-meaning failures the brief describes |
| LLM picks clips from titles/tags only | Rejected: tags are noisy, and the LLM never sees the footage |
| **LLM visual plan → multi-source search → CLIP looks at every candidate → global assignment** | **Chosen.** The LLM resolves meaning and writes concrete queries; CLIP checks the actual frames; assignment avoids repeats |
| Text-to-video generation for every scene | Rejected: no free commercial option on a CPU laptop, slow, and real footage looks better. AI *images* (with depth parallax) compete instead |
| One ffmpeg pass for the whole video | Rejected: swapping one clip would re-render everything. **Per-scene normalised segments** are joined with stream copy |

Pipeline:

1. **Ingest:** upload or URL (mp3/wav/m4a, ≤ 10 min), plain script, script with `[shot]` notes, a pasted
   prompt document (only the voiceover lines are spoken), or an IncuBrix scene list.
2. **Narration:** faster-whisper word timestamps for voiceovers. For text: TTS chain Gemini TTS → local
   Kokoro-82M → local eSpeak NG (Tamil and other Indic text), then Whisper aligns the known words.
3. **Scenes:** cut on sentence/phrase boundaries, 2–8 s, covering every second of narration.
4. **Beat Planner** (Gemini, with a keyword fallback): scene type, word sense, 3 concrete queries, negatives,
   safety flags, mood.
5. **Search:** category-routed over Pixabay, Coverr, Pexels, NASA, Wikimedia Commons, Internet Archive and
   Openverse images. A per-job circuit breaker plus a 10-minute cross-job cooldown handle outages.
6. **CLIP rerank** (OpenCLIP ViT-B-32) with negative/avoid penalties, plus the Safety Guard (face detection
   when faces are avoided; no AI for people, brands or sensitive scenes).
7. **Global assignment** and the source router (stock / AI image / stat card / text card).
8. **Render:** per-scene segments at 1080-class resolution, 30 fps, yuv420p; colour/brightness
   normalisation; saliency-centred reframing for 9:16 and 1:1; captions; each clip is decoded once and split
   into all ratios.
9. **Music** (Openverse CC0/CC-BY or a procedural bed) at -16 LUFS with ducking.
10. **Self-QA:** black/frozen frames, drift, codec and fps checks, with one automatic replace.
11. **Storage:** S3-compatible storage (SeaweedFS), manifest, and links served through the API.
12. **Independent VLM judge:** Gemini scores one frame per scene 1–5 after delivery.

## 2. Measurable source-selection rule

**Statistics** become an animated stat card. Every other candidate is scored with CLIP:

- **Score:** cosine of thumbnail vs. visual plan, narration and top queries, minus a penalty when a negative
  or avoid concept scores higher.
- **Assignment:** pairs are assigned globally, highest first, each clip once, if score ≥ τ = 0.24.
- **AI image:** a scene whose best stock score is below τ + 0.03 may get one, and it wins only if its score
  ≥ stock + 0.02. AI is never used for people, brands or sensitive scenes.
- **Fallbacks:** otherwise the best unused clip within 0.03 of τ is used (with a `LOW_RELEVANCE` warning);
  failing that, a text card.

Each scene's reason and score are in `timeline.json`, alongside the alternatives.

## 3. What works (with evidence)

All evidence is indexed in `testpack/INDEX.md`; numbers are in `testpack/METRICS.md`, generated from
recorded files by `testpack/make/metrics.py`.

- **Inputs:** mp3, wav and m4a uploads, including a 5-minute voiceover. US, UK and Indian-English, Hindi and
  Tamil voiceovers (synthetic voices only). Also scripts, scripts with inline `[shot descriptions]` and
  IncuBrix scene lists.
- **Timing:** scenes are contiguous from 0 to the narration end. Frame counts use cumulative rounding, so
  video length equals narration length; QA reports drift ≤ 0.1 s on every output.
- **Licences:** every clip records source, licence, licence URL, attribution, page link and `ai_generated`.
  Only commercial-use sources are queried; Wikimedia NC/ND/SA and Coverr premium items are filtered out.
- **No repeats:** the repeat-clip rate is 0 across every produced video (global assignment).
- **Resilience:** real outages happened during testing and jobs still finished, with `PROVIDER_DOWN` /
  `PROVIDER_COOLDOWN`, `PLANNER_FALLBACK`, `TTS_FALLBACK` and `MUSIC_SOURCE_DOWN` warnings:
  - Coverr returned 403 for every request on 20 Oct.
  - Openverse timed out.
  - Gemini was slow or rate-limited.

  `POST /api/v1/admin/providers/{name}` switches a source off *while a job runs*. The final live end-to-end
  check (29/29, `testpack/E2E_FINAL_RESULTS.txt`) switched Pixabay off mid-job and the job still finished.
- **Hard cases:** wrong file type (415), corrupt audio, empty script, bad ratio, silent and music-only audio
  (`NO_SPEECH_DETECTED`) and a broken URL (`INPUT_DOWNLOAD_FAILED`) all return clear JSON errors. The content
  hard cases (abstract lines and statistics, named people and brands, double meanings, sad news, a 9:16 edge
  subject, avoiding faces and alcohol) all produce videos.
- **Swap a clip:** only the affected scene's segments are re-rendered; the rest is re-joined without
  re-encoding (16 s for a swap vs. 83 s for the full job in the final e2e check).
- **Integration:** `docker compose up` (API + Postgres + SeaweedFS) and the full test suite in containers run
  on a clean GitHub Actions machine (`.github/workflows/compose.yml`). The author's laptop has no Docker, so
  CI is the evidence for this point.

## 4. What does not work, or is limited (honest list)

- **Bugs the fresh test run found on 20 Oct, both fixed with the run repeated:**
  - *The avoid filter split phrases into words.* The planner also invented avoid terms when the user gave
    none. "instant coffee" then rejected every clip tagged *coffee*, and "gas station" every *charging
    station*, so those scenes silently became text cards. Now a phrase matches only when all its words are
    present, and the LLM may only widen a list the user gave. A regression test covers it.
  - *Thumbnail downloads timed out waiting for a pooled connection* when two jobs ran at once, and a missing
    thumbnail scored 0. Now the wait is separate from the download timeout, with one retry and a log line.

- **Memory on a 7 GB laptop:** this is the main weakness found on 20 Oct.
  - *What happened.* With 2 jobs at once, ffmpeg and Python ran out of memory: the page file cannot grow
    because C: is full, and the API holds about 3.2 GB of models.
  - *Effect in run 4:* 4 cases failed (INTERNAL_ERROR / RENDER_FAILED). Memory errors also tripped the
    footage circuit breaker, so two videos (Hindi tutorial, 5-minute script) fell back to static text cards,
    which QA correctly flagged as frozen (`QA_FAILED` warning).
  - *Fixes:* one job at a time (others queue), 2 scenes rendered in parallel, and a failed encode is retried
    after a pause. The failed cases were re-run as `outputs_run4b`. Run 4 is kept unchanged as evidence.

  The interrupted runs are not in the test pack. The earlier runs 1–3 (older code) are kept with their
  failures.
- **Free-tier limits drive reliability.** Gemini TTS allows about 10 requests per day per model; in run 1 the
  quota ran out and 7 hard-case jobs failed with `TTS_FAILED` (kept in `testpack/outputs_run1`). Fixed by a
  model chain, a narration cache and the local Kokoro and eSpeak fallbacks. On 20 Oct Gemini answered in
  about 20 s even for tiny prompts, so the planner timeout was raised from 10 s to 30 s. When Gemini is down,
  a keyword planner is used and relevance drops (reported as `PLANNER_FALLBACK`).
- **Earlier bugs found by the test pack** (fixed, failures kept as evidence):
  - OpenCV 5 removed `CascadeClassifier`.
  - The OpenCV face detector is not thread-safe (2 jobs failed in run 2).
  - A TTS model read the style prompt aloud.
  - Gemini TTS appended C2PA metadata that played as a noise burst at the end.
  - An empty caption file broke a render.
- **AI images** (FLUX via Cloudflare Workers AI) are implemented and tested with a mock. They are not enabled
  on the demo (no Cloudflare account ID), so produced videos use stock footage, stills with depth parallax
  and cards. The "two footage sources" claim therefore rests on several stock libraries plus generated stat
  and text cards, chosen by the rule in section 2.
- **Speed on a laptop:** turnaround is a few minutes per minute of video (see Numbers) on a 4-core Ryzen 3
  with 7 GB RAM. The GPU is used only to encode H.264 (AMD AMF).
- **Live demo** runs on the author's laptop behind a free Cloudflare quick tunnel; Hugging Face Docker
  Spaces need a paid plan, and serverless hosts such as Vercel cannot run ffmpeg/Whisper/CLIP jobs of
  several minutes. The demo is only up while the laptop is on, and the quick-tunnel URL changes if the
  tunnel restarts.
- **Relevance:**
  - The automatic numbers are CLIP scores and the Gemini judge.
  - Human 1–5 ratings come from the author (`testpack/human_ratings.csv`), not from independent reviewers.
  - Hindi and Tamil quality has not been rated by native speakers.
  - Hindi and Tamil footage search runs on the translated English meaning.

## 5. Numbers

Per-case values are in `testpack/METRICS.md`; the summary is in the Metrics tab of the submission form.

- **Cost:** **$0.00 per minute**. All hosted services are on free tiers, and open models run on the laptop.
- **Hardware:** Windows 11 laptop, AMD Ryzen 3 7320U (4 cores / 8 threads), 7.2 GB RAM, integrated Radeon
  GPU used for H.264 encoding only.
- **Turnaround** (`outputs_turnaround`, one job at a time): the 60-second script came out as 81 s of
  narration. Cold took 193 s and warm 152 s, about 2.4 and 1.9 minutes per minute of video.
- **Relevance:** the independent VLM judge (Gemini) gave a mean of 3.96 / 5 on 107 judged scenes (runs 4, 4b and turnaround), with
  82% rated 4 or 5 and 92% of narration time on clips it rated 3+. Mean CLIP score was 0.261. The repeat-clip
  rate is 0.
- **Concurrency:** the demo runs **1 job at a time** (`WORKER_THREADS=1`) and queues the rest. Two at once
  works on a larger machine but ran out of memory on this 7 GB laptop (section 4).

## 6. What I would do next

1. A small GPU box: faster-whisper and CLIP on GPU, and an open image-to-video model (e.g. Wan 2.1,
   Apache-2.0) for short generated shots.
2. Score 3 frames per clip from a downloaded proxy instead of 1 thumbnail, and add a temporal-consistency
   term so consecutive scenes match in palette.
3. Paid TTS/LLM tiers behind the same adapters for predictable quotas, and a Redis queue with separate worker
   containers to scale jobs horizontally.
4. A human-rating study with independent raters to calibrate τ and the VLM judge.
