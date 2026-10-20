# Test pack index

Every claim in `REPORT.md` and in the submission form points to one of the files below. Rendered videos
(`*.mp4`), thumbnails and waveforms are not in git (too large); they are in the GitHub release
**video-pack-v1**, one zip per run, with the same folder layout.

## Runs

| Folder | Code | What it is |
|---|---|---|
| `outputs_run4/` | current (20 Oct) | **Main evidence.** All 31 inputs on the final code: 10 core inputs, 8 voiceovers (mp3, m4a, wav, Tamil, Hindi, 5-minute upload) and 13 hard cases, 2 jobs at a time |
| `outputs_run4b/` | current | Re-run, one job at a time, of the run-4 cases hit by out-of-memory errors (Hindi, 5-minute script and voiceover, named person, sad news); see REPORT.md section 4 |
| `outputs_turnaround/` | current | The same 60-second script run twice, one job at a time: first = cold, second = warm (search/TTS caches) |
| `outputs_run1/` | 18–19 Oct | First full run. 7 hard cases and the 5-minute script failed with `TTS_FAILED` (Gemini TTS free quota) and `05_finance` with `INTERNAL_ERROR`; kept as evidence |
| `outputs_run2/` | 19 Oct | Re-run of the run 1 failures after the TTS fallback chain; the 5-minute script; 2 OpenCV thread-safety failures kept |
| `outputs_run3/` | 19 Oct | Re-run of the 2 cases that failed in run 2, after the fix |
| `E2E_FINAL_RESULTS.txt` | current | Live end-to-end check against the running service with S3 storage: 29/29 checks |
| `METRICS.md`, `metrics.json` | — | Numbers computed from the recorded files by `make/metrics.py` (no estimates) |
| `human_ratings.csv` | — | One row per scene of run 4 with the VLM judge score and the author's 1–5 rating |

Each case folder has `request.json` (what was sent), `job.json` (final job status, progress, errors,
links), `timeline.json`, `manifest.json`, `captions.srt`, `video_*.mp4` (release), or `response.json` for
requests rejected at submit time. `batch_log.txt`, `summary.csv` and `summary.json` list every case of the
run.

Inputs are in `inputs/` (the 60-second turnaround script is in `inputs_turnaround/`). All voices are
synthetic: Gemini TTS for `audio_01`–`audio_04` and local Kokoro-82M for `audio_07`–`audio_08`, made by
`make/make_audio.py`. `audio_05` is `audio_03` converted to m4a, and `audio_06` is `audio_02` converted to
wav, to cover the formats.

## Claim → evidence

| Claim | Evidence |
|---|---|
| Audio input (mp3, m4a, wav, 5 minutes) | `outputs_run4/audio_01_us_tutorial` … `audio_08_long_5min_agriculture` |
| Text input (we generate the narration) | `outputs_run4/01_*` … `10_*` (manifest `providers.tts`) |
| Script with `[shot]` notes, IncuBrix scene list | `tests/test_api.py::test_script_with_inline_shot_descriptions`, `outputs_run4/06_scene_list_city` |
| Scenes on sentence/phrase boundaries, every second covered | `timeline.json` start/end in any run 4 case; `tests/test_unit.py::test_scenes_cover_every_second_and_follow_sentences`, `test_long_sentence_is_split_at_phrase_boundary`, `test_frame_bounds_sum_to_narration_no_drift` |
| Licence, source, attribution, link, `ai_generated` per clip | `timeline.json` → `scenes[].clip` in every case; `manifest.json` → `rights` |
| All ratios, normalised, no black/frozen frames or drift | `manifest.json` → `qa` (per ratio: codec, fps, size, pixel format, black, frozen, drift) |
| No repeated clip | `METRICS.md` repeat-clip rate; `E2E_FINAL_RESULTS.txt` |
| Meaning, not keyword (double meanings) | `outputs_run4/hc_double_meanings`, `05_finance_money_value` (reasons and scores in `timeline.json`) |
| Statistics → stat card, abstract lines | `outputs_run4/hc_abstract_and_stats` |
| Named person/brand: no AI, neutral footage | `outputs_run4/hc_named_person_brand` (reason says "safety: named person/brand …") |
| Sad news: calm, respectful footage, sombre music | `outputs_run4/hc_sad_news` (manifest `mood`) |
| 9:16 subject at the edge | `outputs_run4/hc_916_edge_subject` (`reframe_subject_x` in `timeline.json`) |
| Avoid faces/alcohol | `outputs_run4/hc_avoid_faces_alcohol` (rejected candidates in `alternatives`) |
| Silent, music-only, corrupt, wrong type, empty, bad ratio, broken URL | `outputs_run4/hc_*` (`job.json` or `response.json` error codes) |
| Provider down mid-job, job still finishes | `E2E_FINAL_RESULTS.txt`; `tests/test_api.py::test_e2e_provider_down_still_finishes`, `test_switch_provider_off_and_on_at_runtime`; real Coverr 403 outage in run 4 warnings |
| Two users at once | run 4 used 2 concurrent jobs throughout; `E2E_FINAL_RESULTS.txt` (voiceover upload concurrent with another job) |
| Swap a clip, faster than a full render | `E2E_FINAL_RESULTS.txt` (16 s vs 83 s); `tests/test_api.py::test_e2e_scene_list_and_swap` |
| Hindi and Tamil | `outputs_run4/09_hindi_tutorial`, `audio_07_hindi_tutorial`, `audio_04_tamil_nature`; `tests/test_unit.py::test_espeak_tamil_offline` |
| Independent VLM judge | `timeline.json` → `scenes[].judge`, `manifest.json` → `independent_judge`; agreement with human ratings in `METRICS.md` |
| Turnaround, cost, memory | `METRICS.md`, `outputs_turnaround/`, `metrics.json` |
| docker compose and tests on a clean machine | `.github/workflows/compose.yml` and its runs in the repository's Actions tab |
| Bugs found and fixed | `REPORT.md` section 4; failures kept in runs 1–2 |
