# Measured metrics (from recorded outputs)

- Attempts: 38, succeeded: 21 (expected rejections counted as not succeeded)
- Videos produced: 21, scenes: 99, repeat-clip rate: 0.0
- QA pass rate (black/frozen frames, drift, codec, fps): 1.0
- Mean CLIP relevance of stock/AI scenes: 0.2719
- Median processing seconds per minute of narration: 443.8

| case | run | narration s | scenes | total s | s per min | QA | sources | cards | warnings / error |
|---|---|---|---|---|---|---|---|---|---|
| 01_tutorial_en_in | outputs_run1 | 21.131 | 4 | 156.291 | 443.8 | pass | card-renderer, coverr, pixabay | 2 | 1 |
| 02_news_summary_en_us | outputs_run1 | 21.571 | 4 | 153.196 | 426.1 | pass | card-renderer, coverr, pixabay | 1 | 2 |
| 03_product_promo_en_gb | outputs_run1 | 17.16 | 4 | 108.714 | 380.1 | pass | coverr, pixabay | 0 | 1 |
| 04_story_en_in | outputs_run1 | 21.72 | 5 | 213.874 | 590.8 | pass | card-renderer, pixabay | 1 | 2 |
| 05_finance_money_value | outputs_run1 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| 05_finance_money_value | outputs_run2 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| 05_finance_money_value | outputs_run3 | 15.408 | 4 | 95.242 | 370.9 | pass | card-renderer, pixabay | 1 | 2 |
| 06_scene_list_city | outputs_run1 | 11.56 | 4 | 154.986 | 804.4 | pass | card-renderer, pixabay | 1 | 2 |
| 07_space_science | outputs_run1 | 17.44 | 3 | 154.519 | 531.6 | pass | nasa, pixabay | 0 | 1 |
| 08_health_short_20s | outputs_run1 | 8.24 | 2 | 70.938 | 516.5 | pass | pixabay | 0 | 1 |
| 09_hindi_tutorial | outputs_run1 | 10.6 | 3 | 69.442 | 393.1 | pass | card-renderer, pixabay | 1 | 1 |
| 10_long_5min_agriculture | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| 10_long_5min_agriculture | outputs_run2 | 336.168 | 38 | 1557.551 | 278.0 | pass | card-renderer, pixabay, wikimedia | 11 | 10 |
| audio_01_us_tutorial | outputs_run1 | 15.131 | 3 | 59.578 | 236.2 | pass | card-renderer, pixabay | 1 | 1 |
| audio_02_gb_travel | outputs_run1 | 14.931 | 3 | 69.657 | 279.9 | pass | card-renderer, pixabay | 1 | 1 |
| audio_03_in_startup | outputs_run1 | 11.811 | 2 | 64.165 | 326.0 | pass | pixabay | 0 | 2 |
| audio_04_tamil_nature | outputs_run1 | 10.371 | 3 | 196.235 | 1135.3 | pass | card-renderer, pixabay, wikimedia | 1 | 5 |
| audio_05_in_startup | outputs_run1 | 11.811 | 2 | 85.137 | 432.5 | pass | pixabay | 0 | 2 |
| hc_916_edge_subject | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_916_edge_subject | outputs_run2 | 7.688 | 2 | 145.389 | 1134.7 | pass | coverr, pixabay | 0 | 1 |
| hc_abstract_and_stats | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_abstract_and_stats | outputs_run2 | 10.768 | 3 | 127.528 | 710.6 | pass | card-renderer, pixabay | 2 | 1 |
| hc_audio_url_404 | outputs_run1 | – | – | – | – | – | – | – | failed: INPUT_DOWNLOAD_FAILED |
| hc_avoid_faces_alcohol | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_avoid_faces_alcohol | outputs_run2 | 6.808 | 1 | 75.301 | 663.6 | pass | pixabay | 0 | 1 |
| hc_bad_ratio | outputs_run1 | – | – | – | – | – | – | – | rejected: INVALID_ASPECT_RATIO |
| hc_corrupted_audio | outputs_run1 | – | – | – | – | – | – | – | rejected: AUDIO_UNREADABLE |
| hc_double_meanings | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_double_meanings | outputs_run2 | 18.288 | 5 | 139.825 | 458.7 | pass | card-renderer, pixabay | 2 | 2 |
| hc_empty_script | outputs_run1 | – | – | – | – | – | – | – | rejected: EMPTY_SCRIPT |
| hc_music_only | outputs_run1 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_named_person_brand | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_named_person_brand | outputs_run2 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| hc_named_person_brand | outputs_run3 | 6.768 | 2 | 66.324 | 588.0 | pass | pixabay | 0 | 3 |
| hc_sad_news | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_sad_news | outputs_run2 | 18.328 | 2 | 71.307 | 233.4 | pass | pixabay | 0 | 1 |
| hc_silent_audio | outputs_run1 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_wrong_file_type_pdf | outputs_run1 | – | – | – | – | – | – | – | rejected: UNSUPPORTED_FILE_TYPE |
