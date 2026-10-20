# Measured metrics (from recorded outputs)

- Attempts: 76, succeeded: 50 (expected rejections counted as not succeeded)
- Videos produced: 50, scenes: 374, repeat-clip rate: 0.0
- QA pass rate (black/frozen frames, drift, codec, fps): 0.96
- Mean CLIP relevance of stock/AI scenes: 0.2612
- Median processing seconds per minute of narration: 432.5
- Independent VLM judge (Gemini, 1-5): 107 scenes, mean 3.96, share rated 4-5: 0.822
- Human ratings: not entered yet (run with --template, fill testpack/human_ratings.csv)

| case | run | narration s | scenes | total s | s per min | QA | sources | cards | warnings / error |
|---|---|---|---|---|---|---|---|---|---|
| 01_tutorial_en_in | outputs_run1 | 21.131 | 4 | 156.291 | 443.8 | pass | card-renderer, coverr, pixabay | 2 | 1 |
| 01_tutorial_en_in | outputs_run4 | 21.051 | 5 | 82.525 | 235.2 | pass | pixabay | 0 | 2 |
| 02_news_summary_en_us | outputs_run1 | 21.571 | 4 | 153.196 | 426.1 | pass | card-renderer, coverr, pixabay | 1 | 2 |
| 02_news_summary_en_us | outputs_run4 | 21.011 | 4 | 139.6 | 398.6 | pass | card-renderer, pixabay | 1 | 3 |
| 03_product_promo_en_gb | outputs_run1 | 17.16 | 4 | 108.714 | 380.1 | pass | coverr, pixabay | 0 | 1 |
| 03_product_promo_en_gb | outputs_run4 | 14.771 | 4 | 130.047 | 528.3 | pass | pixabay | 0 | 4 |
| 04_story_en_in | outputs_run1 | 21.72 | 5 | 213.874 | 590.8 | pass | card-renderer, pixabay | 1 | 2 |
| 04_story_en_in | outputs_run4 | 21.251 | 5 | 146.579 | 413.9 | pass | pixabay | 0 | 2 |
| 05_finance_money_value | outputs_run1 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| 05_finance_money_value | outputs_run2 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| 05_finance_money_value | outputs_run3 | 15.408 | 4 | 95.242 | 370.9 | pass | card-renderer, pixabay | 1 | 2 |
| 05_finance_money_value | outputs_run4 | 15.281 | 4 | 135.126 | 530.6 | pass | card-renderer, pixabay | 2 | 3 |
| 06_scene_list_city | outputs_run1 | 11.56 | 4 | 154.986 | 804.4 | pass | card-renderer, pixabay | 1 | 2 |
| 06_scene_list_city | outputs_run4 | 11.29 | 4 | 140.648 | 747.5 | pass | pixabay | 0 | 2 |
| 07_space_science | outputs_run1 | 17.44 | 3 | 154.519 | 531.6 | pass | nasa, pixabay | 0 | 1 |
| 07_space_science | outputs_run4 | 14.771 | 3 | 79.326 | 322.2 | pass | card-renderer | 3 | 5 |
| 08_health_short_20s | outputs_run1 | 8.24 | 2 | 70.938 | 516.5 | pass | pixabay | 0 | 1 |
| 08_health_short_20s | outputs_run4 | 5.971 | 1 | 98.966 | 994.5 | pass | card-renderer | 1 | 4 |
| 09_hindi_tutorial | outputs_run1 | 10.6 | 3 | 69.442 | 393.1 | pass | card-renderer, pixabay | 1 | 1 |
| 09_hindi_tutorial | outputs_run4 | 9.92 | 3 | 87.077 | 526.7 | FAIL | card-renderer | 3 | 8 |
| 09_hindi_tutorial | outputs_run4b | 9.92 | 3 | 95.942 | 580.3 | pass | pixabay | 0 | 1 |
| 10_long_5min_agriculture | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| 10_long_5min_agriculture | outputs_run2 | 336.168 | 38 | 1557.551 | 278.0 | pass | card-renderer, pixabay, wikimedia | 11 | 10 |
| 10_long_5min_agriculture | outputs_run4 | 335.672 | 38 | 434.319 | 77.6 | FAIL | card-renderer, wikimedia | 34 | 7 |
| 10_long_5min_agriculture | outputs_run4b | 335.672 | 39 | 577.22 | 103.2 | pass | card-renderer, pixabay, wikimedia | 5 | 12 |
| audio_01_us_tutorial | outputs_run1 | 15.131 | 3 | 59.578 | 236.2 | pass | card-renderer, pixabay | 1 | 1 |
| audio_01_us_tutorial | outputs_run4 | 15.131 | 3 | 69.552 | 275.8 | pass | card-renderer | 3 | 4 |
| audio_02_gb_travel | outputs_run1 | 14.931 | 3 | 69.657 | 279.9 | pass | card-renderer, pixabay | 1 | 1 |
| audio_02_gb_travel | outputs_run4 | 14.931 | 3 | 79.791 | 320.6 | pass | card-renderer, wikimedia | 2 | 3 |
| audio_03_in_startup | outputs_run1 | 11.811 | 2 | 64.165 | 326.0 | pass | pixabay | 0 | 2 |
| audio_03_in_startup | outputs_run4 | 11.811 | 2 | 94.183 | 478.5 | pass | card-renderer | 2 | 4 |
| audio_04_tamil_nature | outputs_run1 | 10.371 | 3 | 196.235 | 1135.3 | pass | card-renderer, pixabay, wikimedia | 1 | 5 |
| audio_04_tamil_nature | outputs_run4 | 10.371 | 2 | 77.019 | 445.6 | pass | wikimedia | 0 | 4 |
| audio_05_in_startup | outputs_run1 | 11.811 | 2 | 85.137 | 432.5 | pass | pixabay | 0 | 2 |
| audio_05_in_startup | outputs_run4 | 11.811 | 2 | 53.298 | 270.8 | pass | card-renderer | 2 | 4 |
| hc_916_edge_subject | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_916_edge_subject | outputs_run2 | 7.688 | 2 | 145.389 | 1134.7 | pass | coverr, pixabay | 0 | 1 |
| hc_916_edge_subject | outputs_run4 | 7.561 | 2 | 58.45 | 463.8 | pass | pixabay | 0 | 2 |
| hc_abstract_and_stats | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_abstract_and_stats | outputs_run2 | 10.768 | 3 | 127.528 | 710.6 | pass | card-renderer, pixabay | 2 | 1 |
| hc_abstract_and_stats | outputs_run4 | 10.641 | 3 | 129.144 | 728.2 | pass | card-renderer, pixabay | 1 | 3 |
| hc_audio_url_404 | outputs_run1 | – | – | – | – | – | – | – | failed: INPUT_DOWNLOAD_FAILED |
| hc_audio_url_404 | outputs_run4 | – | – | – | – | – | – | – | failed: INPUT_DOWNLOAD_FAILED |
| hc_avoid_faces_alcohol | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_avoid_faces_alcohol | outputs_run2 | 6.808 | 1 | 75.301 | 663.6 | pass | pixabay | 0 | 1 |
| hc_avoid_faces_alcohol | outputs_run4 | 6.681 | 1 | 54.221 | 486.9 | pass | pixabay | 0 | 1 |
| hc_bad_ratio | outputs_run1 | – | – | – | – | – | – | – | rejected: INVALID_ASPECT_RATIO |
| hc_bad_ratio | outputs_run4 | – | – | – | – | – | – | – | rejected: INVALID_ASPECT_RATIO |
| hc_corrupted_audio | outputs_run1 | – | – | – | – | – | – | – | rejected: AUDIO_UNREADABLE |
| hc_corrupted_audio | outputs_run4 | – | – | – | – | – | – | – | rejected: AUDIO_UNREADABLE |
| hc_double_meanings | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_double_meanings | outputs_run2 | 18.288 | 5 | 139.825 | 458.7 | pass | card-renderer, pixabay | 2 | 2 |
| hc_double_meanings | outputs_run4 | 18.161 | 5 | 90.275 | 298.2 | pass | pixabay | 0 | 3 |
| hc_empty_script | outputs_run1 | – | – | – | – | – | – | – | rejected: EMPTY_SCRIPT |
| hc_empty_script | outputs_run4 | – | – | – | – | – | – | – | rejected: EMPTY_SCRIPT |
| hc_music_only | outputs_run1 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_music_only | outputs_run4 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_named_person_brand | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_named_person_brand | outputs_run2 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| hc_named_person_brand | outputs_run3 | 6.768 | 2 | 66.324 | 588.0 | pass | pixabay | 0 | 3 |
| hc_named_person_brand | outputs_run4 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| hc_named_person_brand | outputs_run4b | 5.873 | 2 | 49.203 | 502.7 | pass | pixabay | 0 | 3 |
| hc_sad_news | outputs_run1 | – | – | – | – | – | – | – | failed: TTS_FAILED |
| hc_sad_news | outputs_run2 | 18.328 | 2 | 71.307 | 233.4 | pass | pixabay | 0 | 1 |
| hc_sad_news | outputs_run4 | – | – | – | – | – | – | – | failed: INTERNAL_ERROR |
| hc_sad_news | outputs_run4b | 18.201 | 2 | 64.352 | 212.1 | pass | pixabay | 0 | 3 |
| hc_silent_audio | outputs_run1 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_silent_audio | outputs_run4 | – | – | – | – | – | – | – | failed: NO_SPEECH_DETECTED |
| hc_wrong_file_type_pdf | outputs_run1 | – | – | – | – | – | – | – | rejected: UNSUPPORTED_FILE_TYPE |
| hc_wrong_file_type_pdf | outputs_run4 | – | – | – | – | – | – | – | rejected: UNSUPPORTED_FILE_TYPE |
| audio_06_gb_travel | outputs_run4 | 14.931 | 3 | 47.875 | 192.4 | pass | card-renderer, wikimedia | 2 | 3 |
| audio_07_hindi_tutorial | outputs_run4 | 10.418 | 2 | 78.229 | 450.5 | pass | pixabay | 0 | 4 |
| audio_08_long_5min_agriculture | outputs_run4 | 298.73 | 52 | 643.391 | 129.2 | pass | card-renderer, pixabay, wikimedia | 11 | 18 |
| audio_08_long_5min_agriculture | outputs_run4b | 298.73 | 52 | 536.269 | 107.7 | pass | card-renderer, coverr, pixabay, wikimedia | 15 | 17 |
| 60s_saving_money | outputs_turnaround | 80.881 | 13 | 188.35 | 139.7 | pass | card-renderer, pixabay | 2 | 3 |
| 60s_saving_money_warm | outputs_turnaround | 80.881 | 13 | 143.875 | 106.7 | pass | card-renderer, pixabay | 3 | 4 |
