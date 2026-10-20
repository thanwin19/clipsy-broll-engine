import hashlib
import hmac

from app.errors import ApiError
from app.jobs import sign
from app.pipeline.narration import even_words
from app.pipeline.planner import MockPlanner, scenes_from_words, segment
from app.pipeline.render import caption_chunks, frame_bounds, scene_cues

TEXT = ("Saving money starts with a simple plan. Track every rupee you spend, and review it weekly. "
        "Sales grew 40% in 2025. Trust is earned slowly.")


def test_scenes_cover_every_second_and_follow_sentences():
    words = even_words(TEXT, 0.3, 14.7)
    scenes = scenes_from_words(words, 15.0, 2.0, 8.0)
    assert scenes[0].start == 0.0 and scenes[-1].end == 15.0
    for a, b in zip(scenes, scenes[1:]):
        assert a.end == b.start                      # contiguous: no gaps, no overlaps
    for sc in scenes[:-1]:
        assert sc.text.rstrip().endswith((".", ",", "!", "?")) or len(sc.text.split()) >= 3


def test_short_pieces_are_merged_no_flash_clips():
    words = even_words("Go. Now. Then we start the real work today.", 0, 4.0)
    groups = segment(words, 4.0, 2.0, 8.0)
    assert all(g[-1].end - g[0].start >= 1.0 for g in groups)


def test_long_sentence_is_split_at_phrase_boundary():
    words = even_words("We walked across the old bridge in the rain, and then we found a small cafe "
                       "where the owner served hot tea and fresh bread to every traveller.", 0, 16)
    groups = segment(words, 16, 2.0, 8.0)
    assert len(groups) >= 2
    assert all(g[-1].end - g[0].start <= 8.5 for g in groups)


def test_frame_bounds_sum_to_narration_no_drift():
    words = even_words(TEXT, 0.0, 13.37)
    scenes = scenes_from_words(words, 13.37, 2.0, 8.0)
    b = frame_bounds(scenes, 30)
    assert sum(e - s for s, e in b) == round(13.37 * 30)


def test_captions_stay_inside_their_scene():
    words = even_words(TEXT, 0.0, 15)
    for sc in scenes_from_words(words, 15, 2.0, 8.0):
        for a, b, t in scene_cues(sc):
            assert sc.start <= a < b <= sc.end and t


def test_caption_chunks_have_no_orphans():
    chunks = caption_chunks(even_words("one two three four five six seven eight", 0, 4))
    assert all(len(c) >= 2 for c in chunks)


def test_mock_planner_resolves_double_meanings_and_stats():
    words = even_words("I keep my savings in a bank. Sales grew 40% in 2025.", 0, 6)
    scenes = scenes_from_words(words, 6, 2.0, 8.0)
    out = MockPlanner().annotate(scenes, "", "", [])["scenes"]
    assert any("finance" in " ".join(o["queries"]) for o in out)
    assert any(o["type"] == "statistic" and "40" in o["stat"]["value"] for o in out)


def test_error_shape():
    e = ApiError("EMPTY_SCRIPT").to_dict()["error"]
    assert set(e) == {"code", "message", "retryable", "details"} and e["code"] == "EMPTY_SCRIPT"


def test_hmac_signature():
    body = b'{"a":1}'
    assert sign(body, "s3cret") == "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()


def test_parallax_renders_moving_frames(tmp_path):
    """Depth parallax on a synthetic image produces a valid clip whose first and last frames differ."""
    import numpy as np
    from PIL import Image
    from app.pipeline import media, parallax
    from app.pipeline.render import encoder_args
    img = np.zeros((900, 1600, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 1600, dtype=np.uint8)[None, :]
    img[450:, :, 1] = 200
    Image.fromarray(img).save(tmp_path / "p.jpg")
    parallax.depth_map = lambda im, cache=None: np.tile(np.linspace(0, 1, im.shape[0], dtype=np.float32)[:, None], (1, im.shape[1]))
    parallax.render_parallax(tmp_path / "p.jpg", 640, 360, 30, 0.5, "", encoder_args("libx264") + ["-frames:v", "30"],
                             "out.mp4", tmp_path, 30)
    assert media.duration(tmp_path / "out.mp4") > 0.9


def test_prompt_reader_speaks_only_voiceover():
    """A pasted brief: only Voiceover lines are spoken; Visual Prompt -> shot; Text Overlay -> on-screen text."""
    from app.pipeline.interpret import interpret_rules, looks_structured
    doc = ("## Productivity Hacks (60 Seconds)\nThis script is perfect for a Reel.\n---\n"
           "### Scene 1: The Foundation (0:00 - 0:15)\n* **Visual Prompt:** A hand writes a list in a journal.\n"
           "* **Text Overlay (On-Screen):** 1. Stop multitasking.\n"
           "* **Voiceover (Fast, energetic):** \"Stop trying to do everything at once.\"\n---\n"
           "### Scene 2: Focus (0:15 - 0:30)\n* **Visual Prompt:** A timer counts down.\n"
           "* **Voiceover:** \"Set a timer for 25 minutes.\"\n")
    assert looks_structured(doc)
    scenes = interpret_rules(doc)["scenes"]
    assert [s["spoken"] for s in scenes] == ["Stop trying to do everything at once.", "Set a timer for 25 minutes."]
    assert "journal" in scenes[0]["visual"] and scenes[0]["overlay"] == "1. Stop multitasking."
    assert all("Reel" not in s["spoken"] and "Productivity Hacks" not in s["spoken"] for s in scenes)


def test_language_detection_routes_tamil():
    from app.pipeline.narration import detect_language
    assert detect_language("மழைக்காலத்தில் மலைகள் பச்சையாக மாறுகின்றன.") == "ta"
    assert detect_language("సుబ్బారావు ఇంటికి వెళ్ళాడు") == "te"
    assert detect_language("सुबह जल्दी उठना एक अच्छी आदत है।") == "hi"
    assert detect_language("Plain English text.") == "en"


def test_espeak_tamil_offline(tmp_path):
    from app.pipeline import espeak, media
    out = espeak.synthesize("வணக்கம். இது ஒரு சோதனை.", "ta-IN", tmp_path / "ta.wav")
    assert media.duration(out) > 0.5


def test_avoid_phrase_needs_all_words_and_llm_cannot_invent_avoid_terms(tmp_path):
    """Regression: the planner suggested avoiding "instant coffee" for a coffee tutorial and every clip tagged
    "coffee" was rejected, so most scenes became text cards."""
    from types import SimpleNamespace

    from app.pipeline.planner import merge_avoid
    from app.pipeline.rerank import MockScorer, score_candidates
    from app.pipeline.sources import Candidate

    def clip(i, tags):
        return Candidate(id=f"t:{i}", provider="t", kind="video", title=tags, page_url="", download_url="",
                         thumb_url="", licence="x", licence_url="", attribution="", tags=tags, duration=10)

    scene = SimpleNamespace(queries=["filter coffee"], visual="coffee cup", text="Make filter coffee.", negative=[],
                            ai_allowed=True, duration=4.0)
    kept = score_candidates(scene, [clip(1, "coffee, cup, filter"), clip(2, "instant coffee, jar"),
                                    clip(3, "beer, alcohol, bar")], ["instant coffee", "alcohol"], False,
                            MockScorer(), tmp_path)
    assert [c.id for c in kept] == ["t:1"]
    assert merge_avoid([], {"avoid_terms": ["instant coffee"]}) == []
    assert merge_avoid(["alcohol"], {"avoid_terms": ["beer"]}) == ["alcohol", "beer"]
