"""Integration (API contract) and end-to-end tests through the HTTP API with mock providers."""
import json
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.pipeline import media


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait(client, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        assert 0 <= j["progress"] <= 100
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job timed out")


def read(link: str) -> bytes:
    from app.storage import get_storage
    st = get_storage()
    if st.name == "local":
        return (get_settings().data_dir / "outputs" / link.split("/files/", 1)[1]).read_bytes()
    import httpx
    return httpx.get(link).content


# ------------------------------------------------------------ integration
def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["ffmpeg"]


def test_correlation_id_echoed(client):
    r = client.get("/health", headers={"X-Correlation-Id": "abc-123"})
    assert r.headers["X-Correlation-Id"] == "abc-123"


def test_empty_script_error(client):
    r = client.post("/api/v1/broll/jobs", json={"script": "   "})
    assert r.status_code == 422 and r.json()["error"]["code"] == "EMPTY_SCRIPT"


def test_no_input_error(client):
    r = client.post("/api/v1/broll/jobs", json={})
    assert r.json()["error"]["code"] == "NO_INPUT"


def test_bad_aspect_ratio(client):
    r = client.post("/api/v1/broll/jobs", json={"script": "hello there world", "aspect_ratios": ["4:3"]})
    assert r.json()["error"]["code"] == "INVALID_ASPECT_RATIO"


def test_wrong_file_type_rejected(client):
    r = client.post("/api/v1/broll/jobs", files={"file": ("notes.txt", b"not audio", "text/plain")},
                    data={"request": "{}"})
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_unknown_job_404_json(client):
    r = client.get("/api/v1/jobs/job_nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_idempotency_same_key_same_job_and_conflict(client):
    body = {"script": "Idempotency check sentence one. And sentence two here.", "aspect_ratios": ["1:1"]}
    h = {"Idempotency-Key": "idem-test-1"}
    a = client.post("/api/v1/broll/jobs", json=body, headers=h)
    b = client.post("/api/v1/broll/jobs", json=body, headers=h)
    assert a.status_code == b.status_code == 202
    assert a.json()["job_id"] == b.json()["job_id"] and b.headers["Idempotent-Replay"] == "true"
    c = client.post("/api/v1/broll/jobs", json={**body, "script": "different"}, headers=h)
    assert c.status_code == 409 and c.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    wait(client, a.json()["job_id"])


def test_openapi_file_matches_running_service(client):
    import yaml
    spec = yaml.safe_load(Path("openapi.yaml").read_text(encoding="utf-8"))
    live = client.get("/openapi.json").json()
    assert spec["openapi"].startswith("3.1")
    assert spec["paths"] == live["paths"]
    assert all(p.startswith("/api/v1/") or p in ("/health", "/ready") for p in live["paths"])


# ------------------------------------------------------------ end-to-end
def test_e2e_script_to_video(client):
    body = {"script": "Saving money starts with a simple plan. Track every rupee you spend each week. "
                      "Sales grew 40% in 2025. Small habits compound over many years.",
            "aspect_ratios": ["16:9", "9:16"], "avoid": ["alcohol"], "captions": {"burn_in": True, "srt": True}}
    r = client.post("/api/v1/broll/jobs", json=body, headers={"X-Correlation-Id": "e2e-1"})
    assert r.status_code == 202
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "succeeded", j["error"]
    links = j["result"]["links"]
    assert {"video_16x9", "video_9x16", "timeline", "manifest", "captions_srt"} <= set(links)
    tl = json.loads(read(links["timeline"]))
    scenes = tl["scenes"]
    assert scenes[0]["start"] == 0 and abs(scenes[-1]["end"] - tl["narration_seconds"]) < 0.01
    ids = [s["clip"]["id"] for s in scenes]
    assert len(ids) == len(set(ids))                             # no repeats
    for s in scenes:
        c = s["clip"]
        assert c["licence"] and c["source"] and s["reason"] and c["commercial_use"] is True
        assert "ai_generated" in c
    assert any(s["clip"]["kind"] == "card" for s in scenes)       # the 40% line becomes a stat card
    m = json.loads(read(links["manifest"]))
    for k in ("providers", "settings", "seed", "timings_seconds", "cost_estimate_usd", "warnings", "rights", "qa"):
        assert k in m
    for ratio, rep in m["qa"].items():
        assert rep["passed"], rep
        assert rep["fps"] == 30 and rep["pix_fmt"] == "yuv420p" and rep["av_drift_seconds"] <= 0.1
    assert m["qa"]["9:16"]["width"] == 1080 and m["qa"]["9:16"]["height"] == 1920
    assert b"-->" in read(links["captions_srt"])


def test_e2e_scene_list_and_swap(client):
    body = {"scenes": [{"text": "The city wakes up before sunrise today.", "visual_hint": "city skyline dawn"},
                       {"text": "Thousands of commuters head to work.", "visual_hint": "train station crowd"},
                       {"text": "By night the streets glow again.", "visual_hint": "night traffic"}],
            "aspect_ratios": ["16:9"]}
    j = wait(client, client.post("/api/v1/broll/jobs", json=body).json()["job_id"])
    assert j["status"] == "succeeded", j["error"]
    tl = json.loads(read(j["result"]["links"]["timeline"]))
    assert len(tl["scenes"]) == 3 and tl["scenes"][0]["visual_hint"] == "city skyline dawn"
    full = j["result"]["timings_seconds"]["total"]
    old = tl["scenes"][1]["clip"]["id"]
    s = client.post(f"/api/v1/broll/jobs/{j['job_id']}/scenes/1/swap", json={})
    assert s.status_code == 202
    sj = wait(client, s.json()["job_id"])
    assert sj["status"] == "succeeded", sj["error"]
    tl2 = json.loads(read(sj["result"]["links"]["timeline"]))
    assert tl2["scenes"][1]["clip"]["id"] != old
    assert tl2["scenes"][0]["clip"]["id"] == tl["scenes"][0]["clip"]["id"]
    assert sj["result"]["timings_seconds"]["total"] < full


def test_e2e_audio_upload_and_silent_audio(client, tmp_path):
    voice = tmp_path / "voice.wav"
    subprocess.run([media.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=300:d=6", voice], check=True)
    r = client.post("/api/v1/broll/jobs", files={"file": ("voice.wav", voice.read_bytes(), "audio/wav")},
                    data={"request": json.dumps({"aspect_ratios": ["1:1"]})})
    assert r.status_code == 202, r.text
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "succeeded", j["error"]

    silent = tmp_path / "silent.wav"
    subprocess.run([media.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=d=5", silent], check=True)
    r = client.post("/api/v1/broll/jobs", files={"file": ("silent.wav", silent.read_bytes(), "audio/wav")},
                    data={"request": "{}"})
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "failed" and j["error"]["code"] == "NO_SPEECH_DETECTED"


def test_e2e_provider_down_still_finishes(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "disabled_providers", "mock")
    j = wait(client, client.post("/api/v1/broll/jobs", json={"script": "Rivers carve canyons over time. Patience matters.",
                                                              "aspect_ratios": ["16:9"]}).json()["job_id"])
    assert j["status"] == "succeeded", j["error"]
    tl = json.loads(read(j["result"]["links"]["timeline"]))
    assert all(s["clip"]["kind"] == "card" for s in tl["scenes"])


def test_script_with_inline_shot_descriptions(client):
    script = ("Scene 1 (0-5 sec):\n[city skyline at dawn] The city wakes up before sunrise.\n"
              "[shot: busy train station] Thousands of commuters head to work every day.")
    j = wait(client, client.post("/api/v1/broll/jobs", json={"script": script, "aspect_ratios": ["16:9"]}).json()["job_id"])
    assert j["status"] == "succeeded", j["error"]
    tl = json.loads(read(j["result"]["links"]["timeline"]))
    assert [s["visual_hint"] for s in tl["scenes"]][:2] == ["city skyline at dawn", "busy train station"]
    assert "[" not in tl["transcript"] and "Scene 1" not in tl["transcript"]   # descriptions are not spoken
    r = client.post("/api/v1/broll/jobs", json={"script": "[only a shot description]"})
    assert r.json()["error"]["code"] == "EMPTY_SCRIPT"


def test_voice_catalog_and_unknown_voice(client):
    j = client.get("/api/v1/voices").json()
    ids = {v["id"] for v in j["voices"]}
    assert "gemini:Charon" in ids and "kokoro:af_bella" in ids
    r = client.post("/api/v1/broll/jobs", json={"script": "Hello there world.", "voice": "kokoro:nobody"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = client.post("/api/v1/broll/jobs", json={"script": "Hello there world.", "voice_speed": 3})
    assert r.status_code == 422


def test_speed_and_clean_options_accepted(client):
    j = wait(client, client.post("/api/v1/broll/jobs", json={
        "script": "A quiet morning by the lake. Birds call across the water.", "voice_speed": 1.2,
        "aspect_ratios": ["1:1"]}).json()["job_id"])
    assert j["status"] == "succeeded", j["error"]


def test_narration_language_validation(client):
    r = client.post("/api/v1/broll/jobs", json={"script": "Hello there world.", "narration_language": "xx"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_translation_failure_falls_back_to_original(client):
    """No Gemini key in tests: translation fails cleanly and the job narrates the original text with a warning."""
    j = wait(client, client.post("/api/v1/broll/jobs", json={
        "script": "Farmers walk into green fields at sunrise. Water makes the difference.",
        "narration_language": "ta", "aspect_ratios": ["1:1"]}).json()["job_id"])
    assert j["status"] == "succeeded", j["error"]
    assert any(w.startswith("TRANSLATION_FAILED") for w in j["result"]["warnings"])
