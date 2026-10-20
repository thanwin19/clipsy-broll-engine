"""Job orchestrator: input -> narration -> plan -> retrieve/score -> route + assign -> render -> QA -> outputs."""
import json
import re
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from app.config import get_settings
from app.errors import ApiError
from app.logs import get_logger
from app.pipeline import media
from app.pipeline.interpret import interpret
from app.pipeline.judge import judge_scenes
from app.pipeline.translate import translate_lines
from app.pipeline.voicestyle import resolve as resolve_style
from app.pipeline.music import find_music
from app.pipeline.narration import Narration, Word, align_script, detect_language, even_words, get_asr, get_tts, speech_words
from app.pipeline.planner import Scene, _split_long, plan_scenes, scenes_from_words, script_to_scenes, segment
from app.pipeline.qa import check_video
from app.pipeline.render import RATIOS, concat_and_mux, frame_bounds, mix_audio, render_segment, render_video_multi, scene_cues, write_srt
from app.pipeline.rerank import get_scorer, score_candidates
from app.pipeline.sources import Candidate, CircuitBreaker, CloudflareFlux, download, download_part, search_all
from app.storage import get_storage

log = get_logger("engine")
Progress = Callable[[int, str], None]


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t[name] = round(timer.t.get(name, 0) + time.perf_counter() - self.s, 3)
        return _C()


def ratio_key(r: str) -> str:
    return r.replace(":", "x")


# ------------------------------------------------------------------ narration
def fetch_audio(url: str, work: Path) -> Path:
    import httpx
    s = get_settings()
    ext = Path(url.split("?")[0]).suffix.lower() or ".mp3"
    dest = work / f"input{ext}"
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as c, c.stream("GET", url) as r:
            r.raise_for_status()
            size = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(1 << 16):
                    size += len(chunk)
                    if size > s.max_upload_mb * 1024 * 1024:
                        raise ApiError("FILE_TOO_LARGE", details={"max_mb": s.max_upload_mb})
                    f.write(chunk)
    except ApiError:
        raise
    except Exception as e:
        raise ApiError("INPUT_DOWNLOAD_FAILED", details={"url": url, "reason": str(e)[:200]})
    if ext not in (".mp3", ".wav", ".m4a"):
        raise ApiError("UNSUPPORTED_FILE_TYPE", details={"url": url})
    if not media.has_audio(dest):
        raise ApiError("UNSUPPORTED_FILE_TYPE", "Downloaded file has no audio stream")
    return dest


def cut_tail_blip(src: Path, dest: Path, warnings: list[str]) -> Path:
    """TTS engines sometimes end with a short full-volume click/burst after the speech. If the audio ends with
    silence followed by less than 0.4 s of sound, cut it at the start of that silence and fade out gently."""
    import subprocess
    total = media.duration(src)
    proc = subprocess.run([media.FFMPEG, "-hide_banner", "-nostdin", "-i", str(src), "-af",
                           "silencedetect=noise=-45dB:d=0.12", "-f", "null", "-"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d\.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d\.]+)", proc.stderr)]
    cut = total
    if starts and ends and len(ends) >= len(starts) and 0 < total - ends[-1] < 0.4 and starts[-1] > 0.5:
        cut = starts[-1] + 0.25           # keep a natural pause after the last word, drop the blip
        warnings.append("TTS_TAIL_CLEANED: removed a short noise burst at the end of the generated voice")
    fade = max(0.0, cut - 0.15)
    media.ffmpeg("-i", src, "-t", f"{cut:.3f}", "-af", f"afade=t=out:st={fade:.3f}:d=0.15", dest)
    return dest


def build_narration(req: dict, work: Path, warnings: list[str],
                    progress: Progress = lambda *a: None) -> tuple[Narration, list[list[Word]] | None, list]:
    s = get_settings()
    asr, groups, hints = get_asr(), None, None
    providers = {"asr": {"name": asr.name, "version": asr.version}}
    if req.get("audio_url") and not req.get("audio_path"):
        req["audio_path"] = str(fetch_audio(req["audio_url"], work))
    if req.get("audio_path"):
        src = Path(req["audio_path"])
        dur = media.duration(src)
        if dur > s.max_audio_seconds:
            raise ApiError("AUDIO_TOO_LONG", details={"seconds": round(dur, 1), "max": s.max_audio_seconds})
        if dur < 0.5:
            raise ApiError("NO_SPEECH_DETECTED", "Audio is shorter than 0.5 seconds")
        wav16 = media.to_wav(src, work / "asr.wav", 16000)
        clean = req.get("clean_audio", True)
        mix = media.to_wav(src, work / "narration.wav", 48000, media.CLEAN_FILTER if clean else None)
        speed = float(req.get("voice_speed") or 1.0)
        if abs(speed - 1.0) > 0.01:
            warnings.append("VOICE_SPEED_IGNORED: voice_speed applies to generated narration, not uploaded audio")
        if media.mean_volume_db(wav16) < -55:
            raise ApiError("NO_SPEECH_DETECTED", "The audio is silent", {"mean_volume_db": media.mean_volume_db(wav16)})
        words, lang = asr.transcribe(wav16, req.get("language"))
        speech = sum(w.end - w.start for w in words)
        if len(words) < 2 or speech < 0.15 * dur and len(words) < 0.5 * dur:
            raise ApiError("NO_SPEECH_DETECTED", "No narration found (silent or music-only audio)",
                           {"words_found": len(words), "seconds": round(dur, 1)})
        total = media.duration(mix)
        source = "audio"
    else:
        if req.get("script") and not req.get("target_seconds"):
            m = re.search(r"(\d{2,3})\s*[- ]?\s*(?:sec|second)s?\b", req["script"][:600], re.I)
            if m and 5 <= int(m.group(1)) <= 600:
                req = {**req, "target_seconds": int(m.group(1))}
        if req.get("script") and not req.get("scenes"):
            progress(6, "reading your prompt")
            read, info, rw = interpret(req["script"])  # prompt documents: speak only the voiceover
            warnings += rw
            if read is not None:
                providers["prompt_reader"] = info
                if not read:
                    raise ApiError("EMPTY_SCRIPT", "No voiceover / spoken lines were found in the prompt",
                                   {"reader": info.get("reader")})
                req = {**req, "scenes": read, "script": None}
        if req.get("script") and not req.get("scenes"):
            shots = script_to_scenes(req["script"])   # inline [shot descriptions] are never spoken
            if shots is not None:
                if not shots:
                    raise ApiError("EMPTY_SCRIPT", "Only shot descriptions in [brackets] were given")
                req = {**req, "scenes": shots, "script": None}
        if req.get("scenes"):
            texts = [(sc.get("text") or "").strip() for sc in req["scenes"]]
            hints = [(sc.get("visual_hint"), sc.get("overlay")) for sc in req["scenes"] if (sc.get("text") or "").strip()]
            texts = [t for t in texts if t]
            script = " ".join(texts)
        else:
            script, texts = (req.get("script") or "").strip(), None
        if not script:
            raise ApiError("EMPTY_SCRIPT")
        target = (req.get("narration_language") or "").lower()
        if target and target != detect_language(script):
            # speak the script in another language: translate scene by scene (or sentence by sentence) and keep
            # the original line as the scene's visual hint, so footage still follows the original meaning
            if not texts:
                texts = [t for t in re.split(r"(?<=[.!?।])\s+", script) if t.strip()]
                hints = [(None, None)] * len(texts)
            try:
                progress(8, f"translating to {target}")
                translated, model = translate_lines(texts, target)
                hints = [(h or orig, ov) for (h, ov), orig in zip(hints, texts)]
                providers["translation"] = {"to": target, "model": model, "lines": len(texts)}
                texts, script = translated, " ".join(translated)
                voice = req.get("voice") or "en-IN-female"
                if ":" not in voice:   # generic voice -> same gender in the target language
                    gender = "male" if ("male" in voice and "female" not in voice) else "female"
                    req = {**req, "voice": f"{target}-IN-{gender}" if target != "en" else f"en-IN-{gender}"}
            except ApiError as e:
                warnings.append(f"TRANSLATION_FAILED: narrating in the original language ({e.message})")
        tts = get_tts()
        style, tts_voice, style_speed, style_af = resolve_style(req.get("voice_style"), req.get("voice") or "en-IN-female", script)
        progress(10, "generating the voice")
        raw = tts.synthesize(script, tts_voice, work / "tts.wav")
        warnings += getattr(tts, "warnings", [])
        providers["tts"] = {"name": tts.name, "version": tts.version, "voice": req.get("voice"),
                            "voice_style": style, "requested_style": req.get("voice_style") or "natural"}
        raw = cut_tail_blip(raw, work / "tts_clean.wav", warnings)
        # TTS output can start/end with long silences: trim them (keeping ~0.3 s padding) so narration covers the video
        trimmed = work / "tts_trimmed.wav"
        media.ffmpeg("-i", raw, "-af", "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3,"
                     "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.4,areverse",
                     trimmed)
        if media.duration(trimmed) > 0.5:
            raw = trimmed
        speed = min(1.3, max(0.75, float(req.get("voice_speed") or 1.0) * style_speed))
        target = float(req.get("target_seconds") or 0)
        if target:   # fit the requested length: adjust pace within natural limits (0.85x-1.15x)
            spoken = media.duration(raw) / speed
            speed = min(1.15 * speed, max(0.85 * speed, speed * spoken / target)) if spoken else speed
            speed = min(1.3, max(0.7, speed))
        chain = [x for x in (style_af, f"atempo={speed:.3f}" if abs(speed - 1.0) >= 0.01 else "", media.LEVEL_FILTER) if x]
        try:
            mix = media.to_wav(raw, work / "narration.wav", 48000, ",".join(chain))
        except ApiError:   # e.g. an ffmpeg build without rubberband: keep speed + levelling, drop the shaping
            warnings.append(f"VOICE_STYLE_SHAPING_SKIPPED: audio shaping for style '{style}' unavailable")
            mix = media.to_wav(raw, work / "narration.wav", 48000, ",".join(chain[1:]) or None)
        if abs(speed - 1.0) >= 0.01:   # keep ASR timing consistent with the narration actually used
            raw = media.to_wav(raw, work / "tts_speed.wav", 24000, f"atempo={speed:.3f}")
        total = media.duration(mix)
        if target and total < target - 0.3:   # still short: hold the last shot (music continues) up to the target
            pad = min(target - total, max(3.0, 0.35 * total))
            padded = work / "narration_padded.wav"
            media.ffmpeg("-i", mix, "-af", f"apad=pad_dur={pad:.3f}", padded)
            mix, total = padded, media.duration(padded)
            warnings.append(f"DURATION_FITTED: narration slowed to fit; last shot held {pad:.1f} s to reach the requested {target:.0f} s")
        elif target and abs(total - target) > 1.5:
            warnings.append(f"DURATION_FITTED: video is {total:.0f} s; the requested {target:.0f} s needs a longer or shorter script")
        text_lang = detect_language(script)
        asr = get_asr(for_alignment=(text_lang == "en"))   # non-English scripts: the larger model times them better
        if text_lang != "en" and not req.get("language"):
            req = {**req, "language": text_lang}
        providers["asr"] = {"name": asr.name, "version": asr.version, "purpose": "timing of known script words",
                            "script_language": text_lang}
        progress(16, "timing every word")
        if asr.name == "mock-asr":
            words, lang = even_words(script, 0.2, total - 0.2), "en"
        elif text_lang != "en":   # known non-English text: timing from the voice's speech regions (fast, no ASR)
            words, lang = speech_words(script, raw), text_lang
            providers["asr"] = {"name": "speech-region alignment", "version": "silencedetect",
                                "purpose": "timing of known script words", "script_language": text_lang}
        else:
            words, lang = asr.transcribe(media.to_wav(raw, work / "asr.wav", 16000), req.get("language"))
            if len(words) >= 2:
                words = align_script(script, words)
            if len(words) < 2:
                words, lang = even_words(script, 0.2, total - 0.2), "en"
                warnings.append("ASR_ON_TTS_EMPTY: timestamps estimated evenly over TTS audio")
        if texts:  # map user scene list onto word timings by token proportion
            counts = [len(t.split()) for t in texts]
            n, acc, groups, prev = sum(counts), 0, [], 0
            for c in counts:
                acc += c
                cut = round(acc / n * len(words))
                if cut > prev:
                    groups.append(words[prev:cut])
                    prev = cut
            if len(groups) != len(hints):
                hints = (hints + [(None, None)] * len(groups))[: len(groups)]
            # a long shot is split on phrase boundaries into several clips that share its description
            split_g, split_h = [], []
            for g, (h, overlay) in zip(groups, hints):
                for k, piece in enumerate(_split_long(g, s.max_scene_seconds, s.min_scene_seconds)):
                    split_g.append(piece)
                    split_h.append((h, overlay if k == 0 else None))   # on-screen text only on the first clip
            groups, hints = split_g, split_h
        source = "scenes" if texts else "script"
    transcript = " ".join(w.text for w in words)
    nar = Narration(audio_path=mix, words=words, duration=total, language=lang, transcript=transcript, providers=providers)
    nar.providers["input_type"] = source
    return nar, groups, hints


# ------------------------------------------------------------------ selection
def card_for(scene: Scene) -> Candidate:
    if scene.overlay and not scene.stat:   # the prompt's on-screen text becomes the card itself (no double text)
        scene._overlay_on_card = True
        c = Candidate(id=f"card:{scene.index}", provider="card-renderer", kind="card", title=f"Card: {scene.overlay[:60]}",
                      page_url="", download_url="", thumb_url="", licence="Rendered by this service (no third-party rights)",
                      licence_url="", attribution="Generated card (IncuBrix B-roll engine)")
        c.score_detail.update({"headline": scene.overlay, "body": "", "card_kind": "text"})
        return c
    if scene.stat:
        head, body, kind = str(scene.stat.get("value", "")), scene.stat.get("label") or scene.text, "stat"
    else:
        head, body, kind = scene.text, "", "text"
    c = Candidate(id=f"card:{scene.index}", provider="card-renderer", kind="card", title=f"Card: {head[:60]}",
                  page_url="", download_url="", thumb_url="", licence="Rendered by this service (no third-party rights)",
                  licence_url="", attribution="Generated card (IncuBrix B-roll engine)")
    c.score_detail.update({"headline": head, "body": body, "card_kind": kind})
    return c


def select_clips(plan, nar: Narration, req: dict, work: Path, warnings: list[str], progress: Progress, timer: Timer):
    s = get_settings()
    breaker = CircuitBreaker(warnings)
    scorer = get_scorer()
    avoid = plan.avoid_terms
    avoid_faces = any(re.search(r"face|people|person", a, re.I) for a in avoid)
    pools: dict[int, list[Candidate]] = {}
    scenes = plan.scenes
    from concurrent.futures import ThreadPoolExecutor

    def work_scene(sc: Scene):
        if sc.type == "statistic" and sc.stat:
            return sc.index, []
        queries = list(sc.queries)
        if sc.sensitive:
            queries = [q + " calm" if "calm" not in q else q for q in queries]
        cands = search_all(queries, sc.category, breaker, s.queries_per_scene)
        return sc.index, score_candidates(sc, cands, avoid, avoid_faces, scorer, work / "thumbs")

    done = 0
    with timer("search_and_score"), ThreadPoolExecutor(max_workers=6) as ex:
        for idx, ranked in ex.map(work_scene, scenes):
            pools[idx] = ranked
            done += 1
            progress(30 + int(30 * done / len(scenes)), f"scored scene {done}/{len(scenes)}")

    # global assignment: best (scene, clip) pairs first, each clip used at most once
    tau = s.stock_min_score
    chosen: dict[int, Candidate] = {}
    reasons: dict[int, str] = {}
    used: set[str] = set()
    pairs = sorted(((c.score, sc.index, c) for sc in scenes for c in pools.get(sc.index, [])), key=lambda x: -x[0])
    for score, idx, c in pairs:
        if idx in chosen or c.id in used or score < tau:
            continue
        chosen[idx], reasons[idx] = c, (f"stock: {c.provider} clip scored {score:.3f} >= tau {tau:.2f}; "
                                        f"best unused of {len(pools[idx])} safe candidates")
        used.add(c.id)

    flux = CloudflareFlux()
    with timer("ai_generation"):
        for sc in scenes:
            if sc.type == "statistic" and sc.stat:
                chosen[sc.index] = card_for(sc)
                reasons[sc.index] = f"statistic '{sc.stat.get('value')}' -> animated stat card (rule: numbers are shown, not searched)"
                continue
            cur = chosen.get(sc.index)
            weak = cur is None or cur.score < tau + 0.03
            if weak and sc.ai_allowed and flux.available():
                try:
                    ai = flux.generate(sc.ai_prompt, work / "ai" / f"scene_{sc.index}.jpg", s.seed + sc.index)
                    scored = score_candidates(sc, [ai], avoid, avoid_faces, scorer, work / "thumbs")
                    if scored and (cur is None or scored[0].score >= cur.score + s.ai_margin):
                        if cur:
                            used.discard(cur.id)
                        chosen[sc.index] = scored[0]
                        reasons[sc.index] = (f"ai: generated image scored {scored[0].score:.3f} vs best stock "
                                             f"{(cur.score if cur else 0):.3f} (rule: AI wins only if >= stock + {s.ai_margin})")
                        continue
                except Exception as e:
                    breaker.record("cloudflare", e)
            if sc.index not in chosen:
                # nothing above tau: take the best remaining safe clip only if it is close, else a text card
                rest = [c for c in pools.get(sc.index, []) if c.id not in used and c.score >= tau - 0.03]
                if rest:
                    chosen[sc.index] = rest[0]
                    used.add(rest[0].id)
                    reasons[sc.index] = (f"stock (low confidence): {rest[0].provider} scored {rest[0].score:.3f} "
                                         f"(< tau {tau:.2f}, within 0.03); no AI option")
                    warnings.append(f"LOW_RELEVANCE: scene {sc.index} best clip scored {rest[0].score:.3f}")
                else:
                    chosen[sc.index] = card_for(sc)
                    reasons[sc.index] = "card: no safe clip scored near tau and AI not allowed/available -> text card"
            if not sc.ai_allowed and (sc.entities or sc.sensitive):
                reasons[sc.index] += ("; safety: " + ("named person/brand -> neutral footage, no AI faces/logos"
                                                       if sc.entities else "sensitive topic -> calm respectful footage"))

    # download chosen clips in parallel; corrupt/failed download -> next best unused candidate -> card
    lock = threading.Lock()

    def fetch(sc: Scene):
        c, tried = chosen[sc.index], 0
        while c.kind in ("video", "image") and c.local_path is None:
            try:
                c.local_path = download_part(c, work / "clips", sc.duration)
            except Exception as e:
                tried += 1
                with lock:
                    breaker.record(c.provider, e)
                    warnings.append(f"CLIP_DOWNLOAD_FAILED: scene {sc.index} {c.id} ({type(e).__name__}); replaced")
                    used.discard(c.id)
                    nxt = [x for x in pools.get(sc.index, []) if x.id not in used and x.id != c.id
                           and x.score >= tau - 0.03 and x.local_path is None] if tried < 3 else []
                    if nxt:
                        c = nxt[0]
                        used.add(c.id)
                        reasons[sc.index] = f"replacement after failed download: {c.provider} scored {c.score:.3f}"
                    else:
                        c = card_for(sc)
                        reasons[sc.index] = "card: all candidate downloads failed"
                    chosen[sc.index] = c

    with timer("download"), ThreadPoolExecutor(max_workers=4) as ex:
        for i, _ in enumerate(ex.map(fetch, scenes)):
            progress(60 + int(8 * (i + 1) / len(scenes)), "downloading clips")
    if not any(c.kind in ("video", "ai_image") for c in chosen.values()) and any(pools.values()) is False:
        warnings.append("ALL_FOOTAGE_SOURCES_EMPTY: every scene uses a rendered card")
    return chosen, reasons, pools


# ------------------------------------------------------------------ outputs
def clip_record(sc: Scene, c: Candidate, reason: str, alternatives: list[Candidate], judge: dict | None = None) -> dict:
    return {
        "scene": sc.index, "start": round(sc.start, 3), "end": round(sc.end, 3), "duration": round(sc.duration, 3),
        "narration": sc.text, "visual_hint": sc.hint, "scene_type": sc.type, "category": sc.category,
        "visual_plan": sc.visual, "queries": sc.queries, "negative_terms": sc.negative, "entities": sc.entities,
        "clip": {"id": c.id, "kind": c.kind, "title": c.title, "source": c.provider, "source_url": c.page_url,
                 "media_url": c.download_url or None, "licence": c.licence, "licence_url": c.licence_url,
                 "attribution": c.attribution, "commercial_use": True, "watermark": False,
                 "ai_generated": c.ai_generated or c.kind == "ai_image", "matched_query": c.query or None},
        "reason": reason,
        "relevance_score": round(c.score, 4) if c.kind != "card" else None,
        "judge": judge,   # independent VLM rating 1-5 (Gemini vision), see app/pipeline/judge.py
        "relevance_detail": {k: v for k, v in c.score_detail.items() if k not in ("headline", "body", "card_kind")},
        "reframe_subject_x": round(c.subject_x, 3),
        "alternatives": [{"id": a.id, "source": a.provider, "score": a.score, "title": a.title[:80], "thumb": a.thumb_url or None,
                          "kind": a.kind, "licence": a.licence} for a in alternatives[:8]],
    }


def render_all(scenes, chosen, nar, music, ratios, burn, work: Path, progress: Progress, only: set[int] | None = None):
    """Renders every (ratio, scene) segment in parallel (RENDER_PARALLEL ffmpeg processes), then joins each ratio."""
    from concurrent.futures import ThreadPoolExecutor
    s = get_settings()
    bounds = frame_bounds(scenes, s.fps)
    tasks, paths = [], {}
    for ratio in ratios:
        seg_dir = work / "segments" / ratio_key(ratio)
        for sc, (a, b) in zip(scenes, bounds):
            name = f"seg_{sc.index:03d}"
            p = seg_dir / f"{name}.mp4"
            paths[(ratio, sc.index)] = p
            if only is None or sc.index in only or not p.exists():
                tasks.append((sc, ratio, max(1, b - a), seg_dir, name))
    done = [0]
    lock = threading.Lock()
    # group by scene: a video clip is decoded once and split into all its ratios (render_video_multi)
    groups: dict[int, list] = {}
    for t in tasks:
        groups.setdefault(t[0].index, []).append(t)

    def tick(k):
        with lock:
            done[0] += k
            if only is None:
                progress(68 + int(22 * done[0] / max(1, len(tasks))), f"rendering {done[0]}/{len(tasks)} segments")

    def one_scene(group):
        sc = group[0][0]
        cand = chosen[sc.index]
        if cand.kind == "video" and s.multi_ratio_render:
            try:
                render_video_multi(sc, cand, [(t[1], t[3]) for t in group], group[0][2], burn, group[0][4])
                tick(len(group))
                return
            except Exception as e:
                log.warning("multi-ratio render failed for scene %s, rendering ratios separately: %s", sc.index, e)
        for sc_, ratio, frames, seg_dir, name in group:
            render_segment(sc_, cand, ratio, frames, seg_dir, burn, name)
            tick(1)

    with ThreadPoolExecutor(max_workers=max(1, s.render_parallel)) as ex:
        list(ex.map(one_scene, groups.values()))
    audio = mix_audio(nar.audio_path, music, nar.duration, work / "mix.m4a")   # once, shared by all ratios
    outputs = {r: work / f"video_{ratio_key(r)}.mp4" for r in ratios}
    with ThreadPoolExecutor(max_workers=max(1, len(ratios))) as ex:
        list(ex.map(lambda r: concat_and_mux([paths[(r, sc.index)] for sc in scenes], audio, nar.duration, outputs[r]),
                    ratios))
    return outputs


def qa_all(outputs: dict, seconds: float) -> dict:
    """QA every output file in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(outputs))) as ex:
        futs = {r: ex.submit(check_video, p, seconds) for r, p in outputs.items()}
    return {r: f.result() for r, f in futs.items()}


def qa_and_fix(outputs, scenes, chosen, pools, nar, music, burn, work, warnings, reasons):
    """If a check finds black/frozen frames inside a scene, swap that scene for its next-best clip once."""
    reports = qa_all(outputs, nar.duration)
    bad: set[int] = set()
    for rep in reports.values():
        times = [b[0] for b in rep["black_segments"]] + rep["freeze_starts"]
        for t in times:
            for sc in scenes:
                if sc.start - 0.05 <= t < sc.end:
                    bad.add(sc.index)
    if bad:
        used = {c.id for c in chosen.values()}
        for idx in bad:
            alt = [c for c in pools.get(idx, []) if c.id not in used and c.kind == "video"]
            for c in alt[:2]:
                try:
                    c.local_path = download(c, work / "clips")
                    chosen[idx] = c
                    used.add(c.id)
                    reasons[idx] = f"auto-replaced by QA (black/frozen frames); {c.provider} scored {c.score:.3f}"
                    break
                except Exception:
                    continue
            else:
                chosen[idx] = card_for(next(s for s in scenes if s.index == idx))
                reasons[idx] = "auto-replaced by QA with card (black/frozen frames, no alternative)"
        warnings.append(f"QA_AUTOFIX: scenes {sorted(bad)} had black/frozen frames and were replaced")
        outputs.update(render_all(scenes, chosen, nar, music, list(outputs), burn, work, lambda *_: None, only=bad))
        reports = qa_all(outputs, nar.duration)
        for r, rep in reports.items():
            if not rep["passed"]:
                warnings.append(f"QA_FAILED: {r} still fails checks after auto-fix: "
                                f"black={rep['black_segments']} freeze={rep['freeze_starts']} drift={rep['av_drift_seconds']}")
    return reports


def publish(job_id: str, version: int, files: dict[str, Path]) -> dict:
    st = get_storage()
    links = {}
    for name, p in files.items():
        links[name] = st.put(p, f"jobs/{job_id}/v{version}/{p.name}")
    return links


def build_manifest(job_id, req, nar, plan, music, timer, warnings, qa, chosen, version, extra=None) -> dict:
    s = get_settings()
    scorer = get_scorer()
    sources = sorted({c.provider for c in chosen.values()})
    return {
        "job_id": job_id, "version": version, "service": {"name": s.app_name, "version": s.app_version},
        "providers": {**nar.providers,
                      "planner": plan.planner,
                      "relevance_model": {"name": scorer.name, "version": getattr(scorer, "version", "")},
                      "footage_sources_enabled": s.provider_list(),
                      "footage_sources_used": sources,
                      "ai_image": {"name": "cloudflare-flux", "model": s.cloudflare_model} if CloudflareFlux().available() else None,
                      "music": music.source if music else None,
                      "renderer": "ffmpeg", "face_detector": "opencv-haarcascade", "saliency": "spectral-residual (numpy)"},
        "settings": {"aspect_ratios": req.get("aspect_ratios"), "style": req.get("style"), "avoid": req.get("avoid"),
                     "captions": req.get("captions"), "voice": req.get("voice"), "fps": s.fps,
                     "stock_min_score_tau": s.stock_min_score, "ai_margin": s.ai_margin,
                     "min_scene_seconds": s.min_scene_seconds, "max_scene_seconds": s.max_scene_seconds,
                     "video_preset": s.video_preset, "crf": s.video_crf},
        "seed": s.seed,
        "narration": {"seconds": round(nar.duration, 3), "language": nar.language, "words": len(nar.words)},
        "mood": plan.mood, "topic": plan.topic, "avoid_terms_expanded": plan.avoid_terms,
        "timings_seconds": timer.t,
        "cost_estimate_usd": {"total": 0.0, "note": "All providers used on free tiers / open-source models; "
                                                    "compute is the host machine."},
        "warnings": warnings,
        "rights": {"all_commercial_use": True,
                   "clips": [{"scene": i, "source": c.provider, "licence": c.licence, "attribution": c.attribution,
                              "ai_generated": c.ai_generated or c.kind == "ai_image"} for i, c in sorted(chosen.items())],
                   "music": ({"title": music.title, "source": music.source, "licence": music.licence,
                              "attribution": music.attribution, "url": music.page_url} if music else None),
                   "narration": "user-supplied audio" if req.get("audio_path") else "synthetic TTS voice (no real person)"},
        "qa": qa,
        **(extra or {}),
    }


# ------------------------------------------------------------------ entry points
def run_job(job_id: str, req: dict, progress: Progress) -> dict:
    s = get_settings()
    work = s.data_dir / "work" / job_id
    work.mkdir(parents=True, exist_ok=True)
    timer, warnings = Timer(), []
    ratios = req.get("aspect_ratios") or ["16:9"]
    caps = req.get("captions") or {"burn_in": True, "srt": True}
    burn = bool(caps.get("burn_in", True))
    t0 = time.perf_counter()

    progress(5, "preparing narration")
    with timer("narration"):
        nar, groups, hints = build_narration(req, work, warnings, progress)
    progress(20, "planning scenes")
    with timer("planning"):
        scenes = scenes_from_words(nar.words, nar.duration, s.min_scene_seconds, s.max_scene_seconds, groups, hints)
        plan = plan_scenes(scenes, nar.transcript, req.get("style") or "", req.get("avoid") or [])
        warnings += plan.warnings
    progress(30, "finding footage")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:   # music (needs only the mood) downloads while footage is found
        t_music = time.perf_counter()
        music_mood = req.get("music_mood") if req.get("music_mood") not in (None, "auto") else plan.mood
        music_future = ex.submit(find_music, music_mood, nar.duration, work) if req.get("music", True) else None
        chosen, reasons, pools = select_clips(plan, nar, req, work, warnings, progress, timer)
        music = None
        if music_future:
            music, mw = music_future.result()
            warnings += mw
        timer.t["music_parallel"] = round(time.perf_counter() - t_music, 3)
    with timer("render"):
        outputs = render_all(plan.scenes, chosen, nar, music, ratios, burn, work, progress)
    progress(92, "quality checks")
    first = ratio_key(ratios[0])
    segs = {sc.index: work / "segments" / first / f"seg_{sc.index:03d}.mp4" for sc in plan.scenes}
    with timer("qa"):
        qa = qa_and_fix(outputs, plan.scenes, chosen, pools, nar, music, burn, work, warnings, reasons)
    judged, judge_model = {}, "pending"   # the independent judge runs after delivery (see finish_judge)
    srt = write_srt([c for sc in plan.scenes for c in scene_cues(sc)], work / "captions.srt")
    preview_files, thumb_urls, wave_url = make_previews(job_id, 1, work, work / "previews", plan.scenes, ratios[0],
                                                        nar.audio_path)
    timeline = {"job_id": job_id, "version": 1, "waveform": wave_url, "narration_seconds": round(nar.duration, 3), "language": nar.language,
                "transcript": nar.transcript, "aspect_ratios": ratios, "mood": plan.mood,
                "selection_rule": selection_rule_text(),
                "scenes": [clip_record(sc, chosen[sc.index], reasons[sc.index], pools.get(sc.index, []), judged.get(sc.index))
                           for sc in plan.scenes],
                "music": asdict(music) | {"path": None} if music else None}
    for rec in timeline["scenes"]:
        rec["thumbnail"] = thumb_urls.get(rec["scene"])
    timer.t["total"] = round(time.perf_counter() - t0, 3)
    scores = [v["score"] for v in judged.values()]
    manifest = build_manifest(job_id, req, nar, plan, music, timer, warnings, qa, chosen, 1, {"independent_judge": {
        "model": judge_model, "scenes_scored": len(scores), "of": len(plan.scenes),
        "mean_score_1_to_5": round(sum(scores) / len(scores), 2) if scores else None,
        "share_scored_4_or_5": round(sum(x >= 4 for x in scores) / len(scores), 3) if scores else None}})
    (work / "timeline.json").write_text(json.dumps(timeline, indent=2, default=str), encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    save_state(work, plan, chosen, pools, reasons, nar, music, req, ratios, burn)

    progress(96, "uploading outputs")
    files = {f"video_{ratio_key(r)}": p for r, p in outputs.items()}
    files.update({"timeline": work / "timeline.json", "manifest": work / "manifest.json"})
    if caps.get("srt", True):
        files["captions_srt"] = srt
    files.update(preview_files)
    links = publish(job_id, 1, files)
    if get_settings().judge_provider != "none":
        threading.Thread(target=finish_judge, args=(job_id, work, plan.scenes, segs), daemon=True).start()
    return {"version": 1, "links": links, "warnings": warnings, "scenes": len(plan.scenes),
            "narration_seconds": round(nar.duration, 3), "timings_seconds": timer.t,
            "qa_passed": all(r["passed"] for r in qa.values())}


def finish_judge(job_id: str, work: Path, scenes, segs: dict) -> None:
    """Runs the independent VLM judge after the video was delivered, then updates timeline.json and
    manifest.json in place (same download links)."""
    try:
        t = time.perf_counter()
        judged, model, jw = judge_scenes(scenes, segs, work)
        tl = json.loads((work / "timeline.json").read_text(encoding="utf-8"))
        mf = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        for rec in tl["scenes"]:
            rec["judge"] = judged.get(rec["scene"])
        scores = [v["score"] for v in judged.values()]
        mf["independent_judge"] = {"model": model, "scenes_scored": len(scores), "of": len(scenes),
                                   "mean_score_1_to_5": round(sum(scores) / len(scores), 2) if scores else None,
                                   "share_scored_4_or_5": round(sum(x >= 4 for x in scores) / len(scores), 3) if scores else None,
                                   "seconds_after_delivery": round(time.perf_counter() - t, 2)}
        if jw:
            mf["warnings"].append(jw)
        (work / "timeline.json").write_text(json.dumps(tl, indent=2, default=str), encoding="utf-8")
        (work / "manifest.json").write_text(json.dumps(mf, indent=2, default=str), encoding="utf-8")
        publish(job_id, 1, {"timeline": work / "timeline.json", "manifest": work / "manifest.json"})
    except Exception as e:
        log.warning("background judge failed for %s: %s", job_id, e)


def make_previews(job_id: str, version: int, work: Path, out_dir: Path, scenes, ratio: str, narration: Path):
    """Editor assets: a 320-px frame from the middle of every scene and a waveform image of the narration.
    Returns (files to publish, {scene_index: url}, waveform url)."""
    from concurrent.futures import ThreadPoolExecutor
    st = get_storage()
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = work / "segments" / ratio_key(ratio)

    def thumb(sc):
        dest = out_dir / f"thumb_{sc.index:03d}.jpg"
        try:
            media.ffmpeg("-ss", f"{max(0.0, sc.duration / 2):.2f}", "-i", seg_dir / f"seg_{sc.index:03d}.mp4",
                         "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "5", dest)
            return sc.index, dest
        except Exception:
            return sc.index, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        thumbs = dict(ex.map(thumb, scenes))
    files, urls = {}, {}
    for i, p in thumbs.items():
        if p:
            files[f"thumb_{i:03d}"] = p
            urls[i] = st.url(f"jobs/{job_id}/v{version}/{p.name}")
    wave = out_dir / "waveform.png"
    wave_url = None
    try:
        media.ffmpeg("-i", narration, "-filter_complex", "aformat=channel_layouts=mono,showwavespic=s=2400x120:colors=0x22D3EE",
                     "-frames:v", "1", wave)
        files["waveform"] = wave
        wave_url = st.url(f"jobs/{job_id}/v{version}/{wave.name}")
    except Exception:
        pass
    return files, urls, wave_url


def selection_rule_text() -> str:
    s = get_settings()
    return (f"Per scene: (1) statistics -> animated stat card. (2) Stock candidates from category-suited sources are "
            f"scored with CLIP (image vs scene visual plan + narration, minus penalty for negative/avoid terms); unsafe "
            f"candidates (avoid list, faces when avoided, AI for people/brands/sensitive) are removed. (3) Global "
            f"assignment takes the highest (scene, clip) pairs first, each clip used once, if score >= tau={s.stock_min_score}. "
            f"(4) If a scene's best stock score < tau+0.03 and AI is allowed, an AI image is generated and wins only if "
            f"its CLIP score >= stock + {s.ai_margin}. (5) Otherwise the best unused clip within 0.03 of tau, else a text card.")


# ------------------------------------------------------------------ swap a clip
def save_state(work, plan, chosen, pools, reasons, nar, music, req, ratios, burn):
    import pickle
    with open(work / "state.pkl", "wb") as f:
        pickle.dump({"plan": plan, "chosen": chosen, "pools": pools, "reasons": reasons, "nar": nar,
                     "music": music, "req": req, "ratios": ratios, "burn": burn}, f)


def swap_scene(parent_id: str, job_id: str, scene_index: int, body: dict, progress: Progress) -> dict:
    import pickle
    s = get_settings()
    work = s.data_dir / "work" / parent_id
    state_file = work / "state.pkl"
    if not state_file.exists():
        raise ApiError("JOB_NOT_FOUND", "Parent job state not found")
    with open(state_file, "rb") as f:
        st = pickle.load(f)
    plan, chosen, pools, reasons = st["plan"], st["chosen"], st["pools"], st["reasons"]
    scene = next((x for x in plan.scenes if x.index == scene_index), None)
    if scene is None:
        raise ApiError("SCENE_NOT_FOUND", details={"scene": scene_index, "scenes": len(plan.scenes)})
    timer, warnings = Timer(), []
    t0 = time.perf_counter()
    used = {c.id for c in chosen.values()}
    old = chosen[scene_index]
    progress(10, "choosing replacement")
    with timer("select"):
        new = None
        if body.get("candidate_id"):
            new = next((c for c in pools.get(scene_index, []) if c.id == body["candidate_id"]), None)
            if new is None:
                raise ApiError("VALIDATION_ERROR", "candidate_id is not among this scene's alternatives")
        elif body.get("query"):
            breaker = CircuitBreaker(warnings)
            cands = search_all([body["query"]], scene.category, breaker, 1)
            ranked = score_candidates(scene, cands, plan.avoid_terms,
                                      any(re.search(r"face|people", a, re.I) for a in plan.avoid_terms),
                                      get_scorer(), work / "thumbs")
            pools[scene_index] = ranked + [c for c in pools.get(scene_index, []) if c.id not in {r.id for r in ranked}]
            new = next((c for c in ranked if c.id not in used), None)
        else:
            new = next((c for c in pools.get(scene_index, []) if c.id not in used), None)
        if new is None:
            new = card_for(scene)
        if new.kind in ("video", "image") and new.local_path is None:
            new.local_path = download(new, work / "clips")
    chosen[scene_index] = new
    reasons[scene_index] = f"user swap: replaced {old.id} with {new.id} (score {new.score:.3f})"
    progress(40, "re-rendering affected scene only")
    with timer("render_affected_scene"):
        outputs = render_all(plan.scenes, chosen, st["nar"], st["music"], st["ratios"], st["burn"], work,
                             lambda *_: None, only={scene_index})
    progress(85, "quality checks")
    with timer("qa"):
        qa = {r: check_video(p, st["nar"].duration) for r, p in outputs.items()}
    version = int(body.get("_version", 2))
    timeline = json.loads((work / "timeline.json").read_text(encoding="utf-8"))
    timeline["version"] = version
    old_judge = {rec["scene"]: rec.get("judge") for rec in timeline.get("scenes", [])}
    timeline["scenes"] = [clip_record(sc, chosen[sc.index], reasons[sc.index], pools.get(sc.index, []),
                                      None if sc.index == scene_index else old_judge.get(sc.index))
                          for sc in plan.scenes]
    out_dir = work / f"v{version}"
    out_dir.mkdir(exist_ok=True)
    preview_files, thumb_urls, wave_url = make_previews(parent_id, version, work, out_dir, plan.scenes, st["ratios"][0],
                                                        st["nar"].audio_path)
    timeline["waveform"] = wave_url
    for rec in timeline["scenes"]:
        rec["thumbnail"] = thumb_urls.get(rec["scene"])
    (out_dir / "timeline.json").write_text(json.dumps(timeline, indent=2, default=str), encoding="utf-8")
    timer.t["total"] = round(time.perf_counter() - t0, 3)
    manifest = build_manifest(job_id, st["req"], st["nar"], plan, st["music"], timer, warnings, qa, chosen, version,
                              {"swap": {"parent_job_id": parent_id, "scene": scene_index, "old_clip": old.id,
                                        "new_clip": new.id, "segments_re_rendered": len(st["ratios"]),
                                        "segments_reused": len(st["ratios"]) * (len(plan.scenes) - 1)}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    for r, p in outputs.items():
        shutil.copyfile(p, out_dir / p.name)
    save_state(work, plan, chosen, pools, reasons, st["nar"], st["music"], st["req"], st["ratios"], st["burn"])
    files = {f"video_{ratio_key(r)}": out_dir / p.name for r, p in outputs.items()}
    files.update({"timeline": out_dir / "timeline.json", "manifest": out_dir / "manifest.json"})
    files.update(preview_files)
    srt = work / "captions.srt"
    if srt.exists():
        files["captions_srt"] = srt
    links = publish(parent_id, version, files)
    return {"version": version, "parent_job_id": parent_id, "scene": scene_index, "links": links,
            "warnings": warnings, "timings_seconds": timer.t, "qa_passed": all(r["passed"] for r in qa.values())}
