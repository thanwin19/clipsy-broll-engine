"""Rendering. Every scene becomes its own normalised segment (1080p-class, 30 fps, yuv420p, H.264) with its
captions burned in, so segments can be joined without re-encoding and a swapped scene re-renders alone."""
import math
import re
from pathlib import Path

from app.config import get_settings
from app.pipeline import media
from app.pipeline.cards import render_card

RATIOS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
TARGET_LUMA = 0.45


def frame_bounds(scenes, fps: int) -> list[tuple[int, int]]:
    """Cumulative rounding: frame counts add up exactly to the narration length (no drift)."""
    return [(round(sc.start * fps), round(sc.end * fps)) for sc in scenes]


# ------------------------------------------------------------- captions
def caption_chunks(words, max_words: int = 7, max_chars: int = 38):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x.text for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w.text) and len(cur) >= 3:
            chunks.append(cur)
            cur = []
    if cur:
        if chunks and len(cur) <= 2 and len(chunks[-1]) + len(cur) <= max_words + 2:
            chunks[-1] = chunks[-1] + cur  # no one-word orphan captions
        else:
            chunks.append(cur)
    return chunks


def _ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def scene_cues(scene) -> list[tuple[float, float, str]]:
    """Cues for one scene, clipped to the scene so no caption crosses a cut."""
    cues = []
    chunks = caption_chunks(scene.words)
    for i, ch in enumerate(chunks):
        start = max(scene.start, ch[0].start)
        end = chunks[i + 1][0].start if i + 1 < len(chunks) else scene.end
        end = min(end, scene.end)
        if end - start > 0.05:
            cues.append((start, end, " ".join(w.text for w in ch)))
    return cues


def write_srt(cues, path: Path, offset: float = 0.0) -> Path:
    lines = []
    for i, (a, b, t) in enumerate(cues, 1):
        lines += [str(i), f"{_ts(a - offset)} --> {_ts(b - offset)}", t, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------- segments
def _cover(sw: int, sh: int, W: int, H: int) -> tuple[int, int]:
    f = max(W / sw, H / sh)
    return math.ceil(sw * f / 2) * 2, math.ceil(sh * f / 2) * 2


def overlay_filter(scene, seg_dir: Path, name: str, W: int, H: int, rel: str = "") -> str:
    """On-screen text from the prompt as a boxed title at the top of the frame for the whole scene."""
    if not getattr(scene, "overlay", None):
        return ""
    import textwrap
    text = "\n".join(textwrap.wrap(scene.overlay, 34 if W >= H else 22)[:3])
    path = seg_dir / f"{name}_title.srt"
    path.write_text(f"1\n00:00:00,000 --> 00:59:59,000\n{text}\n", encoding="utf-8")
    size = 17 if W >= H else 12
    style = (f"FontName=DejaVu Sans,FontSize={size},Bold=1,PrimaryColour=&H00FFFFFF,BackColour=&H66000000,"
             f"BorderStyle=3,Outline=6,Shadow=0,Alignment=8,MarginV={20 if W >= H else 70}")
    return f",subtitles={rel}{path.name}:force_style='{style}'"


def _caption_style(W: int, H: int) -> str:
    size = {1920: 15, 1080: 13}.get(W, 13) if W >= H else 9
    return (f"FontName=DejaVu Sans,FontSize={size},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,"
            f"BorderStyle=1,Outline=1.6,Shadow=0,Alignment=2,MarginV={int(28 if W >= H else 60)}")


def encoder_args(encoder: str) -> list[str]:
    """H.264 encoder settings. h264_amf / h264_nvenc / h264_qsv use the GPU's video engine (much faster);
    libx264 is the portable CPU default. All produce H.264 High, yuv420p, so segments join without re-encoding."""
    s = get_settings()
    common = ["-pix_fmt", "yuv420p", "-profile:v", "high"]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "21", "-qp_p", "23", *common]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", *common]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "23", *common]
    return ["-threads", str(s.ffmpeg_threads), "-c:v", "libx264", "-preset", s.video_preset, "-crf", str(s.video_crf), *common]


def render_segment(scene, cand, ratio: str, frames: int, seg_dir: Path, burn: bool, name: str) -> Path:
    """Renders with the configured encoder; if a GPU encoder fails, retries once with libx264."""
    s = get_settings()
    try:
        return _render_segment(scene, cand, ratio, frames, seg_dir, burn, name, s.video_encoder)
    except Exception:
        if s.video_encoder == "libx264":
            raise
        return _render_segment(scene, cand, ratio, frames, seg_dir, burn, name, "libx264")


def _render_segment(scene, cand, ratio: str, frames: int, seg_dir: Path, burn: bool, name: str, encoder: str) -> Path:
    s = get_settings()
    W, H = RATIOS[ratio]
    seg_dir.mkdir(parents=True, exist_ok=True)
    out = seg_dir / f"{name}.mp4"
    dur = frames / s.fps
    b = max(-0.12, min(0.12, (TARGET_LUMA - (cand.brightness or TARGET_LUMA)) * 0.5))
    norm = f"eq=brightness={b:.3f}:saturation=1.05"
    sub = ""
    cues = scene_cues(scene) if burn and scene.words else []
    if cues:  # an empty .srt makes ffmpeg's subtitles filter fail, so only burn when there is text
        srt = write_srt(cues, seg_dir / f"{name}.srt", offset=scene.start)
        sub = f",subtitles={srt.name}:force_style='{_caption_style(W, H)}'"
    sub += overlay_filter(scene, seg_dir, name, W, H)
    enc = encoder_args(encoder) + ["-r", str(s.fps), "-g", str(s.fps * 2), "-an", "-frames:v", str(frames)]

    if cand.kind == "video":
        vs = media.video_stream(cand.local_path) or {}
        sw, sh = int(vs.get("width") or 1280), int(vs.get("height") or 720)
        cw, ch = _cover(sw, sh, W, H)
        x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
        src_dur = media.duration(cand.local_path)  # real file length (clips may be partial downloads)
        offset = min(1.0, max(0.0, (src_dur - dur) / 3)) if src_dur > dur else 0.0
        loop = ["-stream_loop", "-1"] if src_dur and src_dur < dur + offset else []
        vf = (f"scale={cw}:{ch}:flags=bilinear,crop={W}:{H}:{x}:{(ch - H) // 2},fps={s.fps},{norm},"
              f"setsar=1,format=yuv420p{sub}")
        media.ffmpeg("-filter_threads", str(s.ffmpeg_threads), *loop, "-ss", f"{offset:.2f}", "-i", cand.local_path.resolve(), "-vf", vf, *enc, out.name, cwd=seg_dir)
    elif cand.kind in ("ai_image", "image") and s.parallax:
        # still photo / AI image: 2.5D parallax camera move from a depth map (falls back to Ken Burns on error)
        from app.pipeline.parallax import render_parallax
        try:
            render_parallax(Path(cand.local_path).resolve(), W, H, frames, cand.subject_x, f",{norm}{sub}",
                            enc, out.name, seg_dir, s.fps)
            return out
        except Exception as e:
            from app.logs import get_logger
            get_logger("render").warning("parallax failed for %s, using Ken Burns: %s", cand.id, e)
        return _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out)
    else:  # card: gentle Ken Burns push-in so the shot is never frozen
        return _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out)
    return out


def _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out):
    """Gentle push-in for cards (and fallback for photos) so the shot is never frozen."""
    s = get_settings()
    img = cand.local_path
    if cand.kind == "card":
        img = render_card(seg_dir / f"{name}_card.png", W, H, cand.score_detail.get("headline", ""),
                          cand.score_detail.get("body", ""), cand.score_detail.get("card_kind", "stat"))
    from PIL import Image
    with Image.open(img) as im:
        sw, sh = im.size
    cw, ch = _cover(sw, sh, W, H)
    x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
    zoom = 0.06 if cand.kind != "card" else 0.03
    vf = (f"scale={cw}:{ch},crop={W}:{H}:{x}:{(ch - H) // 2},"
          f"scale={W * 3 // 2}:{H * 3 // 2},zoompan=z='1+{zoom}*on/{max(frames, 1)}':x='iw/2-(iw/zoom/2)':"
          f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={s.fps},{norm},setsar=1,format=yuv420p{sub}")
    media.ffmpeg("-loop", "1", "-framerate", str(s.fps), "-i", Path(img).resolve(), "-vf", vf, *enc, out.name, cwd=seg_dir)
    return out


def render_video_multi(scene, cand, jobs: list[tuple[str, Path]], frames: int, burn: bool, name: str) -> dict:
    """One ffmpeg process per clip: decode (GPU-assisted when available) + colour-normalise ONCE, then split into
    every requested aspect ratio, each with its own subject-aware crop and captions, encoded in parallel.
    jobs = [(ratio, seg_dir)]. Falls back to per-ratio renders on any error."""
    s = get_settings()
    root = jobs[0][1].parent
    for _, d in jobs:
        d.mkdir(parents=True, exist_ok=True)
    dur = frames / s.fps
    vs = media.video_stream(cand.local_path) or {}
    sw, sh = int(vs.get("width") or 1280), int(vs.get("height") or 720)
    src_dur = media.duration(cand.local_path)
    offset = min(1.0, max(0.0, (src_dur - dur) / 3)) if src_dur > dur else 0.0
    loop = ["-stream_loop", "-1"] if src_dur and src_dur < dur + offset else []
    b = max(-0.12, min(0.12, (TARGET_LUMA - (cand.brightness or TARGET_LUMA)) * 0.5))
    cues = scene_cues(scene) if burn and scene.words else []
    n = len(jobs)
    graph = [f"[0:v]fps={s.fps},eq=brightness={b:.3f}:saturation=1.05,split={n}" + "".join(f"[i{k}]" for k in range(n))]
    outs, maps = {}, []
    for k, (ratio, seg_dir) in enumerate(jobs):
        W, H = RATIOS[ratio]
        cw, ch = _cover(sw, sh, W, H)
        x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
        sub = ""
        if cues:
            srt = write_srt(cues, seg_dir / f"{name}.srt", offset=scene.start)
            sub = f",subtitles={seg_dir.name}/{srt.name}:force_style='{_caption_style(W, H)}'"
        sub += overlay_filter(scene, seg_dir, name, W, H, rel=f"{seg_dir.name}/")
        graph.append(f"[i{k}]scale={cw}:{ch}:flags=bilinear,crop={W}:{H}:{x}:{(ch - H) // 2},setsar=1,format=yuv420p{sub}[o{k}]")
        out = seg_dir / f"{name}.mp4"
        outs[ratio] = out
        maps += ["-map", f"[o{k}]", *encoder_args(s.video_encoder), "-r", str(s.fps), "-g", str(s.fps * 2), "-an",
                 "-frames:v", str(frames), f"{seg_dir.name}/{out.name}"]
    hw = ["-hwaccel", "auto"] if s.hw_decode else []
    media.ffmpeg(*hw, *loop, "-ss", f"{offset:.2f}", "-i", cand.local_path.resolve(),
                 "-filter_complex", ";".join(graph), *maps, cwd=root)
    return outs


def mix_audio(narration_wav: Path, music, total: float, out: Path) -> Path:
    """Narration + ducked music, encoded to AAC ONCE; every aspect ratio then copies this track (no re-encode)."""
    s = get_settings()
    if out.exists() and out.stat().st_size > 0:
        return out
    if music:
        media.ffmpeg("-i", narration_wav, "-stream_loop", "-1", "-i", music.path, "-filter_complex",
                     f"[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[v1][v2];"
                     f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={s.music_volume_db}dB,"
                     f"afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5[m];"
                     f"[m][v1]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck];"
                     f"[v2][duck]amix=inputs=2:duration=first:normalize=0[a]",
                     "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}", out)
    else:
        media.ffmpeg("-i", narration_wav, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                     "-t", f"{total:.3f}", out)
    return out


def concat_and_mux(segments: list[Path], audio: Path, total: float, out: Path) -> Path:
    """Joins the segments and the pre-mixed audio with stream copy only (fast, no re-encoding)."""
    work = out.parent
    lst = work / f"{out.stem}_list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8")
    media.ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-map", "0:v", "-map", "1:a",
                 "-c", "copy", "-t", f"{total:.3f}", "-movflags", "+faststart", out)
    return out
