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


def _caption_style(W: int, H: int) -> str:
    size = {1920: 15, 1080: 13}.get(W, 13) if W >= H else 9
    return (f"FontName=DejaVu Sans,FontSize={size},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,"
            f"BorderStyle=1,Outline=1.6,Shadow=0,Alignment=2,MarginV={int(28 if W >= H else 60)}")


def render_segment(scene, cand, ratio: str, frames: int, seg_dir: Path, burn: bool, name: str) -> Path:
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
    enc = ["-c:v", "libx264", "-preset", s.video_preset, "-crf", str(s.video_crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-r", str(s.fps), "-g", str(s.fps * 2), "-an", "-frames:v", str(frames)]

    if cand.kind == "video":
        vs = media.video_stream(cand.local_path) or {}
        sw, sh = int(vs.get("width") or 1280), int(vs.get("height") or 720)
        cw, ch = _cover(sw, sh, W, H)
        x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
        src_dur = cand.duration or media.duration(cand.local_path)
        offset = min(1.0, max(0.0, (src_dur - dur) / 3)) if src_dur > dur else 0.0
        loop = ["-stream_loop", "-1"] if src_dur and src_dur < dur + offset else []
        vf = (f"scale={cw}:{ch}:flags=bicubic,crop={W}:{H}:{x}:{(ch - H) // 2},fps={s.fps},{norm},"
              f"setsar=1,format=yuv420p{sub}")
        media.ffmpeg(*loop, "-ss", f"{offset:.2f}", "-i", cand.local_path.resolve(), "-vf", vf, *enc, out.name, cwd=seg_dir)
    else:  # still image (AI image, card): gentle Ken Burns push-in so the shot is never frozen
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
              f"scale={W * 2}:{H * 2},zoompan=z='1+{zoom}*on/{max(frames, 1)}':x='iw/2-(iw/zoom/2)':"
              f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={s.fps},{norm},setsar=1,format=yuv420p{sub}")
        media.ffmpeg("-loop", "1", "-framerate", str(s.fps), "-i", Path(img).resolve(), "-vf", vf, *enc, out.name, cwd=seg_dir)
    return out


def concat_and_mux(segments: list[Path], narration_wav: Path, music, total: float, out: Path) -> Path:
    s = get_settings()
    work = out.parent
    lst = work / f"{out.stem}_list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8")
    silent = work / f"{out.stem}_video.mp4"
    media.ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent)
    if music:
        media.ffmpeg("-i", silent, "-i", narration_wav, "-stream_loop", "-1", "-i", music.path, "-filter_complex",
                     f"[1:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[v1][v2];"
                     f"[2:a]aresample=48000,aformat=channel_layouts=stereo,volume={s.music_volume_db}dB,"
                     f"afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5[m];"
                     f"[m][v1]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck];"
                     f"[v2][duck]amix=inputs=2:duration=first:normalize=0[a]",
                     "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                     "-t", f"{total:.3f}", "-movflags", "+faststart", out)
    else:
        media.ffmpeg("-i", silent, "-i", narration_wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                     "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", f"{total:.3f}", "-movflags", "+faststart", out)
    silent.unlink(missing_ok=True)
    return out
