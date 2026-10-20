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


def title_png(scene, seg_dir: Path, name: str, W: int, H: int) -> tuple[str, int, int] | None:
    """On-screen text from the prompt, drawn ONCE as a small transparent PNG (rounded dark box, white bold text).
    It is laid over the video with ffmpeg's overlay filter, which is almost free per frame (libass box borders
    cost ~2 s per scene). Returns (file name, x, y) or None."""
    if not getattr(scene, "overlay", None) or getattr(scene, "_overlay_on_card", False):
        return None
    import textwrap
    from PIL import Image, ImageDraw
    from app.pipeline.cards import _font
    landscape = W >= H
    font = _font(int(H * 0.042) if landscape else int(W * 0.05))
    lines = textwrap.wrap(scene.overlay, 40 if landscape else 26)[:4]
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    sizes = [probe.textbbox((0, 0), ln, font=font) for ln in lines]
    lh = max(b[3] - b[1] for b in sizes)
    gap, pad = int(lh * 0.35), int(lh * 0.6)
    bw = max(b[2] - b[0] for b in sizes) + 2 * pad
    bh = len(lines) * lh + (len(lines) - 1) * gap + 2 * pad
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=int(lh * 0.4), fill=(8, 16, 30, 175))
    for i, (ln, b) in enumerate(zip(lines, sizes)):
        d.text(((bw - (b[2] - b[0])) / 2 - b[0], pad + i * (lh + gap) - b[1]), ln, font=font, fill=(255, 255, 255, 255))
    fname = f"{name}_title_{W}x{H}.png"
    img.save(seg_dir / fname)
    return fname, (W - bw) // 2, int(H * (0.05 if landscape else 0.09))


def with_title(chain: str, title, rel: str = "", label: str = "") -> str:
    """Appends the title overlay to a filter chain; `label` names the final output in a complex graph."""
    if not title:
        return chain + (f"[{label}]" if label else "")
    fname, x, y = title
    tag = label or "o"
    return (f"{chain}[{tag}base];movie={rel}{fname}[{tag}t];[{tag}base][{tag}t]overlay={x}:{y}:format=yuv420,scale=out_range=tv,format=yuv420p"
            + (f"[{label}]" if label else ""))


def crop_then_scale(sw: int, sh: int, W: int, H: int, subject_x: float) -> str:
    """Cut the output's aspect ratio out of the SOURCE frame (around the subject), then scale just that region
    to W x H. Same picture as scale-to-cover-then-crop, but never resizes a huge intermediate frame
    (a 720p clip for 9:16 used to be upscaled to 3413x1920 on every frame before cropping)."""
    a = W / H
    if sw / sh > a:
        cw, ch = max(2, int(round(sh * a / 2)) * 2), sh // 2 * 2
        x, y = int(min(max(subject_x * sw - cw / 2, 0), sw - cw)), 0
    else:
        cw, ch = sw // 2 * 2, max(2, int(round(sw / a / 2)) * 2)
        x, y = 0, (sh - ch) // 2
    return f"crop={cw}:{ch}:{x}:{y},scale={W}:{H}:flags=bilinear"


CAPTION_FONTS = {"sans": "DejaVu Sans", "serif": "DejaVu Serif", "mono": "DejaVu Sans Mono", "condensed": "Arial Narrow"}
CAPTION_SIZES = {"small": 0.8, "medium": 1.0, "large": 1.3, "xl": 1.6}


def _ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> ASS &HAABBGGRR."""
    h = (hex_rgb or "#FFFFFF").lstrip("#")
    h = h if len(h) == 6 else "FFFFFF"
    return f"&H{alpha:02X}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _caption_style(W: int, H: int, cap: dict | None = None) -> str:
    """Burned-in caption look. cap: {color, size, font, position, box} from the request (all optional)."""
    cap = cap or {}
    base = {1920: 15, 1080: 13}.get(W, 13) if W >= H else 9
    size = round(base * CAPTION_SIZES.get(cap.get("size") or "medium", 1.0), 1)
    font = CAPTION_FONTS.get(cap.get("font") or "sans", "DejaVu Sans")
    pos = cap.get("position") or "bottom"
    align = {"bottom": 2, "middle": 10, "top": 6}.get(pos, 2)            # SSA numbering
    margin = int(28 if W >= H else 60) if pos != "top" else int(40 if W >= H else 110)
    colour = _ass_colour(cap.get("color") or "#FFFFFF")
    if cap.get("box"):   # solid rounded-ish box behind the text (opaque box style)
        return (f"FontName={font},FontSize={size},Bold=1,PrimaryColour={colour},BackColour=&H66000000,"
                f"OutlineColour=&H66000000,BorderStyle=3,Outline=5,Shadow=0,Alignment={align},MarginV={margin}")
    return (f"FontName={font},FontSize={size},Bold=1,PrimaryColour={colour},OutlineColour=&H00101010,"
            f"BorderStyle=1,Outline=1.6,Shadow=0,Alignment={align},MarginV={margin}")


def encoder_args(encoder: str) -> list[str]:
    """H.264 encoder settings. h264_amf / h264_nvenc / h264_qsv use the GPU's video engine (much faster);
    libx264 is the portable CPU default. All produce H.264 High, yuv420p, so segments join without re-encoding."""
    s = get_settings()
    common = ["-pix_fmt", "yuv420p", "-profile:v", "high"]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "21", "-qp_p", "23", *common]
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
        sub = f",subtitles={srt.name}:force_style='{_caption_style(W, H, getattr(scene, 'caption_style', None))}'"
    title = title_png(scene, seg_dir, name, W, H)
    enc = encoder_args(encoder) + ["-r", str(s.fps), "-g", str(s.fps * 2), "-an", "-frames:v", str(frames)]

    if cand.kind == "video":
        vs = media.video_stream(cand.local_path) or {}
        sw, sh = int(vs.get("width") or 1280), int(vs.get("height") or 720)
        cw, ch = _cover(sw, sh, W, H)
        x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
        src_dur = media.duration(cand.local_path)  # real file length (clips may be partial downloads)
        offset = min(1.0, max(0.0, (src_dur - dur) / 3)) if src_dur > dur else 0.0
        loop = ["-stream_loop", "-1"] if src_dur and src_dur < dur + offset else []
        vf = with_title(f"{crop_then_scale(sw, sh, W, H, cand.subject_x)},fps={s.fps},{norm},"
                        f"setsar=1,format=yuv420p{sub}", title)
        media.ffmpeg("-filter_threads", str(s.ffmpeg_threads), *loop, "-ss", f"{offset:.2f}", "-i", cand.local_path.resolve(), "-vf", vf, *enc, out.name, cwd=seg_dir)
    elif cand.kind in ("ai_image", "image") and s.parallax:
        # still photo / AI image: 2.5D parallax camera move from a depth map (falls back to Ken Burns on error)
        from app.pipeline.parallax import render_parallax
        try:
            render_parallax(Path(cand.local_path).resolve(), W, H, frames, cand.subject_x, with_title(f",{norm}{sub}", title),
                            enc, out.name, seg_dir, s.fps)
            return out
        except Exception as e:
            from app.logs import get_logger
            get_logger("render").warning("parallax failed for %s, using Ken Burns: %s", cand.id, e)
        return _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out, title)
    else:  # card: gentle Ken Burns push-in so the shot is never frozen
        return _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out, title)
    return out


INDIC_RANGES = [(0x0900, 0x0DFF), (0x0600, 0x06FF)]   # Devanagari..Malayalam, Arabic: need complex shaping


def needs_shaping(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in (text or "") for lo, hi in INDIC_RANGES)


def shaped_card_text(text: str, seg_dir: Path, name: str, W: int, H: int, below_number: bool) -> str:
    """Card text in Indic scripts drawn by libass: centred, large, white (or under the statistic)."""
    import textwrap
    landscape = W >= H
    lines = textwrap.wrap(text, 30 if landscape else 18)[:4]
    path = seg_dir / f"{name}_cardtext.srt"
    path.write_text("1\n00:00:00,000 --> 00:59:59,000\n" + "\n".join(lines) + "\n", encoding="utf-8")
    size = (18 if landscape else 13) if not below_number else (12 if landscape else 9)
    align, margin = (10, 0) if not below_number else (2, 95 if landscape else 160)   # SSA: 10 middle, 2 bottom
    style = (f"FontName=Nirmala UI,FontSize={size},Bold=1,PrimaryColour=&H00FFFFFF,BorderStyle=1,Outline=0,"
             f"Shadow=0,Alignment={align},MarginV={margin},MarginL=40,MarginR=40")
    return f",subtitles={path.name}:force_style='{style}'"


def _ken_burns(cand, W, H, frames, seg_dir, name, norm, sub, enc, out, title=None):
    """Gentle push-in for cards (and fallback for photos) so the shot is never frozen."""
    s = get_settings()
    img = cand.local_path
    card_text = ""
    if cand.kind == "card":
        head, body = cand.score_detail.get("headline", ""), cand.score_detail.get("body", "")
        kind = cand.score_detail.get("card_kind", "stat")
        # Pillow cannot shape Tamil/Hindi/... (vowel signs land in the wrong place): draw the card without that
        # text and let libass (HarfBuzz shaping, same engine as the captions) draw it on top
        shaped_head, shaped_body = needs_shaping(head), needs_shaping(body)
        img = render_card(seg_dir / f"{name}_card.png", W, H, "" if shaped_head else head,
                          "" if shaped_body else body, kind)
        if shaped_head or shaped_body:
            card_text = shaped_card_text(head if shaped_head else body, seg_dir, name, W, H,
                                         below_number=(kind == "stat" and not shaped_head))
    from PIL import Image
    with Image.open(img) as im:
        sw, sh = im.size
    cw, ch = _cover(sw, sh, W, H)
    x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
    if cand.kind == "card":
        # cards: slow sideways drift on a 4%-larger card. crop's x is evaluated per frame, so this costs almost
        # nothing (no per-frame resampling like zoompan) and the shot is never frozen
        pw, ph = (W * 104 // 100) // 2 * 2, (H * 104 // 100) // 2 * 2
        dur = max(frames / s.fps, 0.1)
        vf = (f"scale={pw}:{ph},crop={W}:{H}:x='({pw}-{W})*t/{dur:.3f}':y={(ph - H) // 2},"
              f"fps={s.fps},{norm},setsar=1,format=yuv420p{card_text}{sub}")
    else:
        zoom = 0.06
        vf = (f"scale={cw}:{ch},crop={W}:{H}:{x}:{(ch - H) // 2},"
              f"scale={W * 3 // 2}:{H * 3 // 2},zoompan=z='1+{zoom}*on/{max(frames, 1)}':x='iw/2-(iw/zoom/2)':"
              f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={s.fps},{norm},setsar=1,format=yuv420p{sub}")
    media.ffmpeg("-loop", "1", "-framerate", str(s.fps), "-i", Path(img).resolve(), "-vf", with_title(vf, title),
                 *enc, out.name, cwd=seg_dir)
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
    eq = f",eq=brightness={b:.3f}:saturation=1.05" if abs(b) >= 0.02 else ""   # skip when already balanced
    graph = [f"[0:v]fps={s.fps}{eq},split={n}" + "".join(f"[i{k}]" for k in range(n))]
    outs, maps = {}, []
    for k, (ratio, seg_dir) in enumerate(jobs):
        W, H = RATIOS[ratio]
        cw, ch = _cover(sw, sh, W, H)
        x = int(min(max(cand.subject_x * cw - W / 2, 0), cw - W))
        sub = ""
        if cues:
            srt = write_srt(cues, seg_dir / f"{name}.srt", offset=scene.start)
            sub = f",subtitles={seg_dir.name}/{srt.name}:force_style='{_caption_style(W, H, getattr(scene, 'caption_style', None))}'"
        title = title_png(scene, seg_dir, name, W, H)
        graph.append(with_title(f"[i{k}]{crop_then_scale(sw, sh, W, H, cand.subject_x)},setsar=1,format=yuv420p{sub}",
                                title, rel=f"{seg_dir.name}/", label=f"o{k}"))
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
                     f"[1:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-16:TP=-2:LRA=11,volume={s.music_volume_db}dB,"
                     f"afade=t=out:st={max(0, total - 2.5):.2f}:d=2.5[m];"
                     f"[m][v1]sidechaincompress=threshold=0.05:ratio=3:attack=20:release=500[duck];"
                     f"[v2][duck]amix=inputs=2:duration=first:normalize=0[a]",
                     "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", f"{total:.3f}", out)
    else:
        media.ffmpeg("-i", narration_wav, "-af", f"afade=t=out:st={max(0, total - 0.25):.3f}:d=0.25",
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", f"{total:.3f}", out)
    return out


def concat_and_mux(segments: list[Path], audio: Path, total: float, out: Path) -> Path:
    """Joins the segments and the pre-mixed audio with stream copy only (fast, no re-encoding)."""
    work = out.parent
    lst = work / f"{out.stem}_list.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8")
    media.ffmpeg("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-map", "0:v", "-map", "1:a",
                 "-c", "copy", "-t", f"{total:.3f}", "-movflags", "+faststart", out)
    return out
