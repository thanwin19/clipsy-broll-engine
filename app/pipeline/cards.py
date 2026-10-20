"""Code-rendered scene cards for statistics and abstract lines. Licence-clean by construction."""
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
NAVY, NAVY2, TEAL, WHITE, MUTED = (11, 31, 58), (20, 56, 96), (20, 184, 166), (245, 248, 252), (170, 190, 210)


def _font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _background(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), NAVY)
    d = ImageDraw.Draw(img)
    for y in range(h):  # vertical gradient
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(NAVY[i] + (NAVY2[i] - NAVY[i]) * t) for i in range(3)))
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    r = int(min(w, h) * 0.45)
    gd.ellipse([w * 0.7 - r, h * 0.25 - r, w * 0.7 + r, h * 0.25 + r], fill=(10, 90, 100))
    glow = glow.filter(ImageFilter.GaussianBlur(r // 2))
    return Image.blend(img, Image.composite(glow, img, glow.convert("L")), 0.35)


def _center(d, text, font, y, w, fill):
    tw = d.textbbox((0, 0), text, font=font)[2]
    d.text(((w - tw) / 2, y), text, font=font, fill=fill)


def render_card(path: Path, w: int, h: int, headline: str, body: str = "", kind: str = "stat") -> Path:
    img = _background(w, h)
    d = ImageDraw.Draw(img)
    unit = min(w, h)
    if kind == "stat" and headline:
        big = _font(int(unit * 0.26))
        bb = d.textbbox((0, 0), headline, font=big)
        gh = bb[3] - bb[1]                       # glyph height
        top = h * 0.42 - gh / 2 - bb[1]          # draw so the glyphs are centred at 42% height
        _center(d, headline, big, top, w, TEAL)
        line_y = top + bb[3] + unit * 0.04       # bar sits below the glyphs, never through them
        bar_w = int(w * 0.18)
        d.rounded_rectangle([(w - bar_w) / 2, line_y, (w + bar_w) / 2, line_y + unit * 0.012], radius=6, fill=WHITE)
        f = _font(int(unit * 0.055))
        chars = max(14, int(w / (unit * 0.032)))
        for i, line in enumerate(textwrap.wrap(body, chars)[:3]):
            _center(d, line, f, line_y + unit * 0.05 + i * unit * 0.075, w, WHITE)
    else:
        f = _font(int(unit * 0.075))
        chars = max(12, int(w / (unit * 0.045)))
        lines = textwrap.wrap(headline, chars)[:5]
        total = len(lines) * unit * 0.1
        d.rectangle([w * 0.1, h / 2 - total / 2 - unit * 0.06, w * 0.1 + unit * 0.012, h / 2 + total / 2], fill=TEAL)
        for i, line in enumerate(lines):
            d.text((w * 0.1 + unit * 0.05, h / 2 - total / 2 + i * unit * 0.1), line, font=f, fill=WHITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)
    return path
