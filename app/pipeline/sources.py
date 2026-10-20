"""Footage sources behind small adapters, plus a per-job circuit breaker.

Real-footage sources: Pixabay, Coverr, Pexels, NASA, Wikimedia Commons.
Generated sources: Cloudflare Workers AI (FLUX.1-schnell, Apache-2.0) and code-rendered cards.
Each candidate carries its licence, attribution and link so timeline.json is licence-complete.
"""
import base64
import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.logs import get_logger
from app.pipeline import media

log = get_logger("sources")
UA = {"User-Agent": "incubrix-broll-engine/1.0 (+https://github.com/incubrix-broll-engine)"}


@dataclass
class Candidate:
    id: str
    provider: str
    kind: str                      # video | image | card | ai_image
    title: str
    page_url: str
    download_url: str
    thumb_url: str
    licence: str
    licence_url: str
    attribution: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    ai_generated: bool = False
    tags: str = ""
    query: str = ""
    score: float = 0.0
    score_detail: dict = field(default_factory=dict)
    local_path: Path | None = None
    thumb_path: Path | None = None
    subject_x: float = 0.5          # horizontal position of main subject (0..1), for reframing
    start_at: float = 0.0           # seconds into the source to start from (skip title cards of long films)
    brightness: float = 0.45


class ProviderDown(Exception):
    pass


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(headers=UA, timeout=timeout, follow_redirects=True)


# ------------------------------------------------------------- stock providers
class Pixabay:
    name = "pixabay"
    LICENCE = ("Pixabay Content License (free for commercial use, no attribution required)",
               "https://pixabay.com/service/license-summary/")

    def __init__(self, s):
        self.key, self.timeout, self.n = s.pixabay_api_key, s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return bool(self.key)

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://pixabay.com/api/videos/", params={"key": self.key, "q": query[:100],
                                                                  "per_page": max(3, self.n), "safesearch": "true"})
            r.raise_for_status()
            hits = r.json().get("hits", [])
        out = []
        for h in hits:
            v = h.get("videos", {})
            # 720p "medium" renditions: small downloads, upscaled to 1080p by the renderer
            pick = v.get("medium") if v.get("medium", {}).get("url") else v.get("small")
            if not pick or not pick.get("url"):
                continue
            out.append(Candidate(id=f"pixabay:{h['id']}", provider=self.name, kind="video", title=h.get("tags", ""),
                                 page_url=h.get("pageURL", ""), download_url=pick["url"],
                                 thumb_url=pick.get("thumbnail") or v.get("medium", {}).get("thumbnail", ""),
                                 licence=self.LICENCE[0], licence_url=self.LICENCE[1],
                                 attribution=f"Video by {h.get('user', 'unknown')} on Pixabay",
                                 duration=float(h.get("duration", 0)), width=pick.get("width", 0),
                                 height=pick.get("height", 0), tags=h.get("tags", ""), query=query))
        return out


class Coverr:
    name = "coverr"
    LICENCE = ("Coverr License (free for commercial use, no attribution required)", "https://coverr.co/license")

    def __init__(self, s):
        self.key, self.timeout, self.n = s.coverr_api_key, s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return bool(self.key)

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://api.coverr.co/videos", params={"query": query[:100], "page_size": self.n, "urls": "true"},
                      headers={"Authorization": f"Bearer {self.key}"})
            r.raise_for_status()
            hits = r.json().get("hits", [])
        out = []
        for h in hits:
            if h.get("is_premium"):
                continue
            urls = h.get("urls") or {}
            if not urls.get("mp4"):
                continue
            out.append(Candidate(id=f"coverr:{h['id']}", provider=self.name, kind="video", title=h.get("title", ""),
                                 page_url=f"https://coverr.co/videos/{h.get('slug', h['id'])}", download_url=urls["mp4"],
                                 thumb_url=h.get("thumbnail", ""), licence=self.LICENCE[0], licence_url=self.LICENCE[1],
                                 attribution="Video from Coverr", duration=float(h.get("duration") or 0),
                                 width=int(h.get("max_width") or 0), height=int(h.get("max_height") or 0),
                                 ai_generated=bool(h.get("is_ai_generated")),
                                 tags=" ".join(h.get("tags") or []) + " " + (h.get("description") or ""), query=query))
        return out


class Pexels:
    name = "pexels"
    LICENCE = ("Pexels License (free for commercial use, no attribution required)", "https://www.pexels.com/license/")

    def __init__(self, s):
        self.key, self.timeout, self.n = s.pexels_api_key, s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return bool(self.key)

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://api.pexels.com/videos/search", params={"query": query, "per_page": self.n},
                      headers={"Authorization": self.key})
            r.raise_for_status()
            vids = r.json().get("videos", [])
        out = []
        for v in vids:
            files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and (f.get("height") or 0) <= 1080]
            if not files:
                continue
            f = max(files, key=lambda f: f.get("height") or 0)
            out.append(Candidate(id=f"pexels:{v['id']}", provider=self.name, kind="video", title=v.get("url", ""),
                                 page_url=v.get("url", ""), download_url=f["link"], thumb_url=v.get("image", ""),
                                 licence=self.LICENCE[0], licence_url=self.LICENCE[1],
                                 attribution=f"Video by {v.get('user', {}).get('name', 'unknown')} on Pexels",
                                 duration=float(v.get("duration", 0)), width=f.get("width", 0), height=f.get("height", 0),
                                 tags=v.get("url", "").rsplit("/", 2)[-2].replace("-", " "), query=query))
        return out


class Nasa:
    name = "nasa"
    LICENCE = ("NASA media (generally not subject to copyright in the US; NASA Media Usage Guidelines)",
               "https://www.nasa.gov/nasa-brand-center/images-and-media/")

    def __init__(self, s):
        self.timeout, self.n = s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return True

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://images-api.nasa.gov/search", params={"q": query, "media_type": "video"})
            r.raise_for_status()
            items = r.json().get("collection", {}).get("items", [])[: self.n]
        out = []
        for it in items:
            d = (it.get("data") or [{}])[0]
            # skip anything credited to a third party (not NASA-produced, may be copyrighted)
            credit = (d.get("secondary_creator") or "") + " " + (d.get("photographer") or "")
            if re.search(r"©|copyright|getty|reuters|ap photo", credit, re.I):
                continue
            thumb = next((l.get("href") for l in it.get("links", []) if l.get("render") == "image"), "")
            out.append(Candidate(id=f"nasa:{d.get('nasa_id')}", provider=self.name, kind="video", title=d.get("title", ""),
                                 page_url=f"https://images.nasa.gov/details/{quote(d.get('nasa_id', ''))}",
                                 download_url=it.get("href", ""), thumb_url=thumb, licence=self.LICENCE[0],
                                 licence_url=self.LICENCE[1], attribution=f"NASA ({d.get('center', 'NASA')})",
                                 tags=" ".join(d.get("keywords") or []) + " " + d.get("title", ""), query=query))
        return out

    def resolve(self, cand: Candidate) -> str:
        """NASA hrefs point to an asset list; choose a moderately sized mp4."""
        with _client(self.timeout) as c:
            assets = c.get(cand.download_url.replace(" ", "%20")).json()
        mp4s = [a for a in assets if a.lower().endswith(".mp4")]
        for pref in ("~mobile.mp4", "~medium.mp4", "~small.mp4", "~preview.mp4", "~orig.mp4"):
            for a in mp4s:
                if a.endswith(pref):
                    return a.replace(" ", "%20").replace("http://", "https://")
        if not mp4s:
            raise ProviderDown("no mp4 in NASA asset")
        return mp4s[0].replace(" ", "%20")


class Wikimedia:
    name = "wikimedia"
    OK = re.compile(r"^(cc0|public domain|pd|cc by \d|cc-by-\d|cc by-?\d|attribution)", re.I)

    def __init__(self, s):
        self.timeout, self.n = s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return True

    def search(self, query: str) -> list[Candidate]:
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} filetype:video",
                  "gsrnamespace": 6, "gsrlimit": self.n, "prop": "imageinfo",
                  "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 640}
        with _client(self.timeout) as c:
            r = c.get("https://commons.wikimedia.org/w/api.php", params=params)
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages", {}).values()
        out = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            md = ii.get("extmetadata", {})
            lic = re.sub("<[^>]+>", "", md.get("LicenseShortName", {}).get("value", ""))
            # only licences that allow commercial use without share-alike obligations on our output
            if not self.OK.match(lic.strip()) or "nc" in lic.lower() or "sa" in lic.lower().split():
                continue
            if "-sa" in lic.lower() or " sa" in lic.lower():
                continue
            artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "unknown"))[:120]
            out.append(Candidate(id=f"wikimedia:{p.get('pageid')}", provider=self.name, kind="video",
                                 title=p.get("title", ""), page_url=ii.get("descriptionurl", ""),
                                 download_url=ii.get("url", ""), thumb_url=ii.get("thumburl", ""),
                                 licence=lic, licence_url=md.get("LicenseUrl", {}).get("value", ""),
                                 attribution=f"{artist} via Wikimedia Commons ({lic})",
                                 duration=float(ii.get("duration") or 0), width=ii.get("width", 0),
                                 height=ii.get("height", 0), tags=p.get("title", ""), query=query))
        return out


class InternetArchive:
    """Prelinger Archives (archive.org): historical/vintage film. Only items whose licence URL says public domain."""
    name = "internet_archive"

    def __init__(self, s):
        self.timeout, self.n = s.provider_timeout_seconds, s.candidates_per_query

    def available(self):
        return True

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://archive.org/advancedsearch.php", params={
                "q": f"collection:prelinger AND mediatype:movies AND ({query})", "rows": self.n * 2, "output": "json",
                "fl[]": ["identifier", "title", "licenseurl", "runtime", "description"]})
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])
        out = []
        for d in docs:
            lic = d.get("licenseurl") or ""
            if "publicdomain" not in lic:          # unknown or non-PD rights -> never used
                continue
            ident = d["identifier"]
            runtime = 0.0
            try:
                for x in [float(v) for v in str(d.get("runtime", "0")).split(":")]:
                    runtime = runtime * 60 + x
            except ValueError:
                pass
            title = d.get("title") or ident
            if isinstance(title, list):
                title = title[0]
            desc = d.get("description") or ""
            if isinstance(desc, list):
                desc = " ".join(desc)
            out.append(Candidate(id=f"internet_archive:{ident}", provider=self.name, kind="video", title=str(title),
                                 page_url=f"https://archive.org/details/{ident}",
                                 download_url=f"https://archive.org/download/{ident}/{ident}_512kb.mp4",
                                 thumb_url=f"https://archive.org/services/img/{ident}",
                                 licence="Public domain (Prelinger Archives, Internet Archive)", licence_url=lic,
                                 attribution=f'"{title}" - Prelinger Archives via Internet Archive (public domain)',
                                 duration=runtime, start_at=round(runtime / 3, 1) if runtime > 60 else 0.0,
                                 tags=f"{title} {desc[:200]}", query=query))
            if len(out) >= self.n:
                break
        return out


class OpenverseImages:
    """Openverse photo search, restricted to CC0 / Public Domain Mark / CC BY. Animated by the renderer."""
    name = "openverse_images"
    NAMES = {"cc0": "CC0", "pdm": "Public Domain Mark", "by": "CC BY"}

    def __init__(self, s):
        self.timeout, self.n = min(8.0, s.provider_timeout_seconds), s.candidates_per_query  # slow API: cap wait

    def available(self):
        return True

    def search(self, query: str) -> list[Candidate]:
        with _client(self.timeout) as c:
            r = c.get("https://api.openverse.org/v1/images/", params={
                "q": query[:100], "license": "cc0,pdm,by", "page_size": self.n, "mature": "false"})
            r.raise_for_status()
            results = r.json().get("results", [])
        out = []
        for x in results:
            lic = (x.get("license") or "").lower()
            if lic not in self.NAMES or (x.get("width") or 0) < 1000:
                continue
            name = f"{self.NAMES[lic]} {x.get('license_version') or ''}".rstrip()
            title = x.get("title") or "Photo"
            tags = " ".join(t.get("name", "") for t in (x.get("tags") or [])[:15])
            out.append(Candidate(id=f"openverse:{x['id']}", provider=self.name, kind="image", title=title,
                                 page_url=x.get("foreign_landing_url") or "", download_url=x.get("url") or "",
                                 thumb_url=x.get("thumbnail") or "", licence=name, licence_url=x.get("license_url") or "",
                                 attribution=f'"{title}" by {x.get("creator") or "unknown"} ({name}) via Openverse/{x.get("source")}',
                                 width=x.get("width") or 0, height=x.get("height") or 0, tags=f"{tags} {title}", query=query))
        return out


class MockFootage:
    """Synthesises small coloured test clips locally - lets the full pipeline run with no network or keys."""
    name = "mock"

    def __init__(self, s):
        self.n = 3

    def available(self):
        return True

    def search(self, query: str) -> list[Candidate]:
        h = hashlib.md5(query.encode()).hexdigest()
        return [Candidate(id=f"mock:{h[:8]}{i}", provider=self.name, kind="video", title=f"{query} {i}",
                          page_url="mock://", download_url=f"mock://{h[:6]}{i}", thumb_url="", licence="CC0 (synthetic test clip)",
                          licence_url="https://creativecommons.org/publicdomain/zero/1.0/", attribution="synthetic",
                          duration=10, width=1280, height=720, tags=query, query=query) for i in range(self.n)]


PROVIDERS = {"pixabay": Pixabay, "coverr": Coverr, "pexels": Pexels, "nasa": Nasa, "wikimedia": Wikimedia,
             "internet_archive": InternetArchive, "openverse_images": OpenverseImages, "mock": MockFootage}

# which providers suit which scene category (smart routing instead of a fixed order)
CATEGORY_SOURCES = {
    "space_science": ["nasa", "pixabay", "coverr", "pexels", "mock"],
    "place": ["pixabay", "wikimedia", "coverr", "pexels", "openverse_images", "mock"],
    "nature": ["pixabay", "coverr", "pexels", "wikimedia", "openverse_images", "mock"],
    "history": ["internet_archive", "wikimedia", "pixabay", "openverse_images", "mock"],
}
DEFAULT_SOURCES = ["pixabay", "coverr", "pexels", "openverse_images", "mock"]


class CircuitBreaker:
    """Per job: a provider that fails twice is skipped for the rest of the job (with a warning)."""

    def __init__(self, warnings: list[str]):
        self.fail: dict[str, int] = {}
        self.warnings = warnings
        self.lock = threading.Lock()

    def is_open(self, name: str) -> bool:
        return self.fail.get(name, 0) >= 2

    def record(self, name: str, err: Exception):
        with self.lock:
            self.fail[name] = self.fail.get(name, 0) + 1
            if self.fail[name] == 2:
                self.warnings.append(f"PROVIDER_DOWN: '{name}' failed repeatedly ({type(err).__name__}); "
                                     "continuing with other sources")
            log.warning("provider %s failed: %s", name, err)


def active_providers(category: str) -> list:
    s = get_settings()
    enabled = s.provider_list()
    order = CATEGORY_SOURCES.get(category, DEFAULT_SOURCES)
    out = []
    for name in order:
        if name not in enabled or s.is_disabled(name):
            continue
        p = PROVIDERS[name](s)
        if p.available():
            out.append(p)
    return out


_search_cache: dict[tuple[str, str], tuple[float, list]] = {}
_cache_lock = threading.Lock()
SEARCH_CACHE_SECONDS = 3600


def search_all(queries: list[str], category: str, breaker: CircuitBreaker, limit_queries: int = 3) -> list[Candidate]:
    results: list[Candidate] = []
    providers = active_providers(category)
    from concurrent.futures import ThreadPoolExecutor

    def one(p, q):
        if breaker.is_open(p.name):
            return []
        if get_settings().is_disabled(p.name):  # live switch-off mid job
            raise ProviderDown(f"{p.name} disabled")
        import copy
        import time as _t
        key = (p.name, q.lower().strip())
        with _cache_lock:
            hit = _search_cache.get(key)
        if hit and _t.time() - hit[0] < SEARCH_CACHE_SECONDS:   # same query again: no API call
            return copy.deepcopy(hit[1])
        res = p.search(q)
        with _cache_lock:
            _search_cache[key] = (_t.time(), copy.deepcopy(res))
        return res

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, p, q): (p, q) for q in queries[:limit_queries] for p in providers}
        for f, (p, q) in futs.items():
            try:
                results.extend(f.result())
            except Exception as e:
                breaker.record(p.name, e)
    seen, uniq = set(), []
    for c in results:
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq


# ------------------------------------------------------------- downloads
def download(cand: Candidate, dest_dir: Path, max_mb: int = 80) -> Path:
    """Download and verify the clip decodes (catches corrupted downloads)."""
    s = get_settings()
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", cand.id)
    if cand.provider == "mock":
        path = dest_dir / f"{safe}.mp4"
        if not path.exists():
            hue = int(hashlib.md5(cand.id.encode()).hexdigest()[:2], 16) * 360 // 255
            media.ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration=10",
                         "-vf", f"hue=h={hue}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path)
        return path
    url = Nasa(s).resolve(cand) if cand.provider == "nasa" else cand.download_url
    ext = ".jpg" if cand.kind in ("image", "ai_image") else Path(url.split("?")[0]).suffix or ".mp4"
    path = dest_dir / f"{safe}{ext}"
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    with httpx.Client(headers=UA, timeout=httpx.Timeout(60, connect=s.provider_timeout_seconds),
                      follow_redirects=True) as c, c.stream("GET", url) as r:
        r.raise_for_status()
        size = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 16):
                size += len(chunk)
                if size > max_mb * 1024 * 1024:
                    raise ProviderDown("clip too large")
                f.write(chunk)
    if cand.kind == "image":   # verify, and shrink huge originals (Openverse can return 6000-px photos)
        from PIL import Image
        try:
            with Image.open(tmp) as im:
                if max(im.size) < 1280:      # too small for a 1080p frame
                    raise ValueError("image below 1280 px")
                im = im.convert("RGB")
                im.thumbnail((2560, 2560))
                im.save(path, format="JPEG", quality=92)
        except Exception:
            raise ProviderDown("downloaded image is corrupt or too small")
        finally:
            tmp.unlink(missing_ok=True)
        return path
    vs = media.video_stream(tmp)  # raises if corrupt
    if not vs:
        tmp.unlink(missing_ok=True)
        raise ProviderDown("downloaded file has no video stream")
    tmp.replace(path)
    return path


def download_part(cand: Candidate, dest_dir: Path, seconds: float) -> Path:
    """Fetch only the first `seconds` of a clip: ffmpeg reads the URL with HTTP range requests and stops early,
    so a 4-second scene no longer downloads a whole 30-second 1080p file. Falls back to a full download."""
    s = get_settings()
    if cand.provider == "mock" or cand.kind == "image":
        return download(cand, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", cand.id)
    need = math.ceil(seconds + 1.5)   # renderer may skip up to 1 s of intro
    path = dest_dir / f"{safe}_{need}s.mp4"
    if path.exists() and path.stat().st_size > 0:
        return path
    full = dest_dir / f"{safe}.mp4"
    if full.exists() and full.stat().st_size > 0:
        return full
    try:
        url = Nasa(s).resolve(cand) if cand.provider == "nasa" else cand.download_url
        tmp = path.with_suffix(".part.mp4")
        media.ffmpeg("-user_agent", UA["User-Agent"], "-rw_timeout", str(int(s.provider_timeout_seconds * 1e6)),
                     *(["-ss", str(cand.start_at)] if cand.start_at else []), "-t", str(need), "-i", url, "-map", "0:v:0", "-c", "copy", "-an", "-movflags", "+faststart", tmp,
                     timeout=120)
        vs = media.video_stream(tmp)
        if not vs or media.duration(tmp) < min(seconds, (cand.duration or seconds)) * 0.8:
            raise ProviderDown("partial download too short")
        tmp.replace(path)
        return path
    except Exception as e:
        log.info("partial download failed for %s (%s); full download", cand.id, type(e).__name__)
        return download(cand, dest_dir)


def fetch_thumb(cand: Candidate, dest_dir: Path) -> Path | None:
    if not cand.thumb_url:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / (re.sub(r"[^A-Za-z0-9_-]", "_", cand.id) + ".jpg")
    if path.exists():
        return path
    try:
        with _client(8) as c:
            r = c.get(cand.thumb_url.replace(" ", "%20"))
            r.raise_for_status()
            path.write_bytes(r.content)
        return path
    except Exception:
        return None


# ------------------------------------------------------------- AI image (Cloudflare Workers AI)
class CloudflareFlux:
    name = "cloudflare-flux"
    LICENCE = ("AI-generated with FLUX.1-schnell (Apache-2.0) on Cloudflare Workers AI; output usable commercially",
               "https://huggingface.co/black-forest-labs/FLUX.1-schnell")

    def __init__(self):
        s = get_settings()
        self.token, self.account, self.model = s.cloudflare_api_token, s.cloudflare_account_id, s.cloudflare_model

    def available(self) -> bool:
        s = get_settings()
        return bool(self.token and self.account) and s.ai_image_provider == "cloudflare" and not s.is_disabled("cloudflare")

    def generate(self, prompt: str, dest: Path, seed: int) -> Candidate:
        safe_prompt = (f"{prompt}. Photorealistic cinematic b-roll still, natural light, 16:9 composition. "
                       "No people's faces, no text, no letters, no logos, no watermarks.")
        with _client(90) as c:
            r = c.post(f"https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/run/{self.model}",
                       headers={"Authorization": f"Bearer {self.token}"},
                       json={"prompt": safe_prompt, "steps": 6, "seed": seed})
            r.raise_for_status()
            img = base64.b64decode(r.json()["result"]["image"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(img)
        h = hashlib.md5(prompt.encode()).hexdigest()[:10]
        return Candidate(id=f"ai:{h}", provider=self.name, kind="ai_image", title=prompt[:120], page_url="",
                         download_url="", thumb_url="", licence=self.LICENCE[0], licence_url=self.LICENCE[1],
                         attribution="AI-generated image (FLUX.1-schnell)", ai_generated=True, tags=prompt,
                         local_path=dest, thumb_path=dest, width=1024, height=768, query=prompt)
