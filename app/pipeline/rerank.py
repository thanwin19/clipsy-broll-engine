"""Look-then-pick: score candidate frames against the scene with CLIP, apply Safety Guard penalties,
and measure subject position / brightness used later for reframing and colour normalisation."""
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings
from app.logs import get_logger
from app.pipeline.sources import Candidate

log = get_logger("rerank")

FACE_WORDS = re.compile(r"face|people|person|human", re.I)


class ClipScorer:
    name = "open_clip"
    _model = None
    _lock = threading.Lock()

    def __init__(self):
        s = get_settings()
        self.arch, self.pretrained = s.clip_model, s.clip_pretrained
        self.version = f"open_clip/{self.arch}/{self.pretrained}"

    def _load(self):
        with ClipScorer._lock:
            if ClipScorer._model is None:
                import open_clip
                import torch
                import os
                torch.set_num_threads(max(1, os.cpu_count() or 1))
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.arch, pretrained=self.pretrained, cache_dir=str(get_settings().data_dir / "models"))
                model.eval()
                ClipScorer._model = (model, preprocess, open_clip.get_tokenizer(self.arch), torch)
        return ClipScorer._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        model, _, tok, torch = self._load()
        with ClipScorer._lock, torch.inference_mode():
            t = model.encode_text(tok(texts))
            t = t / t.norm(dim=-1, keepdim=True)
        return t.numpy()

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        model, preprocess, _, torch = self._load()
        imgs = [preprocess(Image.open(p).convert("RGB")) for p in paths]
        with ClipScorer._lock, torch.inference_mode():
            v = model.encode_image(torch.stack(imgs))
            v = v / v.norm(dim=-1, keepdim=True)
        return v.numpy()


class MockScorer:
    """Token overlap between scene queries and candidate tags (no model download needed)."""
    name = "mock-scorer"
    version = "mock/1"


def get_scorer():
    return MockScorer() if get_settings().rerank_provider == "mock" else ClipScorer()


_face_local = threading.local()  # CascadeClassifier is not thread-safe: one instance per worker thread


def faces_in(path: Path) -> int:
    """Number of frontal faces in an image. On any detector error returns 1 (fail safe: treat as a face)."""
    import cv2
    try:
        face = getattr(_face_local, "clf", None)
        if face is None:
            face = _face_local.clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape[:2]) < 48:
            return 0
        side = max(24, img.shape[1] // 20)
        return len(face.detectMultiScale(img, scaleFactor=1.15, minNeighbors=6, minSize=(side, side)))
    except Exception as e:
        log.warning("face detection failed on %s: %s", path.name, e)
        return 1


def subject_and_brightness(path: Path) -> tuple[float, float]:
    """Horizontal centre of visual saliency (0..1) and mean luma (0..1)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return 0.5, 0.45
    small = cv2.resize(img, (128, max(1, int(128 * img.shape[0] / img.shape[1]))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # spectral-residual saliency (Hou & Zhang 2007), implemented directly with numpy
    f = np.fft.fft2(gray)
    log_amp = np.log(np.abs(f) + 1e-6)
    phase = np.angle(f)
    resid = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)
    col = sal.sum(axis=0)
    x = float((col * np.arange(len(col))).sum() / (col.sum() + 1e-9) / max(1, len(col) - 1))
    return min(max(x, 0.0), 1.0), float(gray.mean() / 255.0)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", s.lower())}


def score_candidates(scene, cands: list[Candidate], avoid_terms: list[str], avoid_faces: bool,
                     scorer, thumb_dir: Path) -> list[Candidate]:
    """Fills cand.score (0..1-ish CLIP cosine) and drops unsafe candidates."""
    from app.pipeline.sources import fetch_thumb
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=16) as ex:
        thumbs = list(ex.map(lambda c: c.thumb_path or fetch_thumb(c, thumb_dir), cands))
    for c, t in zip(cands, thumbs):
        c.thumb_path = t

    avoid_tokens = set().union(*[_tokens(a) for a in avoid_terms]) - {"people", "faces", "face", "people's"} if avoid_terms else set()
    kept = []
    for c in cands:
        tag_toks = _tokens(c.tags + " " + c.title)
        hit = tag_toks & avoid_tokens
        if hit:
            c.score_detail["rejected"] = f"avoid list match in tags: {sorted(hit)[:3]}"
            continue
        if c.ai_generated and not scene.ai_allowed:
            c.score_detail["rejected"] = "AI-generated footage not allowed for this scene (person/brand/sensitive)"
            continue
        if c.thumb_path and avoid_faces and faces_in(c.thumb_path) > 0:
            c.score_detail["rejected"] = "faces detected (avoid: people's faces)"
            continue
        kept.append(c)

    if not kept:
        return []
    if isinstance(scorer, MockScorer):
        q = _tokens(" ".join(scene.queries) + " " + scene.visual)
        for c in kept:
            ov = len(q & _tokens(c.tags + " " + c.title + " " + c.query)) / (len(q) or 1)
            c.score = round(0.18 + 0.2 * ov, 4)
            c.score_detail.update({"method": "token-overlap", "overlap": round(ov, 3)})
    else:
        with_img = [c for c in kept if c.thumb_path]
        for c in kept:
            if not c.thumb_path:
                c.score = 0.0
                c.score_detail["method"] = "no-thumbnail"
        if with_img:
            pos_texts = [scene.visual or scene.text, scene.text] + scene.queries[:2]
            neg_texts = [f"a photo of {n}" for n in (scene.negative + avoid_terms)[:6]]
            T = scorer.embed_texts(pos_texts + neg_texts)
            V = scorer.embed_images([c.thumb_path for c in with_img])
            sims = V @ T.T
            for c, row in zip(with_img, sims):
                pos = float(0.5 * row[0] + 0.25 * row[1] + 0.25 * max(row[2:len(pos_texts)], default=row[0]))
                neg = float(max(row[len(pos_texts):], default=0.0))
                penalty = max(0.0, neg - pos + 0.02) if neg_texts else 0.0
                c.score = round(pos - penalty, 4)
                c.score_detail.update({"method": "clip", "positive": round(pos, 4), "negative_max": round(neg, 4),
                                       "penalty": round(penalty, 4)})
    for c in kept:
        if c.thumb_path:
            c.subject_x, c.brightness = subject_and_brightness(c.thumb_path)
        if c.kind == "image":
            c.score = round(c.score - 0.015, 4)  # still photo: prefer real motion footage when scores are close
            c.score_detail["still_image_penalty"] = 0.015
        if c.kind == "video" and c.duration and c.duration < scene.duration:
            c.score = round(c.score - 0.01, 4)  # slight preference for clips long enough to avoid looping
            c.score_detail["short_clip_penalty"] = 0.01
    return sorted(kept, key=lambda c: c.score, reverse=True)
