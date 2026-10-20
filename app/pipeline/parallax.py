"""2.5D parallax for still images (AI images, photos): turns a still into a convincing camera move.

Depth-Anything-V2-Small (Apache-2.0, ONNX, CPU) estimates relative depth once per image. Each output frame
is then produced by remapping pixels: near pixels shift more than far ones while the camera slowly pushes
in, which gives the parallax a real dolly shot has. Frames are streamed straight into ffmpeg.
"""
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.errors import ApiError
from app.logs import get_logger
from app.pipeline import media

log = get_logger("parallax")
_session = None
_lock = threading.Lock()
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            d = get_settings().data_dir / "models" / "depth"
            path = d / "onnx" / "model.onnx"
            if not path.exists():
                from huggingface_hub import hf_hub_download
                hf_hub_download("onnx-community/depth-anything-v2-small", "onnx/model.onnx", local_dir=str(d))
            _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _session


def depth_map(img_bgr: np.ndarray, cache: Path | None = None) -> np.ndarray:
    """Relative depth in [0, 1] (1 = near), same size as the image."""
    if cache and cache.exists():
        return np.load(cache)
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (518, 518), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    sess = _model()
    with _lock:
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    d = np.squeeze(out).astype(np.float32)
    d = (d - d.min()) / max(1e-6, float(d.max() - d.min()))
    d = cv2.resize(d, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=max(2.0, img_bgr.shape[1] / 300))  # soft edges -> no tearing
    if cache:
        np.save(cache, d)
    return d


def render_parallax(img_path: Path, W: int, H: int, frames: int, subject_x: float, vf_tail: str,
                    enc: list, out_name: str, cwd: Path, fps: int) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ApiError("RENDER_FAILED", "cannot read image for parallax")
    depth = depth_map(img, img_path.with_suffix(".depth.npy"))
    # crop the image (and its depth) to the output aspect with a 12% margin for camera movement
    margin = 1.12
    ih, iw = img.shape[:2]
    scale = max(W * margin / iw, H * margin / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    depth = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_LINEAR)
    cw, ch = int(W * margin), int(H * margin)
    x0 = int(min(max(subject_x * nw - cw / 2, 0), nw - cw))
    y0 = (nh - ch) // 2
    img, depth = img[y0:y0 + ch, x0:x0 + cw], depth[y0:y0 + ch, x0:x0 + cw]

    cx, cy = cw / 2.0, ch / 2.0
    U, V = np.meshgrid(np.arange(W, dtype=np.float32) - W / 2.0, np.arange(H, dtype=np.float32) - H / 2.0)
    d_out = cv2.remap(depth, cx + U / 1.03, cy + V / 1.03, cv2.INTER_LINEAR) - 0.5  # depth seen through the frame
    amp_x, amp_y = 0.045 * W, 0.012 * H              # max parallax shift of the nearest pixels
    direction = -1.0 if subject_x > 0.5 else 1.0      # drift towards the subject

    cmd = [media.FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", f"setsar=1,format=yuv420p{vf_tail}", *enc, out_name]
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for k in range(frames):
            p = k / max(1, frames - 1)
            p = p * p * (3 - 2 * p)                     # ease in/out
            z = 1.0 + 0.06 * p                          # slow push-in
            shift = (p - 0.5) * 2 * direction
            map_x = cx + U / z - d_out * amp_x * shift
            map_y = cy + V / z - d_out * amp_y * shift
            frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait(timeout=300) != 0:
            raise ApiError("RENDER_FAILED", "parallax encode failed", {"stderr_tail": err[-600:]})
    finally:
        if proc.poll() is None:
            proc.kill()
