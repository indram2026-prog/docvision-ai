"""Image conditioning: contrast normalisation + projection-profile deskew."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def to_rgb(img) -> np.ndarray:
    return np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else img


def normalize_contrast(rgb: np.ndarray) -> np.ndarray:
    """Linear stretch so page background -> 255, darkest ink -> ~0. Fixes washed-out scans."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray, 0.5), np.percentile(gray, 99)
    if hi - lo < 20 or (hi > 245 and lo < 40):
        return rgb
    out = (rgb.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_skew(gray: np.ndarray, max_deg: float = 4.0) -> float:
    """Angle (deg, CCW positive) that maximises row-profile variance. Coarse-to-fine search."""
    scale = 800.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = (small < 160).astype(np.float32)
    if ink.sum() < 50:
        return 0.0
    h, w = ink.shape
    c = (w / 2, h / 2)

    def score(a):
        m = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR)
        return float(np.var(r.sum(1)))
    coarse = np.arange(-max_deg, max_deg + 1e-6, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.1)
    return float(max(fine, key=score))


def rotation_matrix(shape, angle):
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)


def deskew(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ang = estimate_skew(gray)
    if abs(ang) < 0.3:
        return rgb, 0.0
    m = rotation_matrix(rgb.shape, ang)
    out = cv2.warpAffine(rgb, m, (rgb.shape[1], rgb.shape[0]), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, ang


def map_bbox_back(bbox, shape, angle):
    """Map a bbox from the deskewed frame to the original image frame (axis-aligned hull)."""
    if abs(angle) < 1e-6:
        return tuple(int(v) for v in bbox)
    inv = cv2.invertAffineTransform(rotation_matrix(shape, angle))
    x0, y0, x1, y1 = bbox
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float64)
    q = pts @ inv.T
    return (int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max()))


def prepare(img):
    """PIL/ndarray -> (conditioned rgb, original rgb, skew_deg)."""
    rgb = to_rgb(img)
    norm = normalize_contrast(rgb)
    desk, ang = deskew(norm)
    return desk, rgb, ang
