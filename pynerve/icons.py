"""Icon (template-image) fallback for non-text UI elements.

Text labels and accessibility trees cover most controls, but icon-only
buttons, canvas widgets, and game UIs have neither. This module adds a
third perception tier — used only when text/UIA fail::

    nv.click_image("assets/save-icon.png")          # move + left-click
    m = nv.find_image(template, threshold=0.92)     # MatchResult or raise

Implementation is stdlib + Pillow (no OpenCV/numpy): a coarse-to-fine
pyramid with early-exit SAD. Fullscreen 1080p searches take ~1-5s, so
constrain with ``region=(x, y, w, h)`` when possible. Unit-tested with
synthetic images; see ``tests/test_new_capabilities.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

logger = logging.getLogger("pynerve.icons")

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _RESAMPLE = Image.BILINEAR  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class ImageMatch:
    """Best template location: score 0..1 (1.0 = pixel-identical)."""

    center: tuple[float, float]
    bounds: tuple[float, float, float, float]  # left, top, right, bottom
    score: float

    @property
    def x(self) -> float:
        return self.center[0]

    @property
    def y(self) -> float:
        return self.center[1]


def load_template(template: Image.Image | str | Path) -> Image.Image:
    """Load a template as grayscale PIL image."""
    if isinstance(template, Image.Image):
        return template.convert("L")
    img = Image.open(template)
    img.load()
    return img.convert("L")


def match_template(
    screen: Image.Image,
    template: Image.Image | str | Path,
    threshold: float = 0.9,
    region: tuple[int, int, int, int] | None = None,
    coarse_stride: int = 3,
) -> ImageMatch | None:
    """Find ``template`` in ``screen``. Returns best match or None.

    Args:
        screen: Full screenshot (grayscale or RGB).
        template: Template image, path, or grayscale image.
        threshold: Minimum normalized score 0..1 (1.0 = identical).
        region: Optional (x, y, w, h) sub-region of ``screen`` to search;
            returned coordinates are still in screen space.
        coarse_stride: Coarse-pass pixel stride (higher = faster, coarser).
    """
    gray_screen = screen.convert("L")
    sw, sh = gray_screen.size
    ox, oy = 0, 0
    if region is not None:
        rx, ry, rw, rh = region
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, sw - rx)
        rh = min(rh, sh - ry)
        if rw <= 0 or rh <= 0:
            return None
        gray_screen = gray_screen.crop((rx, ry, rx + rw, ry + rh))
        sw, sh = gray_screen.size
        ox, oy = rx, ry

    tmpl = load_template(template)
    tw, th = tmpl.size
    if tw > sw or th > sh or tw < 2 or th < 2:
        return None

    # Pyramid: downscale so the coarse screen is at most ~480px wide.
    scale = 1
    while scale < 4 and sw // (scale * 2) >= 480 and tw // (scale * 2) >= 8:
        scale *= 2
    small_screen = gray_screen if scale == 1 else gray_screen.resize(
        (sw // scale, sh // scale), _RESAMPLE
    )
    small_tmpl = tmpl if scale == 1 else tmpl.resize((tw // scale, th // scale), _RESAMPLE)

    sb = small_screen.tobytes()
    tb = small_tmpl.tobytes()
    cw, ch = small_screen.size
    ctw, cth = small_tmpl.size
    n = ctw * cth

    # Coarse pass: strided SAD with early termination, keep top candidates.
    best_sad = 256 * n
    candidates: list[tuple[int, int, int]] = []  # (sad, x, y)
    for cy in range(0, ch - cth + 1, coarse_stride):
        row_base = cy * cw
        for cx in range(0, cw - ctw + 1, coarse_stride):
            sad = 0
            # Early exit once worse than the 8th-best candidate budget.
            budget = best_sad
            if len(candidates) >= 8:
                budget = candidates[-1][0]
            for ty in range(cth):
                s_off = row_base + ty * cw + cx
                t_off = ty * ctw
                for tx in range(ctw):
                    sad += abs(sb[s_off + tx] - tb[t_off + tx])
                    if sad >= budget:
                        break
                if sad >= budget:
                    break
            if sad < best_sad:
                best_sad = sad
            # Insert into bounded top-8 list.
            inserted = False
            for i, (s, _, _) in enumerate(candidates):
                if sad < s:
                    candidates.insert(i, (sad, cx, cy))
                    inserted = True
                    break
            if not inserted and len(candidates) < 8:
                candidates.append((sad, cx, cy))
            if len(candidates) > 8:
                candidates.pop()

    if not candidates:
        return None

    # Refine top candidates at full resolution in a ±2*scale window.
    full_sb = gray_screen.tobytes()
    full_tb = tmpl.tobytes()
    full_n = tw * th
    best: tuple[int, int, int] | None = None  # (sad, fx, fy)
    for _, ccx, ccy in candidates[:4]:
        fx0 = max(0, ccx * scale - 2 * scale)
        fy0 = max(0, ccy * scale - 2 * scale)
        fx1 = min(sw - tw, ccx * scale + 2 * scale)
        fy1 = min(sh - th, ccy * scale + 2 * scale)
        for fy in range(fy0, fy1 + 1):
            for fx in range(fx0, fx1 + 1):
                sad = 0
                budget = best[0] if best else 256 * full_n
                for ty in range(th):
                    s_off = (fy + ty) * sw + fx
                    t_off = ty * tw
                    for tx in range(tw):
                        sad += abs(full_sb[s_off + tx] - full_tb[t_off + tx])
                        if sad >= budget:
                            break
                    if sad >= budget:
                        break
                if best is None or sad < best[0]:
                    best = (sad, fx, fy)

    if best is None:
        return None
    sad, fx, fy = best
    score = 1.0 - sad / (255.0 * full_n)
    if score < threshold:
        logger.debug("Template best score %.3f below threshold %.3f", score, threshold)
        return None
    left, top = float(fx + ox), float(fy + oy)
    right, bottom = left + tw, top + th
    return ImageMatch(
        center=((left + right) / 2.0, (top + bottom) / 2.0),
        bounds=(left, top, right, bottom),
        score=round(score, 4),
    )
