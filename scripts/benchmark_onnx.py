#!/usr/bin/env python3
"""Benchmark: PP-OCRv5 mobile on ONNX Runtime vs the bundled MNN engine.

End-to-end OCR (detection + recognition) on synthetic text images, reporting
total and per-region latency for both backends on the same inputs.

The ONNX pipeline mirrors ocr-rs's preprocessing:
  det:  resize to max-side 960, pad to multiple of 32, ImageNet normalization
  rec:  resize to height 48, width padded to multiple of 8, mean/std 0.5

Usage (any venv with pynerve installed; onnxruntime needed for --onnx):
    python scripts/benchmark_onnx.py --both \
        --det path/to/ppocrv5_det.onnx --rec path/to/ppocrv5_rec.onnx \
        --keys path/to/matching_dict.txt
    python scripts/benchmark_onnx.py --onnx --lines 10 50

Models: https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx
(rec is the ~18k-class general model; use its matching dict for readable
output — decode is only a sanity check, latency is the benchmark.)
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODELS_DIR = Path("/tmp/pynerve-onnx-models")
DEFAULT_DET = MODELS_DIR / "ppocrv5_det.onnx"
DEFAULT_REC = MODELS_DIR / "ppocrv5_rec.onnx"
DEFAULT_KEYS = Path(__file__).resolve().parent.parent / "pynerve" / "models" / "ppocr_keys_en.txt"


# ---------------------------------------------------------------- synthetic data

def make_image(width: int, lines: int) -> Image.Image:
    height = 40 + lines * 40
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    for i in range(lines):
        d.text((20, 20 + i * 40), "File Edit View Settings Help Search Submit Cancel OK Apply", fill="black")
    return img


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- ONNX backend

class OnnxOCR:
    def __init__(self, det_path: str, rec_path: str, keys_path: str) -> None:
        import onnxruntime as ort

        self.det = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])
        det_in = self.det.get_inputs()[0]
        rec_in = self.rec.get_inputs()[0]
        self.det_name = det_in.name
        self.rec_name = rec_in.name
        self.det_out = self.det.get_outputs()[0].name
        self.rec_out = self.rec.get_outputs()[0].name
        print(f"  [onnx] det in={det_in.shape} out={self.det.get_outputs()[0].shape}")
        print(f"  [onnx] rec in={rec_in.shape} out={self.rec.get_outputs()[0].shape}")

        # keys: index 0 is the CTC blank, chars start at index 1 (same as ocr-rs)
        self.chars = [""] + [
            line.rstrip("\n") for line in Path(keys_path).read_text(encoding="utf-8").splitlines()
        ]

    # -- det ------------------------------------------------------------------

    def _det_preprocess(self, img: Image.Image) -> tuple[np.ndarray, float, int, int]:
        w, h = img.size
        scale = 1.0
        if max(w, h) > 960:
            scale = 960.0 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w2, h2 = img.size
        pad_w = (32 - w2 % 32) % 32
        pad_h = (32 - h2 % 32) % 32
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return arr[None], scale, w2, h2

    @staticmethod
    def _unclip(box: np.ndarray, ratio: float = 1.5) -> np.ndarray:
        area = cv2_contour_area(box)
        peri = cv2_arc_length(box, closed=True)
        dist = area * ratio / max(peri, 1e-6)
        return box + dist * (box / np.linalg.norm(box, axis=1, keepdims=True))

    def _detect(self, prob: np.ndarray, scale: float, w: int, h: int) -> list[tuple[int, int, int, int]]:
        import cv2

        mask = (prob > 0.3).astype(np.uint8)
        boxes: list[tuple[int, int, int, int]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 3:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = self._unclip(box)
            x0, y0 = box[:, 0].min(), box[:, 1].min()
            x1, y1 = box[:, 0].max(), box[:, 1].max()
            x0, y0 = int(max(0, x0 / scale)), int(max(0, y0 / scale))
            x1, y1 = int(min(w, x1 / scale)), int(min(h, y1 / scale))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append((x0, y0, x1 - x0, y1 - y0))
        return boxes

    # -- rec ------------------------------------------------------------------

    @staticmethod
    def _rec_preprocess(crop: Image.Image) -> np.ndarray:
        w, h = crop.size
        target_h = 48
        new_w = max(int(w * target_h / h), 16)
        pad = (8 - new_w % 8) % 8
        crop = crop.resize((new_w, target_h), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        arr = np.pad(arr, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return arr[None]

    @staticmethod
    def _decode(probs: np.ndarray, chars: list[str]) -> str:
        idxs = probs.argmax(axis=1)
        out = []
        prev = -1
        for i in idxs:
            # index 0 = CTC blank; skip out-of-range (model charset wider than
            # the provided keys — decode is only a sanity check, not the point
            # of this latency benchmark)
            if i != 0 and i != prev and i < len(chars):
                out.append(chars[i])
            prev = i
        return "".join(out)

    # -- pipeline --------------------------------------------------------------

    def ocr(self, img: Image.Image) -> tuple[list[str], float]:
        import onnxruntime as ort  # noqa: F401  (keep import visible for provider errors)

        t0 = time.perf_counter()
        arr, scale, w, h = self._det_preprocess(img)
        prob = self.det.run([self.det_out], {self.det_name: arr})[0]
        prob = prob[0, 0]  # (1,1,H,W) -> (H,W)
        boxes = self._detect(prob, scale, w, h)

        texts = []
        for (x, y, bw, bh) in boxes:
            crop = img.crop((x, y, x + bw, y + bh))
            inp = self._rec_preprocess(crop)
            out = self.rec.run([self.rec_out], {self.rec_name: inp})[0][0]
            texts.append(self._decode(out, self.chars))
        dt = time.perf_counter() - t0
        return texts, dt


# small cv2 wrappers to keep the import lazy
def cv2_contour_area(box: np.ndarray) -> float:
    import cv2

    return float(cv2.contourArea(box.astype(np.float32)))


def cv2_arc_length(box: np.ndarray, closed: bool) -> float:
    import cv2

    return float(cv2.arcLength(box.astype(np.float32), closed))


# ---------------------------------------------------------------- MNN backend

def mnn_ocr(png_bytes: bytes) -> tuple[list[str], float]:
    from pynerve import _native

    t0 = time.perf_counter()
    results = _native.ocr_from_png_bytes(png_bytes)
    dt = time.perf_counter() - t0
    return [text for text, _, _ in results], dt


# --------------------------------------------------------------------- driver

def run_mnn(width: int, lines: int, runs: int) -> tuple[float, int]:
    img = make_image(width, lines)
    png = to_png_bytes(img)
    times = []
    count = 0
    for _ in range(runs):
        texts, dt = mnn_ocr(png)
        times.append(dt)
        count = len(texts)
    return min(times), count


def run_onnx(ocr: OnnxOCR, width: int, lines: int, runs: int) -> tuple[float, int]:
    img = make_image(width, lines)
    times = []
    count = 0
    for _ in range(runs):
        texts, dt = ocr.ocr(img)
        times.append((dt, len(texts)))
        count = len(texts)
    return min(t for t, _ in times), count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnn", action="store_true")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--lines", nargs="+", type=int, default=[2, 10, 50])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--det", default=str(DEFAULT_DET))
    ap.add_argument("--rec", default=str(DEFAULT_REC))
    ap.add_argument("--keys", default=str(DEFAULT_KEYS))
    args = ap.parse_args()

    do_mnn = args.mnn or args.both
    do_onnx = args.onnx or args.both
    if not (do_mnn or do_onnx):
        ap.error("pass --mnn, --onnx or --both")

    onnx_ocr = None
    if do_onnx:
        print("Loading ONNX models ...")
        onnx_ocr = OnnxOCR(args.det, args.rec, args.keys)

    if do_mnn:
        from pynerve import _native

        root = Path(__file__).resolve().parent.parent / "pynerve" / "models"
        print("Loading MNN models ...")
        _native.init_ocr(
            str(root / "PP-OCRv5_mobile_det.mnn"),
            str(root / "en_PP-OCRv5_mobile_rec_infer.mnn"),
            str(root / "ppocr_keys_en.txt"),
        )

    print(f"\n{'lines':>6} | {'regions':>8} | {'MNN total':>12} | {'MNN ms/reg':>11} | {'ONNX total':>12} | {'ONNX ms/reg':>12}")
    print("-" * 78)
    for lines in args.lines:
        mnn_total = mnn_count = None
        onnx_total = nreg = None
        if do_mnn:
            mnn_total, mnn_count = run_mnn(args.width, lines, args.runs)
        if do_onnx:
            onnx_total, nreg = run_onnx(onnx_ocr, args.width, lines, args.runs)
        regions = nreg if nreg else mnn_count
        mnn_ms = f"{mnn_total*1000:8.0f}ms" if mnn_total else "-"
        mnn_per = f"{mnn_total*1000/mnn_count:6.1f}ms" if mnn_total and mnn_count else "-"
        onnx_ms = f"{onnx_total*1000:8.0f}ms" if onnx_total else "-"
        onnx_per = f"{onnx_total*1000/nreg:6.1f}ms" if onnx_total and nreg else "-"
        print(
            f"{lines:>6} | {regions:>8} | {mnn_ms:>12} | {mnn_per:>11} | {onnx_ms:>12} | {onnx_per:>12}"
        )
    print("-" * 78)
    print("Same PP-OCRv5 mobile det+rec; MNN uses batched recognition, ONNX runs serial per-crop rec;")
    print("preprocessing parity is approximate (resize algorithm + padding). Lower ms/reg wins.")


if __name__ == "__main__":
    main()
