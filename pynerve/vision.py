from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

from ._types import Element
from .exceptions import VisionError

logger = logging.getLogger("pynerve.vision")

# Models ship inside the package so pip-installed wheels work out of the box.
# Slim installs (or shared caches) can redirect with DEXFLOW_MODELS_DIR.
def _default_models_dir() -> Path:
    import os

    override = os.environ.get("DEXFLOW_MODELS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent / "models"


_MODELS_DIR = _default_models_dir()


class VisionEngine:
    """Local OCR engine using Rust-native PP-OCRv5 via ocr-rs."""

    def __init__(
        self,
        lang: str = "en",
        models_dir: str | Path | None = None,
        models_base_url: str | None = None,
    ) -> None:
        """Initialize the vision engine.

        Args:
            lang: Language for OCR. Default is English.
            models_dir: Directory containing MNN models and charset files.
                        Defaults to the models bundled with the package (or
                        ``DEXFLOW_MODELS_DIR`` when set).
            models_base_url: Base URL for on-demand model-pack downloads for
                non-English languages. Falls back to
                ``DEXFLOW_MODELS_BASE_URL``. English always uses the bundled
                pack and never touches the network.
        """
        self._lang = lang
        self._models_dir = Path(models_dir) if models_dir else _default_models_dir()
        self._models_base_url = models_base_url
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load the Rust OCR engine on first use."""
        if self._initialized:
            return

        try:
            from . import _native

            det_file = self._models_dir / "PP-OCRv5_mobile_det.mnn"

            # Pick recognition model based on language
            if self._lang == "en":
                rec_file = self._models_dir / "en_PP-OCRv5_mobile_rec_infer.mnn"
                keys_file = self._models_dir / "ppocr_keys_en.txt"
            else:
                # Default to multi-language v5 model
                rec_file = self._models_dir / "PP-OCRv5_mobile_rec.mnn"
                keys_file = self._models_dir / "ppocr_keys_v5.txt"

            if not det_file.exists() or not rec_file.exists() or not keys_file.exists():
                missing = [p.name for p in (det_file, rec_file, keys_file) if not p.exists()]
                if self._lang != "en":
                    # Try an on-demand download before giving up (English pack
                    # is bundled, so this path never triggers for lang="en").
                    from .model_packs import ensure_lang_pack

                    try:
                        if ensure_lang_pack(
                            self._lang, self._models_dir,
                            base_url=self._models_base_url,
                        ):
                            det_file = self._models_dir / "PP-OCRv5_mobile_det.mnn"
                            if self._lang == "en":
                                rec_file = self._models_dir / "en_PP-OCRv5_mobile_rec_infer.mnn"
                                keys_file = self._models_dir / "ppocr_keys_en.txt"
                            else:
                                rec_file = self._models_dir / "PP-OCRv5_mobile_rec.mnn"
                                keys_file = self._models_dir / "ppocr_keys_v5.txt"
                    except Exception as e:
                        logger.warning("Model-pack download failed: %s", e)
                if not det_file.exists() or not rec_file.exists() or not keys_file.exists():
                    missing = [p.name for p in (det_file, rec_file, keys_file) if not p.exists()]
                    raise FileNotFoundError(
                        f"OCR model files {missing} not found in {self._models_dir}. "
                        "For non-English OCR, provide the model files via models_dir=, "
                        "or set models_base_url=/DEXFLOW_MODELS_BASE_URL for on-demand download."
                    )

            _native.init_ocr(str(det_file), str(rec_file), str(keys_file))
            self._initialized = True
            logger.debug(
                "Rust OCR initialized (lang=%s, det=%s, rec=%s)",
                self._lang,
                det_file,
                rec_file,
            )
        except Exception as e:
            raise VisionError(f"Failed to initialize Rust OCR engine: {e}") from e

    def extract_layout(
        self,
        image: Image.Image | str | bytes | None = None,
        region: tuple[int, int, int, int] | None = None,
        monitor_index: int | None = None,
    ) -> list[Element]:
        """Process an image and return standardized structural elements.

        Fast paths (no Python-side image handling at all):
        - ``region=...`` captures the screen region and OCRs it natively in Rust.
        - ``image`` as ``bytes`` is OCR'd straight from memory.
        - ``image=None`` OCRs the full screen natively.

        Args:
            image: PIL Image, file path, or PNG bytes to process. Defaults to a
                   full-screen capture when both ``image`` and ``region`` are None.
            region: Optional (x, y, width, height) screen region to capture.
            monitor_index: Optional zero-based monitor index to capture.

        Returns:
            List of Element objects with text, confidence, center, and bounds.
        """
        self._ensure_initialized()

        from . import _native

        try:
            if region is not None or image is None:
                ocr_results = _native.capture_ocr(region, monitor_index)
            elif isinstance(image, bytes):
                ocr_results = _native.ocr_from_png_bytes(image)
            else:
                png_bytes = self._to_png_bytes(image)
                ocr_results = _native.ocr_from_png_bytes(png_bytes)
        except Exception as e:
            raise VisionError(f"OCR processing failed: {e}") from e

        elements: list[Element] = []

        for text, confidence, (x, y, w, h) in ocr_results:
            if not text or not text.strip():
                continue

            bounds = (float(x), float(y), float(x + w), float(y + h))
            center_x = (bounds[0] + bounds[2]) / 2.0
            center_y = (bounds[1] + bounds[3]) / 2.0

            elements.append(
                Element(
                    text=text.strip(),
                    confidence=confidence,
                    center=(center_x, center_y),
                    bounds=bounds,
                )
            )

        logger.debug("Extracted %d elements from image", len(elements))
        return elements

    @staticmethod
    def _to_png_bytes(image: Image.Image | str | bytes) -> bytes:
        """Convert input to PNG bytes for the Rust OCR engine."""
        if isinstance(image, bytes):
            # Already PNG bytes (from screenshot)
            return image

        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()

        if isinstance(image, str):
            # File path
            img = Image.open(image)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()

        raise VisionError(f"Unsupported image type: {type(image)}")
