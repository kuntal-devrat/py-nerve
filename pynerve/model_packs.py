"""On-demand OCR model packs.

English ships inside the wheel (``pynerve/models/``) and never touches the
network. Other languages share the detection model and need a recognition
model + charset file; when they are missing, :func:`ensure_lang_pack`
downloads them from a configurable base URL::

    export DEXFLOW_MODELS_BASE_URL=https://example.com/dexflow-models
    nv.configure(lang="fr", models_base_url="https://...")  # or env only

Layout under the base URL mirrors the bundled directory::

    <base>/PP-OCRv5_mobile_det.mnn
    <base>/PP-OCRv5_mobile_rec.mnn
    <base>/ppocr_keys_v5.txt

Downloads are atomic (``.part`` + rename) and SHA-256 verified when a
checksum is known (``SHASUMS`` may be extended as packs are published).
Set ``DEXFLOW_MODELS_DIR`` to redirect the cache (slim installs).
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger("pynerve.models")

DET_FILE = "PP-OCRv5_mobile_det.mnn"

#: Recognition model + charset per language. English is bundled; every other
#: language currently maps to the multilingual PP-OCRv5 pack.
LANG_PACKS: dict[str, dict[str, str]] = {
    "en": {"rec": "en_PP-OCRv5_mobile_rec_infer.mnn", "keys": "ppocr_keys_en.txt"},
}
_MULTILINGUAL = {"rec": "PP-OCRv5_mobile_rec.mnn", "keys": "ppocr_keys_v5.txt"}

#: Optional SHA-256 checksums, keyed by filename. Empty by default — fill in
#: when publishing packs. Verification is skipped (with a warning) for
#: filenames without an entry.
SHASUMS: dict[str, str] = {}


def pack_filenames(lang: str) -> list[str]:
    """Filenames required for ``lang`` (detection + recognition + keys)."""
    pack = LANG_PACKS.get(lang, _MULTILINGUAL)
    return [DET_FILE, pack["rec"], pack["keys"]]


def pack_complete(lang: str, models_dir: str | Path) -> tuple[bool, list[str]]:
    """Check whether ``models_dir`` contains the pack. Returns (ok, missing)."""
    directory = Path(models_dir)
    missing = [name for name in pack_filenames(lang) if not (directory / name).exists()]
    return (not missing, missing)


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Explicit URL wins, otherwise ``DEXFLOW_MODELS_BASE_URL``."""
    return explicit or os.environ.get("DEXFLOW_MODELS_BASE_URL")


def ensure_lang_pack(
    lang: str,
    models_dir: str | Path,
    base_url: str | None = None,
    timeout: float = 60.0,
) -> bool:
    """Ensure the model pack for ``lang`` exists, downloading if possible.

    Returns True when the pack is complete afterwards. Returns False (never
    raises for network issues) when files are still missing and no base URL
    is configured — the caller then raises its usual helpful error.
    """
    directory = Path(models_dir)
    complete, missing = pack_complete(lang, directory)
    if complete:
        return True
    url = resolve_base_url(base_url)
    if not url:
        logger.debug("Model files %s missing and no base URL configured.", missing)
        return False
    directory.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in missing:
        try:
            _download(url.rstrip("/") + "/" + name, directory / name, timeout)
        except Exception as e:
            logger.warning("Failed to download model file %s: %s", name, e)
            ok = False
    complete, _ = pack_complete(lang, directory)
    return complete and ok


def _download(url: str, dest: Path, timeout: float) -> None:
    """Download ``url`` atomically to ``dest`` with optional SHA-256 check."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dexflow-model-fetch"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            digest.update(chunk)
            fh.write(chunk)
    expected = SHASUMS.get(dest.name)
    if expected and digest.hexdigest() != expected.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {dest.name} from {url}")
    if expected is None:
        logger.warning("No checksum known for %s; skipping verification.", dest.name)
    tmp.replace(dest)
    logger.info("Downloaded model file %s (%d bytes).", dest.name, dest.stat().st_size)
