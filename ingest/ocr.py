"""Local OCR fallback for scanned/image-based PDF pages -- fully offline
(Tesseract, a local binary, via pytesseract; page rasterization via
PyMuPDF/fitz, a pip-only dependency with no external system package like
poppler required). No page or image data ever leaves this machine.

This is a FALLBACK, not the primary extraction path: ingest/pdf.py still
tries pypdf's direct text-layer extraction first for every page (fast,
exact, no OCR error risk) and only calls into this module for the specific
pages that came back empty -- i.e. pages that are actually scanned images,
not real text. A mixed document (some real text pages, some scanned pages)
gets the right treatment per page, not an all-or-nothing choice.

HONEST ABOUT ACCURACY: OCR output is machine-read text, not a guarantee of
correctness -- a smudge, poor scan quality, or unusual layout can produce
wrong characters. It still goes through the exact same citation-verification
pipeline as any other passage (harness/verify.py): a claimed quote must be a
verbatim substring of what OCR actually produced. That doesn't make the OCR
itself perfect, but it does mean the model can never claim OCR text says
something it doesn't -- the same guarantee this app makes everywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

# Higher DPI = better OCR accuracy but slower. 300 is the standard
# "good enough for real-world scanned documents" tesseract recommendation.
OCR_RENDER_DPI = 300


class OcrUnavailableError(Exception):
    """Raised when the local Tesseract binary isn't installed/reachable --
    distinct from "this page just has no text", so callers can tell an OCR
    engine problem apart from a genuinely blank page."""


def _check_tesseract_available() -> None:
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OcrUnavailableError(
            "Local Tesseract OCR engine not found. Install it (e.g. `brew install tesseract` "
            "on macOS) -- this app never sends scanned pages to an external OCR service."
        ) from exc


def ocr_pdf_page(pdf_path: str | Path, page_index: int) -> str:
    """Rasterize one page (0-indexed) of a PDF and OCR it locally. Returns
    the extracted text (possibly empty if the page is genuinely blank, a
    photo with no text, etc. -- not every OCR miss is an error)."""
    _check_tesseract_available()
    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_index]
        zoom = OCR_RENDER_DPI / 72  # PDF points are 72 DPI by definition
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return pytesseract.image_to_string(image).strip()
    finally:
        doc.close()


def ocr_image_bytes(image_bytes: bytes) -> str:
    """OCR a standalone image (e.g. a photographed page or a scanned upload
    that isn't wrapped in a PDF at all) -- same local engine, same honesty
    about accuracy as ocr_pdf_page above."""
    _check_tesseract_available()
    import io

    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()
