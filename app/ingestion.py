"""PDF ingestion with PyMuPDF. Keeps filename / page number / doc id on every page."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf

logger = logging.getLogger(__name__)


class PDFIngestionError(Exception):
    """Base class for ingestion problems."""


class CorruptPDFError(PDFIngestionError):
    """The file is not a readable PDF."""


class EmptyPDFError(PDFIngestionError):
    """The PDF has no extractable text (empty or scanned images)."""


@dataclass
class PageContent:
    doc_id: str
    filename: str
    page_number: int  # 1-indexed
    total_pages: int
    text: str


def make_doc_id(content: bytes) -> str:
    """Stable id from file content, so re-uploading the same file does not duplicate it."""
    return hashlib.sha1(content).hexdigest()[:12]


def parse_pdf_bytes(content: bytes, filename: str) -> List[PageContent]:
    if not content:
        raise EmptyPDFError(f"'{filename}' is an empty file.")
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise CorruptPDFError(f"'{filename}' is not a valid PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise CorruptPDFError(f"'{filename}' is password protected.")
        doc_id = make_doc_id(content)
        total = doc.page_count
        pages: List[PageContent] = []
        for i in range(total):
            try:
                text = doc.load_page(i).get_text("text").strip()
            except Exception as exc:  # one bad page should not kill the whole file
                logger.warning("Could not read page %d of %s: %s", i + 1, filename, exc)
                continue
            if text:
                pages.append(PageContent(doc_id, filename, i + 1, total, text))
    finally:
        doc.close()

    if not pages:
        raise EmptyPDFError(
            f"'{filename}' contains no extractable text (it may be a scanned PDF that needs OCR)."
        )
    return pages


def parse_pdf_file(path: Path) -> List[PageContent]:
    path = Path(path)
    return parse_pdf_bytes(path.read_bytes(), path.name)
