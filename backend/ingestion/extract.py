"""File validation and text extraction for all supported formats.

Common interface: every extractor takes raw bytes and returns a list of
PageText. The rest of the pipeline (chunking, embedding, storage) never knows
which format the text came from. PDF is the only format with real page
numbers; DOCX and TXT return a single "page" so citations degrade to the
filename rather than breaking.

All validation happens BEFORE any embedding spend: wrong file type, oversized
files, too many pages, encrypted PDFs, and image-only (scanned) PDFs are
rejected with a clear, user-facing error. OCR is out of scope for v1, so a
scanned PDF is a hard reject rather than a silent empty ingest.

Adding a format = one function + one dispatch entry. CSV/HTML/FHIR are
deliberately NOT here: each needs structure-aware handling (rows, DOM,
resources), not plain text extraction — see README known limitations.
"""
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import Settings


class IngestionError(ValueError):
    """User-facing ingestion failure. Surfaced by the API as HTTP 400."""


@dataclass
class PageText:
    page_number: int  # 1-based, matches what a PDF viewer shows
    text: str


# Pages with fewer non-whitespace characters than this are treated as
# image-only. Real text pages are hundreds of characters; this only filters
# out blanks and pure-image pages.
_MIN_CHARS_PER_PAGE = 25

_NO_TEXT_ERROR = "No extractable text found in this file."


# --- PDF -------------------------------------------------------------------


def _extract_pdf(file_bytes: bytes, settings: Settings) -> tuple[list[PageText], list[str]]:
    if not file_bytes.lstrip().startswith(b"%PDF"):
        raise IngestionError("File does not look like a valid PDF (missing %PDF header).")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            raise IngestionError("Encrypted PDFs are not supported.")
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise IngestionError(f"Could not parse PDF: {exc}") from exc

    if page_count > settings.max_pages:
        raise IngestionError(
            f"PDF has {page_count} pages; the limit is {settings.max_pages}."
        )

    pages: list[PageText] = []
    empty_pages = 0
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text.replace(" ", "").replace("\n", "")) < _MIN_CHARS_PER_PAGE:
            empty_pages += 1
            continue
        pages.append(PageText(page_number=i, text=text))

    if not pages:
        raise IngestionError(
            "No extractable text found — this appears to be a scanned or "
            "image-only PDF. OCR is not supported in v1."
        )

    warnings = []
    if empty_pages:
        warnings.append(
            f"{empty_pages} of {page_count} pages had no extractable text "
            "(likely images or blanks) and were skipped."
        )
    return pages, warnings


# --- DOCX --------------------------------------------------------------------


def _extract_docx(file_bytes: bytes, settings: Settings) -> tuple[list[PageText], list[str]]:
    try:
        document = Document(io.BytesIO(file_bytes))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise IngestionError("Could not parse DOCX — is this a real Word file?") from exc

    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    if not text.strip():
        raise IngestionError(_NO_TEXT_ERROR)

    # DOCX has no page concept before rendering, so everything is "page 1"
    # and citations fall back to the filename.
    warnings = []
    if document.tables:
        warnings.append(
            f"{len(document.tables)} table(s) were skipped — DOCX table "
            "extraction is not supported in v1."
        )
    return [PageText(page_number=1, text=text)], warnings


# --- TXT --------------------------------------------------------------------


def _extract_txt(file_bytes: bytes, settings: Settings) -> tuple[list[PageText], list[str]]:
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        # latin-1 maps every byte, so this cannot fail — worst case is a few
        # mangled characters in a file that wasn't really latin-1.
        text = file_bytes.decode("latin-1")
    if not text.strip():
        raise IngestionError(_NO_TEXT_ERROR)
    return [PageText(page_number=1, text=text)], []


# --- dispatch -----------------------------------------------------------------

_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}

SUPPORTED_EXTENSIONS = tuple(_EXTRACTORS)


def validate_and_extract(
    file_bytes: bytes, filename: str, settings: Settings
) -> tuple[list[PageText], list[str]]:
    """Validate an uploaded file and extract text, dispatching on extension.

    Returns (pages_with_text, warnings). Raises IngestionError on any hard
    reject so the caller can map it straight to a 400 response.
    """
    # PureWindowsPath handles both separator styles; suffix of "resume" is "".
    suffix = PureWindowsPath(PurePosixPath(filename).name).suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        shown = suffix or "(none)"
        raise IngestionError(
            f"Unsupported file type {shown!r}. Supported formats: PDF, DOCX, TXT."
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise IngestionError(
            f"File is {len(file_bytes) / 1_000_000:.1f} MB; the limit is "
            f"{settings.max_upload_mb} MB."
        )

    return extractor(file_bytes, settings)
