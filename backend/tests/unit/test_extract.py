"""File-type validation and per-format extraction, including rejections."""
import pytest

from config import Settings
from ingestion.extract import IngestionError, validate_and_extract
from tests.helpers import make_docx, make_pdf

SETTINGS = Settings(anthropic_api_key="x")


def test_pdf_extracts_pages():
    pdf = make_pdf("The reactor cooling system operates at 400 RPM under load.")
    pages, warnings = validate_and_extract(pdf, "manual.pdf", SETTINGS)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "400 RPM" in pages[0].text
    assert warnings == []


def test_docx_extracts_single_page():
    docx = make_docx(["Revenue grew 12 percent.", "Headcount reached 240."])
    pages, warnings = validate_and_extract(docx, "report.docx", SETTINGS)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Revenue grew 12 percent." in pages[0].text
    assert "Headcount reached 240." in pages[0].text


def test_txt_plain_and_bom():
    pages, _ = validate_and_extract(b"Budget approved for Q3.", "notes.txt", SETTINGS)
    assert pages[0].text == "Budget approved for Q3."

    bom = "﻿Budget approved for Q3.".encode("utf-8")
    pages, _ = validate_and_extract(bom, "notes.txt", SETTINGS)
    assert pages[0].text == "Budget approved for Q3."  # BOM stripped


def test_txt_non_utf8_falls_back():
    pages, _ = validate_and_extract("café menu".encode("latin-1"), "menu.txt", SETTINGS)
    assert "menu" in pages[0].text


@pytest.mark.parametrize(
    "filename",
    ["data.csv", "page.html", "bundle.fhir.json", "notes.md", "archive.zip", "noext"],
)
def test_unsupported_types_rejected(filename):
    with pytest.raises(IngestionError, match="Unsupported file type"):
        validate_and_extract(b"some content", filename, SETTINGS)


def test_extension_check_is_case_insensitive():
    pages, _ = validate_and_extract(b"Shouted content here.", "NOTES.TXT", SETTINGS)
    assert pages[0].text == "Shouted content here."


def test_fake_pdf_rejected():
    with pytest.raises(IngestionError, match="%PDF header"):
        validate_and_extract(b"not a pdf at all", "fake.pdf", SETTINGS)


def test_fake_docx_rejected():
    with pytest.raises(IngestionError, match="Could not parse DOCX"):
        validate_and_extract(b"not a zip archive", "fake.docx", SETTINGS)


def test_empty_txt_rejected():
    with pytest.raises(IngestionError, match="No extractable text"):
        validate_and_extract(b"   \n\t  ", "empty.txt", SETTINGS)


def test_oversized_file_rejected():
    small_limit = Settings(anthropic_api_key="x", max_upload_mb=1)
    with pytest.raises(IngestionError, match="limit is 1 MB"):
        validate_and_extract(b"x" * (2 * 1024 * 1024), "big.txt", small_limit)


def test_type_check_applies_before_size_check():
    # An oversized file with a bad extension reports the type error, not the
    # size error — the user should fix the format before worrying about size.
    small_limit = Settings(anthropic_api_key="x", max_upload_mb=1)
    with pytest.raises(IngestionError, match="Unsupported file type"):
        validate_and_extract(b"x" * (2 * 1024 * 1024), "big.exe", small_limit)
