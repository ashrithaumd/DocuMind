"""PII redaction pattern matching: masks the four categories, leaves normal
technical text alone, and never returns the matched values."""
import pytest

from graph.redaction import redact


@pytest.mark.parametrize(
    "text,expected_token,expected_categories",
    [
        ("Email john.doe+tag@example.co.uk please", "[EMAIL_REDACTED]", ["email"]),
        ("Call 555-123-4567 today", "[PHONE_REDACTED]", ["phone"]),
        ("Call (555) 123 4567 now", "[PHONE_REDACTED]", ["phone"]),
        ("Or +1 555.123.4567 works", "[PHONE_REDACTED]", ["phone"]),
        ("SSN is 123-45-6789 ok", "[SSN_REDACTED]", ["ssn"]),
        ("SSN 123 45 6789 spaced", "[SSN_REDACTED]", ["ssn"]),
        ("Card 4111-1111-1111-1111 on file", "[CARD_REDACTED]", ["credit_card"]),
        ("Card 4111111111111111 raw", "[CARD_REDACTED]", ["credit_card"]),
    ],
)
def test_pii_masked(text, expected_token, expected_categories):
    masked, categories = redact(text)
    assert expected_token in masked
    assert categories == expected_categories


def test_multiple_categories_reported_sorted():
    masked, categories = redact("Contact a@b.io, SSN 123-45-6789, tel 555-123-4567")
    assert categories == ["email", "phone", "ssn"]
    for fragment in ("a@b.io", "123-45-6789", "555-123-4567"):
        assert fragment not in masked


def test_phone_leading_punctuation_consumed():
    masked, _ = redact("call (555) 123 4567")
    assert "(" not in masked  # the mask swallows the paren, no stray "("


@pytest.mark.parametrize(
    "text",
    [
        "chunks are 600 tokens with 90-token overlap",
        "the pump runs at 400 RPM and p. 12 covers it",
        "revenue was 1,234,567 dollars in 2026",
        "see section 3.4.1 of the manual",
        "the year range 2020-2026 applies",
        "meeting at 10:30, room 204",
    ],
)
def test_normal_text_untouched(text):
    masked, categories = redact(text)
    assert masked == text
    assert categories == []


def test_empty_string():
    assert redact("") == ("", [])
