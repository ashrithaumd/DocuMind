"""Regex-based PII masking, applied before any text reaches an LLM.

Deliberately basic v1: four high-signal patterns (emails, phone numbers,
SSN-like, credit-card-like), no NLP entity recognition. The patterns favor
precision over recall — a missed phone number in an odd format goes to the
LLM unmasked, but a false positive would corrupt a legitimate question, so
digit runs only match when they carry PII-shaped separators/grouping.

What this does NOT catch (and the production upgrade path): names, addresses,
dates of birth, unseparated 10-digit phone numbers, non-US formats. In
production this would be a dedicated detector — Microsoft Presidio is the
obvious drop-in (same mask-and-categorize contract), or a cloud DLP API.

Only the CATEGORY of what was masked is ever logged, never the value.
"""
import re

# Order matters: credit cards are matched before SSN/phone so a 16-digit
# grouped number is consumed as one card, not partially as a phone number.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    (
        "credit_card",
        # 13-16 digits, either grouped 4-4-4-4 or one unbroken run.
        re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{13,16}\b"),
        "[CARD_REDACTED]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
        "[SSN_REDACTED]",
    ),
    (
        "phone",
        # US-style with separators: 555-123-4567, (555) 123 4567, +1 555.123.4567
        # (?<!\w) instead of \b so a leading "(" or "+" is consumed by the mask.
        re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]?\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
]


def redact(text: str) -> tuple[str, list[str]]:
    """Mask PII in `text`. Returns (masked_text, sorted category names)."""
    categories: set[str] = set()
    for name, pattern, replacement in _PATTERNS:
        text, hits = pattern.subn(replacement, text)
        if hits:
            categories.add(name)
    return text, sorted(categories)
