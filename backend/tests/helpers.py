"""Shared test doubles and tiny real test-file generators."""
import hashlib
import io
import zlib


# --- fake LLM + embedder ------------------------------------------------------


def _fake_vector(text: str) -> list[float]:
    """Deterministic 32-dim embedding: same text -> same vector, and vectors
    for different texts differ, which is all retrieval tests need."""
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:32]]


class FakeLLM:
    """Stands in for the app's single LLM gateway (llm.GeminiLLM /
    llm.ClaudeLLM — same interface; tests/unit/test_llm_gateway.py covers the
    real classes against stubbed SDK clients). Dispatches
    on the system prompt (grader / groundedness auditor / generator) and
    records every call so tests can assert call counts and prompt contents.

    Knobs:
        relevant   - what the grading call returns
        grounded   - what the groundedness call returns (a list pops one value
                     per call, so a test can fail the first check then pass)
        answer     - what generation returns

    `embed_calls` mirrors the paired FakeEmbedder's activity so tests can
    assert "zero model traffic" (chat AND embeddings) in one place.
    """

    def __init__(self):
        self.relevant = True
        self.grounded: bool | list[bool] = True
        self.answer = "The document says the pump runs at 400 RPM [1]."
        self.chat_calls: list[dict] = []
        self.embed_calls: list[list[str]] = []

    async def complete_text(self, model, system, user, max_tokens=4096) -> str:
        self.chat_calls.append({"model": model, "system": system, "user": user})
        return self.answer

    async def complete_json(self, model, system, user, schema, max_tokens=1024) -> dict:
        self.chat_calls.append({"model": model, "system": system, "user": user})
        if "relevance grader" in system:
            return {"relevant": self.relevant}
        if "groundedness auditor" in system:
            grounded = self.grounded
            if isinstance(grounded, list):
                grounded = grounded.pop(0)
            return {
                "grounded": grounded,
                "unsupported_claims": [] if grounded else ["a bad claim"],
            }
        raise AssertionError(f"unexpected JSON call with system prompt: {system[:60]}")

    def prompts_sent(self) -> str:
        """Every prompt string that reached the fake LLM, concatenated."""
        return "\n".join(
            part for call in self.chat_calls for part in (call["system"], call["user"])
        )


class FakeEmbedder:
    """Stands in for the sentence-transformers model: same interface the
    store uses (encode + get_embedding_dimension, the sentence-transformers 6
    name — deliberately not the deprecated get_sentence_embedding_dimension),
    zero deps."""

    def __init__(self, record_into: list | None = None):
        self.embed_calls = record_into if record_into is not None else []

    def encode(self, texts, normalize_embeddings=True):
        self.embed_calls.append(list(texts))
        return [_fake_vector(t) for t in texts]

    def get_embedding_dimension(self) -> int:
        return 32


# --- tiny real test files ------------------------------------------------------


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with real extractable text (no library needed —
    pypdf can read but not author text, and a PDF-writing dependency just for
    tests isn't worth it)."""
    # Escape PDF string delimiters.
    safe = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1", "replace")
    compressed = zlib.compress(stream)

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(compressed)
        + compressed
        + b"\nendstream",
    ]

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + obj + b"\nendobj\n")
    xref_at = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
        % (len(objects) + 1, xref_at)
    )
    return out.getvalue()


def make_docx(paragraphs: list[str]) -> bytes:
    from docx import Document

    buf = io.BytesIO()
    document = Document()
    for p in paragraphs:
        document.add_paragraph(p)
    document.save(buf)
    return buf.getvalue()
