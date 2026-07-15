"""Full ingestion path through the API against real (embedded) Chroma:
upload each supported format, verify chunks land in the index, verify
/documents ordering, and verify API-level rejections."""
from tests.helpers import make_docx, make_pdf


def test_pdf_docx_txt_ingest_and_are_listed_newest_first(client):
    uploads = [
        ("alpha.pdf", make_pdf("The reactor cooling system operates at 400 RPM under load."), "application/pdf"),
        ("beta.docx", make_docx(["Revenue grew 12 percent year over year.",
                                 "Headcount reached 240 employees in June."]),
         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("gamma.txt", b"Meeting notes. The Q3 budget was approved at 1.2 million.", "text/plain"),
    ]

    doc_ids = []
    for name, data, mime in uploads:
        response = client.post("/upload", files={"file": (name, data, mime)})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["filename"] == name
        assert body["chunks_created"] >= 1
        doc_ids.append(body["doc_id"])

    # Index actually grew (real Chroma + BM25, not a mock).
    health = client.get("/health").json()
    assert health["chunks_indexed"] >= 3

    # /documents: newest first, so the reverse of upload order.
    documents = client.get("/documents").json()["documents"]
    assert [d["doc_id"] for d in documents] == list(reversed(doc_ids))
    assert documents[0]["filename"] == "gamma.txt"
    assert all(d["uploaded_at"] for d in documents)


def test_unsupported_type_rejected_with_400(client):
    response = client.post(
        "/upload", files={"file": ("data.csv", b"a,b\n1,2", "text/csv")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert "PDF, DOCX, TXT" in response.json()["detail"]


def test_scanned_pdf_still_rejected(client):
    # A PDF with no extractable text (the pre-existing OCR hard-reject).
    response = client.post(
        "/upload", files={"file": ("scan.pdf", make_pdf("tiny"), "application/pdf")}
    )
    assert response.status_code == 400
    assert "scanned or image-only" in response.json()["detail"]


def test_empty_docx_rejected(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.docx", make_docx([]),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 400
    assert "No extractable text" in response.json()["detail"]
