import { useState, useRef } from "react";
import { API_BASE, type DocumentInfo } from "./api";

interface UploadResult {
  filename: string;
  pages: number;
  chunks_created: number;
  warnings: string[];
}

interface Props {
  onUploadSuccess: () => void;
  documents: DocumentInfo[];
  selectedDocIds: string[];
  onToggleDoc: (docId: string) => void;
}

export default function Upload({
  onUploadSuccess,
  documents,
  selectedDocIds,
  onToggleDoc,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setStatus("idle");
      setResult(null);
    }
  }

  async function handleUpload() {
    if (!file) return;
    setStatus("uploading");
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        // The backend rejects bad uploads (wrong type, oversized, scanned
        // PDFs) with a specific message — show it verbatim.
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail ?? `Upload failed (${res.status})`);
      }

      const data: UploadResult = await res.json();
      setResult(data);
      setStatus("done");
      onUploadSuccess();
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  }

  return (
    <div className="upload-panel">
      <h2>Upload Document</h2>
      <p className="subtitle">
        PDF, DOCX, or TXT with extractable text (no scans — OCR is out of scope).
      </p>

      <label className="file-drop">
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        <span className="file-icon">📄</span>
        <span>{file ? file.name : "Click to select a PDF, DOCX, or TXT"}</span>
      </label>

      <button
        className="upload-btn"
        onClick={handleUpload}
        disabled={!file || status === "uploading"}
      >
        {status === "uploading" ? "Ingesting…" : "Upload"}
      </button>

      {status === "done" && result && (
        <div className="upload-success">
          ✅ <strong>{result.filename}</strong>
          <br />
          {result.pages} pages · {result.chunks_created} chunks indexed
        </div>
      )}

      {status === "done" && result && result.warnings.length > 0 && (
        <div className="upload-warning">
          {result.warnings.map((w) => (
            <div key={w}>⚠️ {w}</div>
          ))}
        </div>
      )}

      {status === "error" && <div className="upload-error">❌ {errorMsg}</div>}

      {documents.length > 0 && (
        <div className="doc-list">
          <h3>Documents</h3>
          <p className="doc-hint">
            {selectedDocIds.length === 0
              ? "Questions go to the newest document. Tick boxes to choose."
              : `Questions go to ${selectedDocIds.length} selected document${
                  selectedDocIds.length > 1 ? "s" : ""
                }.`}
          </p>
          {documents.map((doc, i) => (
            <label key={doc.doc_id} className="doc-item">
              <input
                type="checkbox"
                checked={selectedDocIds.includes(doc.doc_id)}
                onChange={() => onToggleDoc(doc.doc_id)}
              />
              <span className="doc-name" title={doc.filename}>
                {doc.filename}
              </span>
              {selectedDocIds.length === 0 && i === 0 && (
                <span className="doc-default">default</span>
              )}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
