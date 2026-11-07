import { useCallback, useEffect, useState } from "react";
import Upload from "./Upload";
import Chat from "./Chat";
import { API_BASE, type DocumentInfo } from "./api";
import "./index.css";

/** The document list, or null if the backend isn't reachable yet. */
async function fetchDocuments(): Promise<DocumentInfo[] | null> {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) return null;
    const data = await res.json();
    return data.documents ?? [];
  } catch {
    return null; // Backend not reachable yet — the list just stays as it is.
  }
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  // Empty selection = default scope (the backend queries the newest upload).
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  const refreshDocuments = useCallback(async () => {
    const docs = await fetchDocuments();
    if (docs) setDocuments(docs);
  }, []);

  useEffect(() => {
    // setState only in the async callback, never in the effect body; skipped
    // if the component unmounted first.
    let active = true;
    fetchDocuments().then((docs) => {
      if (active && docs) setDocuments(docs);
    });
    return () => {
      active = false;
    };
  }, []);

  function toggleDoc(docId: string) {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((d) => d !== docId) : [...prev, docId]
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>DocuMind</h1>
        <span className="tagline">AI Document Intelligence</span>
      </header>

      <main className="app-main">
        <Upload
          onUploadSuccess={refreshDocuments}
          documents={documents}
          selectedDocIds={selectedDocIds}
          onToggleDoc={toggleDoc}
        />
        <Chat docIds={selectedDocIds} />
      </main>
    </div>
  );
}
