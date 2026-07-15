import { useCallback, useEffect, useState } from "react";
import Upload from "./Upload";
import Chat from "./Chat";
import { API_BASE, type DocumentInfo } from "./api";
import "./index.css";

export default function App() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  // Empty selection = default scope (the backend queries the newest upload).
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  const refreshDocuments = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (!res.ok) return;
      const data = await res.json();
      setDocuments(data.documents ?? []);
    } catch {
      // Backend not reachable yet — the list just stays empty.
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

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
