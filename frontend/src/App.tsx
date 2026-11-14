import { useEffect, useState } from "react";
import type { User } from "firebase/auth";
import AuthGate from "./AuthGate";
import Chat from "./Chat";
import { apiFetch, type DocumentInfo, type Me } from "./api";
import "./index.css";

/** The document list, or null if the backend isn't reachable yet. */
async function fetchDocuments(): Promise<DocumentInfo[] | null> {
  try {
    const res = await apiFetch("/documents");
    if (!res.ok) return null;
    const data = await res.json();
    return data.documents ?? [];
  } catch {
    return null; // Backend not reachable yet — the list just stays as it is.
  }
}

export default function App() {
  return (
    <AuthGate>
      {(me, user, signOut) => <Workspace me={me} user={user} onSignOut={signOut} />}
    </AuthGate>
  );
}

function Workspace({ me, user, onSignOut }: { me: Me; user: User; onSignOut: () => void }) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  // Empty selection = default scope (the backend queries the newest document).
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

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
        <h1>FinPilot</h1>
        <span className="tagline">Your banking assistant</span>
        <span className="user-badge" title={`Org ${me.org_id}, clearance ${me.clearance}`}>
          {user.email} · {me.role}
        </span>
        <button className="link-btn" onClick={onSignOut}>Sign out</button>
      </header>

      <main className="app-main">
        <DocumentList
          documents={documents}
          selectedDocIds={selectedDocIds}
          onToggleDoc={toggleDoc}
        />
        <Chat docIds={selectedDocIds} />
      </main>
    </div>
  );
}

function DocumentList({
  documents,
  selectedDocIds,
  onToggleDoc,
}: {
  documents: DocumentInfo[];
  selectedDocIds: string[];
  onToggleDoc: (docId: string) => void;
}) {
  return (
    <div className="upload-panel">
      <h2>Documents</h2>
      {documents.length === 0 ? (
        <p className="doc-hint">No documents loaded yet.</p>
      ) : (
        <div className="doc-list">
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
