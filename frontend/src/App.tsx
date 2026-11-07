import { useCallback, useEffect, useState } from "react";
import type { User } from "firebase/auth";
import AuthGate from "./AuthGate";
import Upload from "./Upload";
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
        <span className="user-badge" title={`Org ${me.org_id}, clearance ${me.clearance}`}>
          {user.email} · {me.role}
        </span>
        <button className="link-btn" onClick={onSignOut}>Sign out</button>
      </header>

      <main className="app-main">
        <Upload
          me={me}
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
