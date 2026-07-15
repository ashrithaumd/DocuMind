import { useState, useRef, useEffect } from "react";
import { API_BASE, type Citation, type StageEvent, type StreamEvent } from "./api";

interface Message {
  role: "user" | "assistant";
  text: string;
  citations: Citation[];
  stages: StageEvent[];
  outcome?: string; // answered | small_talk | refused_* | rejected_input | error
  totalMs?: number;
  queryId?: number | null; // Postgres log row id — what feedback links to
  streaming: boolean;
}

function FeedbackControl({ queryId }: { queryId: number }) {
  const [verdict, setVerdict] = useState<"up" | "down" | null>(null);
  const [comment, setComment] = useState("");
  const [commentSent, setCommentSent] = useState(false);

  // Fire-and-forget: feedback must never disrupt the chat, so errors are
  // swallowed (the backend logs its side of any failure).
  function send(thumbsUp: boolean, text?: string) {
    fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_id: queryId,
        thumbs_up: thumbsUp,
        comment: text || undefined,
      }),
    }).catch(() => {});
  }

  function clickThumb(thumbsUp: boolean) {
    setVerdict(thumbsUp ? "up" : "down");
    send(thumbsUp);
  }

  function submitComment() {
    if (!comment.trim()) return;
    // The comment box only exists after a thumbs-down; upserts onto the same row.
    send(false, comment.trim());
    setCommentSent(true);
  }

  return (
    <div className="feedback">
      <button
        className={`thumb${verdict === "up" ? " active" : ""}`}
        onClick={() => clickThumb(true)}
        title="Good answer"
        aria-label="Thumbs up"
      >
        👍
      </button>
      <button
        className={`thumb${verdict === "down" ? " active" : ""}`}
        onClick={() => clickThumb(false)}
        title="Bad answer"
        aria-label="Thumbs down"
      >
        👎
      </button>
      {verdict === "down" && !commentSent && (
        <span className="feedback-comment">
          <input
            type="text"
            placeholder="What went wrong? (optional)"
            value={comment}
            maxLength={2000}
            onChange={(e) => setComment(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submitComment()}
          />
          <button onClick={submitComment} disabled={!comment.trim()}>
            Send
          </button>
        </span>
      )}
      {(verdict === "up" || commentSent) && (
        <span className="feedback-thanks">Thanks!</span>
      )}
    </div>
  );
}

const isRefusal = (outcome?: string) =>
  !!outcome && outcome !== "answered" && outcome !== "small_talk";

interface Props {
  docIds: string[]; // empty = backend default (newest upload)
}

export default function Chat({ docIds }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Immutably update the in-flight assistant message (always the last one).
  function patchLast(patch: (last: Message) => Message) {
    setMessages((prev) => [...prev.slice(0, -1), patch(prev[prev.length - 1])]);
  }

  async function sendMessage() {
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput("");
    setIsStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question, citations: [], stages: [], streaming: false },
      { role: "assistant", text: "", citations: [], stages: [], streaming: true },
    ]);

    try {
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          doc_ids: docIds.length > 0 ? docIds : undefined,
        }),
      });
      if (!res.ok || !res.body) throw new Error(`Backend returned ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      // SSE events can be split across reads, so buffer until "\n\n".
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const raw of events) {
          if (!raw.startsWith("data: ")) continue;
          const event = JSON.parse(raw.slice(6)) as StreamEvent;

          if (event.type === "stage") {
            patchLast((m) => ({ ...m, stages: [...m.stages, event] }));
          } else if (event.type === "token") {
            patchLast((m) => ({ ...m, text: m.text + event.text }));
          } else if (event.type === "citations") {
            patchLast((m) => ({ ...m, citations: event.citations }));
          } else if (event.type === "done") {
            patchLast((m) => ({
              ...m,
              outcome: event.outcome,
              totalMs: event.timings?.total_ms,
              queryId: event.query_id,
              streaming: false,
            }));
          } else if (event.type === "error") {
            patchLast((m) => ({
              ...m,
              text: event.message,
              outcome: "error",
              streaming: false,
            }));
          }
        }
      }
    } catch {
      patchLast((m) => ({
        ...m,
        text: "Something went wrong. Is the backend running?",
        outcome: "error",
        streaming: false,
      }));
    } finally {
      setIsStreaming(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-panel">
      <h2>Ask DocuMind</h2>
      <p className="subtitle">
        Answers come only from your documents — with citations, or an honest refusal.
      </p>

      <div className="message-list">
        {messages.length === 0 && (
          <p className="empty-state">
            Upload a document (PDF, DOCX, or TXT), then ask anything about it.
          </p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className={`bubble${isRefusal(msg.outcome) ? " refusal" : ""}`}>
              <span className="role-label">
                {msg.role === "user" ? "You" : "DocuMind"}
              </span>

              {msg.role === "assistant" && msg.stages.length > 0 && (
                <div className="stage-row">
                  {msg.stages.map((s, j) => {
                    // Surface PII masking explicitly — category names only,
                    // never the masked values (the backend doesn't send them).
                    const masked =
                      s.stage === "input_validation" &&
                      Array.isArray(s.detail?.redacted) &&
                      (s.detail.redacted as string[]).length > 0
                        ? (s.detail.redacted as string[])
                        : null;
                    return (
                      <span
                        key={j}
                        className={`stage-chip${masked ? " redacted" : ""}`}
                        title={JSON.stringify(s.detail)}
                      >
                        {masked ? `PII masked: ${masked.join(", ")}` : s.label}
                      </span>
                    );
                  })}
                  {msg.streaming && <span className="stage-chip pulsing">…</span>}
                </div>
              )}

              <p>
                {msg.text}
                {msg.streaming && <span className="cursor">▌</span>}
              </p>

              {msg.citations.length > 0 && (
                <div className="sources">
                  <span className="sources-label">Citations:</span>
                  {msg.citations.map((c) => (
                    <span
                      key={c.chunk_id}
                      className="source-chip"
                      title={c.snippet}
                    >
                      [{c.n}] {c.filename} · p. {c.page_number}
                    </span>
                  ))}
                </div>
              )}

              {msg.role === "assistant" && !msg.streaming && msg.totalMs != null && (
                <span className="meta-line">
                  {isRefusal(msg.outcome) ? `${msg.outcome} · ` : ""}
                  {(msg.totalMs / 1000).toFixed(1)}s
                </span>
              )}

              {msg.role === "assistant" &&
                !msg.streaming &&
                msg.queryId != null &&
                msg.outcome !== "error" && (
                  <FeedbackControl key={msg.queryId} queryId={msg.queryId} />
                )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <input
          type="text"
          placeholder="Ask a question…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
          className="chat-input"
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim() || isStreaming}
          className="send-btn"
        >
          {isStreaming ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
