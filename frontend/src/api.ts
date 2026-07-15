// Single place the backend location is known. "/api" is proxied by Vite in
// dev (vite.config.ts) and by nginx in the Docker build (nginx.conf).
export const API_BASE = "/api";

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  uploaded_at: string | null;
  chunks: number;
}

export interface Citation {
  n: number;
  chunk_id: string;
  filename: string;
  page_number: number;
  snippet: string;
}

export interface StageEvent {
  stage: string;
  label: string;
  detail: Record<string, unknown>;
}

// Events emitted by POST /query/stream (SSE)
export type StreamEvent =
  | ({ type: "stage" } & StageEvent)
  | { type: "token"; text: string }
  | { type: "citations"; citations: Citation[] }
  | {
      type: "done";
      outcome: string;
      query_id: number | null; // null when backend logging is disabled
      timings: Record<string, number>;
    }
  | { type: "error"; message: string };
