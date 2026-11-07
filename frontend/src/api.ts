import { auth } from "./firebase";

// Single place the backend location is known. "/api" is proxied by Vite in
// dev (vite.config.ts) and by nginx in the Docker build (nginx.conf).
export const API_BASE = "/api";

// GET /me: who the token belongs to and whether they may use the app.
export interface Me {
  firebase_uid: string;
  set_up: boolean;
  reason: "unknown_user" | "no_org" | "inactive" | null;
  user_id?: string;
  org_id?: string;
  role?: "viewer" | "editor" | "admin";
  clearance?: number;
  team_ids?: string[];
}

// Called when any API call gets a 403, so AuthGate can re-check /me (the
// account may have just been deactivated or moved out of its org).
let onForbidden: (() => void) | null = null;
export function setForbiddenHandler(handler: (() => void) | null) {
  onForbidden = handler;
}

/** fetch() against the backend, signed in: every call carries the user's
 * Firebase ID token. getIdToken() returns the cached token, or refreshes it
 * first when it expires within five minutes, so a long session keeps working. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const user = auth?.currentUser;
  if (user) headers.set("Authorization", `Bearer ${await user.getIdToken()}`);
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 403 && path !== "/me") onForbidden?.();
  return res;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  uploaded_at: string | null;
  chunks: number;
  team_id?: string | null;
  clearance?: number;
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
