// Firebase Auth only identifies the user; org, role and clearance come from
// the backend (GET /me). The web config is public by design (it names the
// project, it doesn't grant access) and comes from the VITE_FIREBASE_* values
// in the repo-root .env (vite.config.ts envDir) or the Docker build args.
import { initializeApp } from "firebase/app";
import { getAuth, type Auth } from "firebase/auth";

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// getAuth throws without an API key, so an unconfigured build shows a clear
// message (AuthGate) instead of a blank page.
export const auth: Auth | null = config.apiKey ? getAuth(initializeApp(config)) : null;
