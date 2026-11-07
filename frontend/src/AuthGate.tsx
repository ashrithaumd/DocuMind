import { useCallback, useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
  type User,
} from "firebase/auth";
import { auth } from "./firebase";
import { apiFetch, setForbiddenHandler, type Me } from "./api";

// What each /me reason means for the person looking at the screen.
const NOT_SET_UP: Record<string, string> = {
  unknown_user: "Your account isn't registered in DocuMind yet. Ask an admin to add you.",
  no_org: "Your account isn't assigned to an organization yet. Ask an admin to add you to one.",
  inactive: "Your account has been deactivated. Ask an admin if you think that's a mistake.",
};

const SIGN_IN_ERRORS: Record<string, string> = {
  "auth/invalid-credential": "Wrong email or password.",
  "auth/invalid-email": "That isn't a valid email address.",
  "auth/too-many-requests": "Too many attempts. Wait a minute and try again.",
  "auth/network-request-failed": "Couldn't reach Firebase. Check your connection.",
};

interface MeState {
  uid: string;
  me: Me | null;
  error?: string;
}

async function fetchMe(forUser: User): Promise<MeState> {
  try {
    const res = await apiFetch("/me");
    if (!res.ok) throw new Error(`The backend answered ${res.status}.`);
    return { uid: forUser.uid, me: await res.json() };
  } catch (err) {
    return { uid: forUser.uid, me: null,
             error: err instanceof Error ? err.message : "Couldn't reach the backend." };
  }
}

interface Props {
  // The app itself, once the user is signed in and set up.
  children: (me: Me, user: User, signOut: () => void) => ReactNode;
}

/** Sign-in screen, then GET /me: a set-up user gets the app; anyone else
 * gets a "not set up yet" screen with the reason. Any 403 from the API
 * re-checks /me, so a deactivated user lands there on their next request. */
export default function AuthGate({ children }: Props) {
  // undefined = Firebase hasn't reported yet (restoring a saved session).
  const [user, setUser] = useState<User | null | undefined>(auth ? undefined : null);
  // /me for one uid; ignored once a different user (or nobody) is signed in.
  const [me, setMe] = useState<MeState | null>(null);

  useEffect(() => (auth ? onAuthStateChanged(auth, setUser) : undefined), []);

  useEffect(() => {
    if (!user) return;
    let active = true;
    // State is set only once /me answers, never in the effect body.
    const reload = () => fetchMe(user).then((result) => {
      if (active) setMe(result);
    });
    reload();
    setForbiddenHandler(reload);
    return () => {
      active = false;
      setForbiddenHandler(null);
    };
  }, [user]);

  const doSignOut = useCallback(() => {
    if (auth) signOut(auth);
  }, []);

  if (!auth) {
    return (
      <Screen title="Sign-in isn't configured">
        Set the VITE_FIREBASE_* values in the repo-root .env (see .env.example) and rebuild.
      </Screen>
    );
  }
  if (user === undefined) return <Screen title="Loading…" />;
  if (user === null) return <SignIn />;

  const current = me?.uid === user.uid ? me : null;
  if (!current) return <Screen title="Loading…" />;
  if (current.error || !current.me) {
    return (
      <Screen title="Can't reach DocuMind" onSignOut={doSignOut} email={user.email}>
        {current.error}
      </Screen>
    );
  }
  if (!current.me.set_up) {
    return (
      <Screen title="Not set up yet" onSignOut={doSignOut} email={user.email}>
        <p>{NOT_SET_UP[current.me.reason ?? ""] ?? "Your account can't use DocuMind yet."}</p>
        <p className="auth-reason">Reason: {current.me.reason}</p>
      </Screen>
    );
  }
  return <>{children(current.me, user, doSignOut)}</>;
}

function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!auth) return;
    setBusy(true);
    setError("");
    try {
      await signInWithEmailAndPassword(auth, email.trim(), password);
      // onAuthStateChanged takes it from here.
    } catch (err) {
      const code = (err as { code?: string }).code ?? "";
      setError(SIGN_IN_ERRORS[code] ?? `Sign-in failed (${code || "unknown error"}).`);
      setBusy(false);
    }
  }

  return (
    <Screen title="Sign in to DocuMind">
      <form className="auth-form" onSubmit={submit}>
        <label>
          Email
          <input type="email" autoComplete="username" value={email} required
                 onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Password
          <input type="password" autoComplete="current-password" value={password} required
                 onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <div className="upload-error" role="alert">{error}</div>}
        <button className="upload-btn" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </Screen>
  );
}

function Screen({ title, children, onSignOut, email }: {
  title: string;
  children?: ReactNode;
  onSignOut?: () => void;
  email?: string | null;
}) {
  return (
    <div className="auth-screen">
      <div className="auth-card">
        <h1>{title}</h1>
        {children}
        {onSignOut && (
          <p className="auth-signed-in">
            Signed in as {email}.{" "}
            <button className="link-btn" onClick={onSignOut}>Sign out</button>
          </p>
        )}
      </div>
    </div>
  );
}
