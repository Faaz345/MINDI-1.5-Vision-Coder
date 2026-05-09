import { FormEvent, useState } from "react";
import { Lock, Mail } from "lucide-react";
import type { AuthMode } from "@/hooks/useAuthModal";
import { useAuthSession } from "./AuthProvider";

type AuthFormProps = {
  mode: AuthMode;
  onModeChange: (mode: AuthMode) => void;
  onDone: () => void;
};

type NoticeState = {
  tone: "info" | "error" | "success";
  text: string;
} | null;

function GithubLogo() {
  return (
    <svg className="auth-brand-svg github" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 .5C5.65.5.5 5.65.5 12c0 5.1 3.29 9.42 7.86 10.95.58.1.79-.25.79-.56v-2.02c-3.2.7-3.88-1.38-3.88-1.38-.53-1.34-1.3-1.7-1.3-1.7-1.06-.72.08-.7.08-.7 1.17.08 1.79 1.2 1.79 1.2 1.04 1.78 2.73 1.27 3.4.97.1-.75.41-1.27.75-1.56-2.56-.29-5.26-1.28-5.26-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .98-.31 3.19 1.18a11.1 11.1 0 0 1 5.8 0c2.21-1.49 3.18-1.18 3.18-1.18.64 1.59.24 2.77.12 3.06.74.81 1.18 1.84 1.18 3.1 0 4.43-2.7 5.4-5.27 5.69.42.36.8 1.08.8 2.18v3.23c0 .31.21.67.8.56A11.52 11.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"
      />
    </svg>
  );
}

function GoogleLogo() {
  return (
    <svg className="auth-brand-svg google" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#4285F4" d="M23.49 12.27c0-.79-.07-1.54-.19-2.27H12v4.29h6.47c-.28 1.5-1.12 2.78-2.39 3.63v2.96h3.87c2.26-2.08 3.54-5.15 3.54-8.61Z" />
      <path fill="#34A853" d="M12 24c3.24 0 5.96-1.07 7.95-2.91l-3.87-2.96c-1.07.72-2.44 1.15-4.08 1.15-3.13 0-5.78-2.11-6.73-4.95H1.27v3.05C3.25 21.3 7.31 24 12 24Z" />
      <path fill="#FBBC05" d="M5.27 14.33A7.2 7.2 0 0 1 4.89 12c0-.81.14-1.59.38-2.33V6.62H1.27A11.94 11.94 0 0 0 0 12c0 1.93.46 3.75 1.27 5.38l4-3.05Z" />
      <path fill="#EA4335" d="M12 4.72c1.76 0 3.34.61 4.58 1.8l3.44-3.44C17.95 1.15 15.24 0 12 0 7.31 0 3.25 2.7 1.27 6.62l4 3.05C6.22 6.83 8.87 4.72 12 4.72Z" />
    </svg>
  );
}

function HuggingFaceLogo() {
  return (
    <svg className="auth-brand-svg huggingface" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12.7" r="7.1" fill="#FFD21E" />
      <circle cx="8.8" cy="11.5" r="0.95" fill="#111111" />
      <circle cx="15.2" cy="11.5" r="0.95" fill="#111111" />
      <path d="M8.8 15.2c1.8 1.55 4.6 1.55 6.4 0" fill="none" stroke="#111111" strokeLinecap="round" strokeWidth="1.25" />
      <circle cx="5.1" cy="9.4" r="2.1" fill="#FFD21E" />
      <circle cx="18.9" cy="9.4" r="2.1" fill="#FFD21E" />
      <path d="M5.2 8.45c.62-.3 1.2-.28 1.76.06M17.04 8.51c.56-.34 1.14-.36 1.76-.06" fill="none" stroke="#111111" strokeLinecap="round" strokeWidth="0.8" opacity="0.5" />
    </svg>
  );
}

export function AuthForm({ mode, onModeChange, onDone }: AuthFormProps) {
  const { isConfigured, signInWithPassword, signUpWithPassword, signInWithOAuth } = useAuthSession();
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<NoticeState>(null);
  const [termsOpen, setTermsOpen] = useState(false);
  const isSignup = mode === "signup";

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNotice(null);
    setLoading(true);

    const result = isSignup
      ? await signUpWithPassword({ firstName, lastName, email, phone, password })
      : await signInWithPassword(email, password);

    setLoading(false);
    setNotice({
      tone: result.ok ? "success" : "error",
      text: result.message,
    });

    if (result.ok && result.shouldClose) {
      window.setTimeout(onDone, 650);
    }
  }

  async function handleOAuth(provider: "google" | "github" | "huggingface") {
    setNotice(null);
    setLoading(true);
    const result = await signInWithOAuth(provider);
    setLoading(false);
    setNotice({
      tone: result.ok ? "info" : "error",
      text: result.message,
    });
  }

  return (
    <form className="auth-form" onSubmit={handleSubmit}>
      <div className="auth-logo-wrap">
        <span className="logo-shine-mask auth-logo-mask">
          <img className="auth-logo-image" src="/assets/mindigenous-white-logo.png" alt="MINDIGENOUS" />
        </span>
      </div>

      <header className="auth-copy">
        <h2 id="auth-modal-title">{isSignup ? "Create Account" : "Welcome Back"}</h2>
        <p>
          {isSignup ? "Already have an account?" : "Don't have an account yet?"}
          <button
            type="button"
            onClick={() => {
              onModeChange(isSignup ? "signin" : "signup");
              setNotice(null);
            }}
          >
            {isSignup ? "Sign in" : "Sign up"}
          </button>
        </p>
      </header>

      {!isConfigured && (
        <p className="auth-config-warning">
          Add Supabase environment keys to enable live authentication.
        </p>
      )}

      <div className="auth-fields">
        {isSignup && (
          <div className="auth-name-grid">
            <label>
              <span>First name</span>
              <div className="auth-input-wrap">
                <input
                  data-auth-autofocus
                  value={firstName}
                  onChange={(event) => setFirstName(event.target.value)}
                  autoComplete="given-name"
                  placeholder="John"
                  required
                />
              </div>
            </label>
            <label>
              <span>Last name</span>
              <div className="auth-input-wrap">
                <input
                  value={lastName}
                  onChange={(event) => setLastName(event.target.value)}
                  autoComplete="family-name"
                  placeholder="Last name"
                />
              </div>
            </label>
          </div>
        )}

        <label>
          <span>Email</span>
          <div className="auth-input-wrap">
            <Mail size={18} />
            <input
              data-auth-autofocus={!isSignup ? true : undefined}
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete={isSignup ? "email" : "username"}
              placeholder="email address"
              required
            />
          </div>
        </label>

        <label>
          <span>Password</span>
          <div className="auth-input-wrap">
            <Lock size={18} />
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete={isSignup ? "new-password" : "current-password"}
              placeholder="Password"
              minLength={6}
              required
            />
          </div>
        </label>

        {isSignup && (
          <label>
            <span>Phone</span>
            <div className="auth-input-wrap">
              <input
                type="tel"
                value={phone}
                onChange={(event) => setPhone(event.target.value)}
                autoComplete="tel"
                placeholder="Phone (optional)"
              />
            </div>
          </label>
        )}
      </div>

      {notice && <p className={`auth-notice ${notice.tone}`}>{notice.text}</p>}

      <button className="auth-primary" type="submit" disabled={loading}>
        {loading ? "Please wait..." : isSignup ? "Create account" : "Login"}
      </button>

      <div className="auth-divider">
        <span />
        <p>or</p>
        <span />
      </div>

      <div className="auth-oauth">
        <button type="button" onClick={() => handleOAuth("github")} disabled={loading} aria-label="Continue with GitHub">
          <GithubLogo />
        </button>
        <button type="button" onClick={() => handleOAuth("google")} disabled={loading} aria-label="Continue with Google">
          <GoogleLogo />
        </button>
        <button type="button" onClick={() => handleOAuth("huggingface")} disabled={loading} aria-label="Continue with Hugging Face">
          <HuggingFaceLogo />
        </button>
      </div>

      <p className="auth-terms">
        By continuing, you agree to our{" "}
        <button type="button" onClick={() => setTermsOpen((open) => !open)}>
          Terms &amp; Conditions
        </button>
      </p>

      {termsOpen && (
        <section className="auth-terms-panel" aria-label="MINDIGENOUS terms and conditions">
          <strong>MINDIGENOUS Terms &amp; Conditions</strong>
          <p>
            You are responsible for prompts, uploaded assets, generated code, and deployment decisions made in your workspace.
            MINDIGENOUS may process prompts, files, and project metadata through AI agents and cloud services to generate,
            preview, improve, and deploy websites.
          </p>
          <p>
            Do not use the service to create illegal, harmful, infringing, deceptive, or abusive content. Generated outputs
            should be reviewed before publishing, and production deployments remain your responsibility.
          </p>
        </section>
      )}
    </form>
  );
}
