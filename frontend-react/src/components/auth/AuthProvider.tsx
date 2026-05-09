import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import type { Provider, Session, User } from "@supabase/supabase-js";
import { isSupabaseConfigured, supabase } from "@/lib/supabase";

type SignUpPayload = {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  password: string;
};

type AuthActionResult = {
  ok: boolean;
  message: string;
  shouldClose?: boolean;
};

type AuthSessionContextValue = {
  session: Session | null;
  user: User | null;
  loading: boolean;
  isConfigured: boolean;
  signInWithPassword: (email: string, password: string) => Promise<AuthActionResult>;
  signUpWithPassword: (payload: SignUpPayload) => Promise<AuthActionResult>;
  signInWithOAuth: (provider: "google" | "github" | "huggingface") => Promise<AuthActionResult>;
  signOut: () => Promise<AuthActionResult>;
};

const AuthSessionContext = createContext<AuthSessionContextValue | null>(null);

function missingConfigResult(): AuthActionResult {
  return {
    ok: false,
    message: "Supabase is not configured. Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to your environment.",
  };
}

function messageFromError(error: unknown) {
  return error instanceof Error ? error.message : "Authentication failed. Please try again.";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(isSupabaseConfigured);

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return undefined;
    }

    let mounted = true;

    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session);
      setLoading(false);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setLoading(false);
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const value = useMemo<AuthSessionContextValue>(() => ({
    session,
    user: session?.user ?? null,
    loading,
    isConfigured: isSupabaseConfigured,
    async signInWithPassword(email, password) {
      if (!supabase) return missingConfigResult();

      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) return { ok: false, message: error.message };

      return { ok: true, message: "Signed in.", shouldClose: true };
    },
    async signUpWithPassword(payload) {
      if (!supabase) return missingConfigResult();

      const fullName = [payload.firstName, payload.lastName].filter(Boolean).join(" ").trim();
      const { data, error } = await supabase.auth.signUp({
        email: payload.email,
        password: payload.password,
        options: {
          data: {
            first_name: payload.firstName,
            last_name: payload.lastName,
            full_name: fullName,
            phone: payload.phone,
          },
        },
      });

      if (error) return { ok: false, message: error.message };

      return {
        ok: true,
        message: data.session ? "Account created." : "Check your email to confirm your account.",
        shouldClose: Boolean(data.session),
      };
    },
    async signInWithOAuth(provider) {
      if (!supabase) return missingConfigResult();

      const { error } = await supabase.auth.signInWithOAuth({
        provider: provider as Provider,
        options: {
          redirectTo: window.location.origin,
        },
      });

      if (error) return { ok: false, message: error.message };
      return { ok: true, message: `Redirecting to ${provider}.` };
    },
    async signOut() {
      if (!supabase) return missingConfigResult();

      try {
        const { error } = await supabase.auth.signOut();
        if (error) return { ok: false, message: error.message };
        return { ok: true, message: "Signed out." };
      } catch (error) {
        return { ok: false, message: messageFromError(error) };
      }
    },
  }), [loading, session]);

  return (
    <AuthSessionContext.Provider value={value}>
      {children}
    </AuthSessionContext.Provider>
  );
}

export function useAuthSession() {
  const context = useContext(AuthSessionContext);
  if (!context) {
    throw new Error("useAuthSession must be used inside AuthProvider");
  }

  return context;
}
