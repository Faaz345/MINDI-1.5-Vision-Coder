import { useCallback, useSyncExternalStore } from "react";

export type AuthMode = "signin" | "signup";

type AuthModalState = {
  isAuthOpen: boolean;
  mode: AuthMode;
};

let authState: AuthModalState = {
  isAuthOpen: false,
  mode: "signin",
};

const listeners = new Set<() => void>();

function emit(nextState: Partial<AuthModalState>) {
  authState = { ...authState, ...nextState };
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot() {
  return authState;
}

export function useAuthModal() {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);

  const openAuth = useCallback((mode: AuthMode = "signin") => {
    emit({ isAuthOpen: true, mode });
  }, []);

  const closeAuth = useCallback(() => {
    emit({ isAuthOpen: false });
  }, []);

  const setAuthMode = useCallback((mode: AuthMode) => {
    emit({ mode });
  }, []);

  return {
    ...state,
    openAuth,
    closeAuth,
    setAuthMode,
  };
}
