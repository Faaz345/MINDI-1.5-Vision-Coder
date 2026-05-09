import { KeyboardEvent, MouseEvent, useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useAuthModal } from "@/hooks/useAuthModal";
import { AuthForm } from "./AuthForm";

const focusableSelector = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export function AuthModal() {
  const { isAuthOpen, mode, setAuthMode, closeAuth } = useAuthModal();
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<Element | null>(null);

  useEffect(() => {
    if (!isAuthOpen) return;

    previousFocusRef.current = document.activeElement;
    const frame = window.requestAnimationFrame(() => {
      const firstInput =
        modalRef.current?.querySelector<HTMLElement>("[data-auth-autofocus]") ??
        modalRef.current?.querySelector<HTMLElement>(focusableSelector);
      firstInput?.focus();
    });

    return () => {
      window.cancelAnimationFrame(frame);
      if (previousFocusRef.current instanceof HTMLElement) {
        previousFocusRef.current.focus();
      }
    };
  }, [isAuthOpen, mode]);

  useEffect(() => {
    if (!isAuthOpen) return;

    function handleEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        closeAuth();
      }
    }

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [closeAuth, isAuthOpen]);

  function handleBackdropClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      closeAuth();
    }
  }

  function handleModalKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Tab" || !modalRef.current) return;

    const focusable = Array.from(
      modalRef.current.querySelectorAll<HTMLElement>(focusableSelector),
    ).filter((element) => !element.hasAttribute("disabled"));

    if (!focusable.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
      return;
    }

    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <AnimatePresence>
      {isAuthOpen && (
        <motion.div
          className="auth-modal-backdrop"
          onMouseDown={handleBackdropClick}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.24, ease: "easeOut" }}
        >
          <div className="auth-circuit auth-circuit-left" aria-hidden="true">
            <span className="auth-chip auth-chip-top" />
            <span className="auth-chip auth-chip-bottom" />
          </div>
          <div className="auth-circuit auth-circuit-right" aria-hidden="true">
            <span className="auth-chip auth-chip-top" />
            <span className="auth-chip auth-chip-bottom" />
          </div>
          <motion.div
            ref={modalRef}
            className="auth-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="auth-modal-title"
            onKeyDown={handleModalKeyDown}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            <button className="auth-close" type="button" onClick={closeAuth} aria-label="Close authentication modal">
              <X size={24} />
            </button>
            <AuthForm mode={mode} onModeChange={setAuthMode} onDone={closeAuth} />
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
