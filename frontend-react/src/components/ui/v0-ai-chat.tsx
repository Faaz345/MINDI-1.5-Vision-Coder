"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Code,
  Info,
  Link,
  Loader2,
  Menu,
  Mic,
  Paperclip,
  Send,
} from "lucide-react";

import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

interface UseAutoResizeTextareaProps {
  minHeight: number;
  maxHeight?: number;
}

interface VercelV0ChatProps {
  value?: string;
  onValueChange?: (value: string) => void;
  onSubmit?: () => void;
  busy?: boolean;
  placement?: "center" | "top";
  showHeading?: boolean;
  showActions?: boolean;
  onMoreMenuChange?: (open: boolean) => void;
}

function useAutoResizeTextarea({ minHeight, maxHeight }: UseAutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      if (reset) {
        textarea.style.height = `${minHeight}px`;
        return;
      }

      textarea.style.height = `${minHeight}px`;

      const newHeight = Math.max(
        minHeight,
        Math.min(textarea.scrollHeight, maxHeight ?? Number.POSITIVE_INFINITY)
      );

      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = `${minHeight}px`;
    }
  }, [minHeight]);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [adjustHeight]);

  return { textareaRef, adjustHeight };
}

export function VercelV0Chat({
  value: controlledValue,
  onValueChange,
  onSubmit,
  busy = false,
  placement = "center",
  showHeading = false,
  showActions = true,
  onMoreMenuChange,
}: VercelV0ChatProps) {
  const [internalValue, setInternalValue] = useState("");
  const isControlled = controlledValue !== undefined;
  const value = controlledValue ?? internalValue;
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const maxChars = 2000;
  const charCount = value.length;
  const compact = placement === "top";
  const promptPresets = [
    {
      label: "Website",
      prompt: "Create a production-ready website with a clear hero, responsive sections, polished typography, and deploy-ready HTML, CSS, and JavaScript.",
    },
    {
      label: "Dashboard",
      prompt: "Build a focused SaaS dashboard with sidebar navigation, metrics, filters, tables, charts, and a clean responsive layout.",
    },
    {
      label: "App UI",
      prompt: "Generate a modern app interface with navigation, primary workflow screens, empty states, forms, and responsive interaction states.",
    },
    {
      label: "Component",
      prompt: "Create a reusable UI component with states, accessibility, responsive behavior, and clean production styling.",
    },
    {
      label: "Wireframe",
      prompt: "Create a structured wireframe for a web product with layout hierarchy, sections, content blocks, and clear user flow.",
    },
  ];
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: compact ? 42 : 68,
    maxHeight: compact ? 120 : 170,
  });

  function updateValue(nextValue: string) {
    const safeValue = nextValue.slice(0, maxChars);

    if (!isControlled) {
      setInternalValue(safeValue);
    }

    onValueChange?.(safeValue);
  }

  function submitPrompt() {
    if (!value.trim() || busy) return;

    onSubmit?.();

    if (!isControlled) {
      setInternalValue("");
      adjustHeight(true);
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitPrompt();
    }
  };

  useEffect(() => {
    adjustHeight(!value);
  }, [adjustHeight, value]);

  useEffect(() => {
    onMoreMenuChange?.(moreMenuOpen && !compact);
  }, [compact, moreMenuOpen, onMoreMenuChange]);

  useEffect(() => {
    if (!moreMenuOpen) return undefined;

    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setMoreMenuOpen(false);
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [moreMenuOpen]);

  return (
    <div
      className={cn(
        "flex w-full flex-col items-center mx-auto",
        compact ? "max-w-none gap-3" : "max-w-4xl gap-8 p-0"
      )}
    >
      {showHeading && (
        <h1 className="text-4xl font-bold text-white">What can I help you ship?</h1>
      )}

      <div className="w-full">
        <div
          className={cn(
            "mindi-prompt-card relative flex flex-col overflow-visible rounded-3xl border border-zinc-500/50 shadow-2xl",
            compact && "rounded-2xl shadow-none"
          )}
          data-more-menu-open={moreMenuOpen && !compact ? "true" : undefined}
        >
          {compact && (
            <div className="flex items-center justify-end px-4 pt-2.5">
              <span className="rounded-2xl bg-emerald-400/10 px-2 py-1 text-xs font-medium text-emerald-100 ring-1 ring-emerald-300/15">
                MINDIGENOUS
              </span>
            </div>
          )}

          <div className={cn("relative overflow-hidden", !compact && "pt-3")}>
            <Textarea
              ref={textareaRef}
              value={value}
              onChange={(event) => {
                updateValue(event.target.value);
                adjustHeight();
              }}
              onKeyDown={handleKeyDown}
              maxLength={maxChars}
              rows={compact ? 2 : 2}
              placeholder={
                compact
                  ? "Ask MINDIGENOUS..."
                  : "What would you like to explore today? Ask anything, share ideas, or request assistance..."
              }
              className={cn(
                "w-full resize-none border-none bg-transparent px-6 py-3 text-base font-normal leading-relaxed text-zinc-100 outline-none",
                "focus-visible:ring-0 focus-visible:ring-offset-0",
                "placeholder:text-zinc-500",
                compact ? "min-h-[42px] px-4 py-2 text-sm" : "min-h-[68px] py-2.5"
              )}
              style={{
                overflow: "hidden",
                scrollbarWidth: "none",
                msOverflowStyle: "none",
              }}
            />
            <div className="mindi-textarea-shade pointer-events-none absolute inset-0 bg-gradient-to-t from-zinc-800/5 to-transparent" />
          </div>

          <div className={cn("relative px-4 pb-3", compact && "pb-2.5")}>
            <div className="mindi-prompt-control-row flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <div className="mindi-tool-cluster flex items-center gap-1.5 rounded-2xl border border-zinc-700/50 bg-zinc-800/40 p-1">
                  <PromptToolButton
                    icon={<Paperclip className="size-4" />}
                    label="Upload files"
                    tiltClassName="hover:-rotate-3"
                    iconTiltClassName="group-hover:-rotate-12"
                  />
                  <PromptToolButton
                    icon={<Link className="size-4" />}
                    label="Web link"
                    hoverClassName="hover:text-red-400"
                    tiltClassName="hover:rotate-6"
                    iconTiltClassName="group-hover:rotate-12"
                  />
                  <PromptToolButton
                    icon={<Code className="size-4" />}
                    label="Code repo"
                    hoverClassName="hover:text-green-400"
                    tiltClassName="hover:rotate-3"
                    iconTiltClassName="group-hover:-rotate-6"
                  />
                  <PromptToolButton
                    icon={<FigmaGlyph />}
                    label="Design file"
                    hoverClassName="hover:text-purple-400"
                    tiltClassName="hover:-rotate-6"
                    iconTiltClassName="group-hover:rotate-12"
                  />
                </div>

                <PromptToolButton
                  icon={<Mic className="size-4" />}
                  label="Voice input"
                  className="border border-zinc-700/30"
                  hoverClassName="hover:border-red-500/30 hover:text-red-400"
                  tiltClassName="hover:rotate-2"
                  iconTiltClassName="group-hover:-rotate-3"
                />

                {!compact && (
                  <button
                    type="button"
                    onClick={() => setMoreMenuOpen((open) => !open)}
                    className={cn(
                      "group relative rounded-2xl border border-zinc-700/30 bg-transparent p-2.5 text-zinc-500 transition-all duration-300 hover:scale-105 hover:border-zinc-500 hover:bg-zinc-800/80 hover:text-zinc-200",
                      "mindi-menu-button",
                      moreMenuOpen && "border-zinc-500 bg-zinc-800/80 text-zinc-200"
                    )}
                    aria-label="Open prompt menu"
                    aria-expanded={moreMenuOpen}
                  >
                    <Menu className="size-4 transition-transform duration-300 group-hover:scale-110" />
                  </button>
                )}
              </div>

              <div className="mindi-send-group flex items-center gap-2">
                <div
                  className={cn(
                    "mindi-char-count text-xs font-medium text-zinc-500",
                    charCount > maxChars * 0.9 && "text-red-300"
                  )}
                >
                  <span>{charCount}</span>/<span className="text-zinc-400">{maxChars}</span>
                </div>

                <button
                  type="button"
                  onClick={submitPrompt}
                  disabled={!value.trim() || busy}
                  className={cn(
                    "mindi-send-button group relative transition-colors duration-150",
                    "disabled:cursor-not-allowed"
                  )}
                  aria-label="Send prompt"
                >
                  {busy ? (
                    <Loader2 className="mindi-send-icon relative z-10 size-4 animate-spin" />
                  ) : (
                    <Send className="mindi-send-icon relative z-10 size-4 transition-transform duration-200 group-hover:translate-x-px" />
                  )}
                </button>
              </div>
            </div>

            <AnimatePresence initial={false}>
              {moreMenuOpen && !compact && (
                <motion.div
                  initial={{ y: -8, opacity: 0, scaleY: 0.96 }}
                  animate={{ y: 0, opacity: 1, scaleY: 1 }}
                  exit={{ y: -8, opacity: 0, scaleY: 0.96 }}
                  transition={{ duration: 0.22, ease: "easeOut" }}
                  className="mindi-prompt-presets-shell"
                >
                  <div className="mindi-prompt-presets mt-3 grid grid-cols-5 gap-2 pb-1">
                    {promptPresets.map((item) => (
                      <button
                        key={item.label}
                        type="button"
                        className="mindi-preset-button min-h-8 rounded-lg border border-zinc-700/45 bg-zinc-900/35 px-2 text-xs font-medium text-zinc-400 transition-colors hover:border-zinc-500 hover:bg-zinc-800/65 hover:text-zinc-100"
                        onClick={() => {
                          updateValue(item.prompt);
                          setMoreMenuOpen(false);
                          window.requestAnimationFrame(() => textareaRef.current?.focus());
                        }}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {showActions && !compact && (
              <div className="mindi-prompt-hint-row mt-2 flex items-center justify-start gap-6 pt-1 text-xs text-zinc-500">
                <div className="flex items-center gap-2">
                  <Info className="size-3" />
                  <span>
                    Press{" "}
                    <kbd className="rounded border border-zinc-600 bg-zinc-800 px-1.5 py-1 font-mono text-xs text-zinc-400 shadow-sm">
                      Shift + Enter
                    </kbd>{" "}
                    for new line
                  </span>
                </div>
              </div>
            )}
          </div>
          <div className="mindi-prompt-lightwash pointer-events-none absolute inset-0 rounded-3xl bg-[linear-gradient(135deg,rgba(73,255,176,0.06),transparent_45%,rgba(67,238,255,0.065))]" />
        </div>
      </div>
    </div>
  );
}

interface PromptToolButtonProps {
  icon: ReactNode;
  label: string;
  className?: string;
  hoverClassName?: string;
  tiltClassName?: string;
  iconTiltClassName?: string;
}

function PromptToolButton({
  icon,
  label,
  className,
  hoverClassName = "hover:text-zinc-200",
  tiltClassName,
  iconTiltClassName,
}: PromptToolButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        "mindi-tool-button group relative rounded-2xl bg-transparent p-2.5 text-zinc-500 transition-all duration-300 will-change-transform hover:scale-105 hover:bg-zinc-800/80",
        hoverClassName,
        tiltClassName,
        className
      )}
      aria-label={label}
    >
      <span className={cn("block transition-transform duration-300 will-change-transform group-hover:scale-125", iconTiltClassName)}>
        {icon}
      </span>
      <span className="pointer-events-none absolute -top-10 left-1/2 z-20 -translate-x-1/2 whitespace-nowrap rounded-lg border border-zinc-700/50 bg-zinc-900/95 px-3 py-2 text-xs text-zinc-200 opacity-0 shadow-lg backdrop-blur-sm transition-all duration-300 group-hover:-translate-y-1 group-hover:opacity-100">
        {label}
        <span className="absolute left-1/2 top-full h-0 w-0 -translate-x-1/2 border-x-4 border-t-4 border-x-transparent border-t-zinc-900/95" />
      </span>
    </button>
  );
}

function FigmaGlyph() {
  return (
    <svg className="size-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M15.852 8.981h-4.588V0h4.588c2.476 0 4.49 2.014 4.49 4.49s-2.014 4.491-4.49 4.491zM12.735 7.51h3.117c1.665 0 3.019-1.355 3.019-3.019s-1.354-3.019-3.019-3.019h-3.117V7.51zm0 1.471H8.148c-2.476 0-4.49-2.015-4.49-4.49S5.672 0 8.148 0h4.588v8.981zm-4.587-7.51c-1.665 0-3.019 1.355-3.019 3.019s1.354 3.02 3.019 3.02h3.117V1.471H8.148zm4.587 15.019H8.148c-2.476 0-4.49-2.014-4.49-4.49s2.014-4.49 4.49-4.49h4.588v8.98zM8.148 8.981c-1.665 0-3.019 1.355-3.019 3.019s1.355 3.019 3.019 3.019h3.117v-6.038H8.148zm7.704 0c-2.476 0-4.49 2.015-4.49 4.49s2.014 4.49 4.49 4.49 4.49-2.015 4.49-4.49-2.014-4.49-4.49-4.49zm0 7.509c-1.665 0-3.019-1.355-3.019-3.019s1.355-3.019 3.019-3.019 3.019 1.354 3.019 3.019-1.354 3.019-3.019 3.019zM8.148 24c-2.476 0-4.49-2.015-4.49-4.49s2.014-4.49 4.49-4.49h4.588V24H8.148zm3.117-1.471V16.49H8.148c-1.665 0-3.019 1.355-3.019 3.019s1.355 3.02 3.019 3.02h3.117z" />
    </svg>
  );
}
