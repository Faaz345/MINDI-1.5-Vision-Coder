import { useCallback, useRef, useState } from "react";
import { MindiApiError, MindiEvent, WorkflowRequest, streamWorkflow } from "@/lib/mindiApi";

export type AgentTimelineEvent = MindiEvent & {
  id: string;
  createdAt: number;
};

export function useAgentStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [events, setEvents] = useState<AgentTimelineEvent[]>([]);
  const [fileDeltas, setFileDeltas] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setStreamText("");
    setEvents([]);
    setFileDeltas([]);
    setError(null);
  }, []);

  const cancelWorkflow = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const pushEvent = useCallback((event: MindiEvent) => {
    setEvents((current) => [
      ...current,
      {
        ...event,
        id: `${event.event}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        createdAt: Date.now(),
      },
    ].slice(-80));
  }, []);

  const startWorkflow = useCallback(
    async (
      request: WorkflowRequest,
      handlers: {
        onToken?: (text: string) => void;
        onFileDelta?: (delta: Record<string, unknown>) => void;
        onDone?: (data: Record<string, unknown>) => void;
      } = {}
    ) => {
      cancelWorkflow();
      reset();
      const controller = new AbortController();
      controllerRef.current = controller;
      setIsStreaming(true);

      try {
        await streamWorkflow(
          request,
          {
            onEvent: pushEvent,
            token: (data) => {
              const text = String(data.text ?? "");
              setStreamText((current) => current + text);
              handlers.onToken?.(text);
            },
            file_delta: (data) => {
              setFileDeltas((current) => [...current, data]);
              handlers.onFileDelta?.(data);
            },
            done: (data) => {
              handlers.onDone?.(data);
            },
            error: (data) => {
              setError(String(data.message ?? "Workflow failed"));
            },
          },
          controller.signal
        );
      } catch (cause) {
        if ((cause as Error).name !== "AbortError") {
          const message = cause instanceof MindiApiError ? cause.message : "Workflow request failed";
          setError(message);
          throw cause;
        }
      } finally {
        controllerRef.current = null;
        setIsStreaming(false);
      }
    },
    [cancelWorkflow, pushEvent, reset]
  );

  return {
    isStreaming,
    streamText,
    events,
    fileDeltas,
    error,
    startWorkflow,
    cancelWorkflow,
    reset,
  };
}
