export type MindiStreamEvent =
  | "meta"
  | "token"
  | "tool_start"
  | "tool_result"
  | "file_delta"
  | "log"
  | "done"
  | "error";

export type MindiEvent = {
  event: MindiStreamEvent;
  data: Record<string, unknown>;
};

export type MindiStreamHandlers = Partial<Record<MindiStreamEvent, (data: Record<string, unknown>) => void>> & {
  onEvent?: (event: MindiEvent) => void;
};

export type WorkflowRequest = {
  prompt: string;
  project_id?: string | null;
  files?: Record<string, string>;
  design_settings?: Record<string, unknown>;
  mode?: "chat" | "workspace";
};

export type ChatRequest = {
  messages: Array<{ role: "system" | "user" | "assistant" | "tool"; content: string }>;
  project_id?: string | null;
  files?: Record<string, string>;
  stream?: boolean;
};

export type WebSearchRequest = {
  query: string;
  sources?: Array<"web" | "docs" | "github" | "stackoverflow">;
  max_results?: number;
};

export class MindiApiError extends Error {
  code: string;

  constructor(message: string, code = "MINDI_API_ERROR") {
    super(message);
    this.name = "MindiApiError";
    this.code = code;
  }
}

export async function streamWorkflow(
  request: WorkflowRequest,
  handlers: MindiStreamHandlers = {},
  signal?: AbortSignal
) {
  await streamSsePost("/api/workflow", request, handlers, signal);
}

export async function streamChat(
  request: ChatRequest,
  handlers: MindiStreamHandlers = {},
  signal?: AbortSignal
) {
  await streamSsePost("/api/chat", request, handlers, signal);
}

export async function webSearch(request: WebSearchRequest, signal?: AbortSignal) {
  const response = await fetch("/api/web-search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new MindiApiError(`Web search failed with ${response.status}`, "WEB_SEARCH_FAILED");
  }

  return response.json();
}

async function streamSsePost(
  url: string,
  body: unknown,
  handlers: MindiStreamHandlers,
  signal?: AbortSignal
) {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new MindiApiError("MINDI backend is unavailable", "BACKEND_UNAVAILABLE");
  }

  if (!response.ok || !response.body) {
    throw new MindiApiError(`MINDI backend returned ${response.status}`, "BACKEND_ERROR");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split(/\n\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      dispatchFrame(frame, handlers);
    }
  }

  if (buffer.trim()) {
    dispatchFrame(buffer, handlers);
  }
}

function dispatchFrame(frame: string, handlers: MindiStreamHandlers) {
  const lines = frame.split(/\r?\n/);
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  const event = (eventLine?.replace("event:", "").trim() || "log") as MindiStreamEvent;
  const rawData = dataLines.map((line) => line.replace("data:", "").trim()).join("\n");
  let data: Record<string, unknown> = {};

  if (rawData) {
    try {
      data = JSON.parse(rawData);
    } catch {
      data = { text: rawData };
    }
  }

  const payload = { event, data };
  handlers.onEvent?.(payload);
  handlers[event]?.(data);
}
