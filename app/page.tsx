"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type RiskLevel = "Low" | "Medium" | "High";

type MessageMeta = {
  riskLevel?: RiskLevel;
  sourcesUsed?: string[];
  fromCache?: boolean;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
  meta?: MessageMeta;
  pending?: boolean;
};

type HistoryResponse = {
  messages: ChatMessage[];
};

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi, I am your AI Health Assistant. Tell me what you are feeling, how long it has been happening, and anything important like fever, pain, medications, or medical conditions. I will answer in a chat format and flag urgent situations clearly.",
};

const RISK_STYLES: Record<RiskLevel, string> = {
  Low: "bg-emerald-100 text-emerald-800 border-emerald-200",
  Medium: "bg-amber-100 text-amber-800 border-amber-200",
  High: "bg-rose-100 text-rose-800 border-rose-200",
};

function formatTime(timestamp?: string) {
  if (!timestamp) {
    return "";
  }

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function generateSafeId() {
  const cryptoObject =
    typeof globalThis !== "undefined" ? globalThis.crypto : undefined;

  if (cryptoObject?.randomUUID) {
    return cryptoObject.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function createMessageId(prefix: string) {
  return `${prefix}-${generateSafeId()}`;
}

function SourcePills({ sources }: { sources: string[] }) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2 pt-3">
      {sources.map((source) => (
        <span
          key={source}
          className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-600"
        >
          {source}
        </span>
      ))}
    </div>
  );
}

function AssistantBubble({ message }: { message: ChatMessage }) {
  const riskLevel = message.meta?.riskLevel;
  const sourcesUsed = message.meta?.sourcesUsed ?? [];

  return (
    <div className="flex justify-start">
      <div className="max-w-[88%] rounded-[24px] rounded-tl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <div className="whitespace-pre-wrap text-sm leading-7 text-slate-700">
          {message.content}
          {message.pending && (
            <span className="ml-1 inline-block h-4 w-2 animate-pulse rounded-full bg-sky-400 align-middle" />
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
          {riskLevel && (
            <span className={`rounded-full border px-2.5 py-1 font-semibold ${RISK_STYLES[riskLevel]}`}>
              {riskLevel} risk
            </span>
          )}
          {message.meta?.fromCache && (
            <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2.5 py-1 font-semibold text-cyan-700">
              Reused from chat history
            </span>
          )}
          {message.createdAt && <span>{formatTime(message.createdAt)}</span>}
        </div>

        <SourcePills sources={sourcesUsed} />
      </div>
    </div>
  );
}

function UserBubble({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[82%] rounded-[24px] rounded-tr-md bg-slate-900 px-4 py-3 text-sm leading-7 text-white shadow-sm">
        <div className="whitespace-pre-wrap">{message.content}</div>
        {message.createdAt && (
          <div className="mt-2 text-[11px] text-slate-300">{formatTime(message.createdAt)}</div>
        )}
      </div>
    </div>
  );
}

export default function Home() {
  const [userId] = useState<string>(() => {
    if (typeof globalThis.window === "undefined") return "";
    const key = "health_advisor_user_id";
    const existing = localStorage.getItem(key);
    if (existing) return existing;
    const newId = `user_${generateSafeId()}`;
    localStorage.setItem(key, newId);
    return newId;
  });
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusText]);

  useEffect(() => {
    if (!userId) {
      return;
    }

    let cancelled = false;

    void (async () => {
      try {
        const response = await fetch(`/api/analyze?userId=${encodeURIComponent(userId)}`, {
          cache: "no-store",
        });
        const data = (await response.json()) as HistoryResponse;

        if (cancelled || !Array.isArray(data.messages) || data.messages.length === 0) {
          return;
        }

        setMessages(data.messages);
      } catch {
        // Keep the welcome state if loading old messages fails.
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [userId]);

  const sendableHistory = useMemo(
    () =>
      messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-8)
        .map((message) => ({
          role: message.role,
          content: message.content,
        })),
    [messages]
  );

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    const trimmed = input.trim();
    if (!trimmed || loading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: createMessageId("user"),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    };
    const assistantId = createMessageId("assistant");
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      pending: true,
    };

    setMessages((previous) => [...previous, userMessage, assistantPlaceholder]);
    setInput("");
    setLoading(true);
    setError("");
    setStatusText("Checking previous conversations...");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          userId,
          chat_history: sendableHistory,
        }),
      });

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({ error: "Unable to chat right now." }));
        throw new Error(data.error || "Unable to chat right now.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const updateAssistant = (updater: (current: ChatMessage) => ChatMessage) => {
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId ? updater(message) : message
          )
        );
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          if (!rawEvent.trim()) {
            continue;
          }

          const lines = rawEvent.split("\n");
          let eventName = "message";
          const dataLines: string[] = [];

          for (const line of lines) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }

          if (dataLines.length === 0) {
            continue;
          }

          const payload = JSON.parse(dataLines.join("\n")) as {
            text?: string;
            message?: string;
            riskLevel?: RiskLevel;
            sourcesUsed?: string[];
            fromCache?: boolean;
            createdAt?: string;
          };

          if (eventName === "status" && payload.message) {
            setStatusText(payload.message);
          }

          if (eventName === "chunk" && payload.text) {
            updateAssistant((current) => ({
              ...current,
              content: current.content + payload.text!,
            }));
          }

          if (eventName === "done") {
            setStatusText("");
            updateAssistant((current) => ({
              ...current,
              content: payload.message || current.content,
              pending: false,
              createdAt: payload.createdAt || new Date().toISOString(),
              meta: {
                riskLevel: payload.riskLevel,
                sourcesUsed: payload.sourcesUsed ?? [],
                fromCache: payload.fromCache,
              },
            }));
          }

          if (eventName === "error") {
            throw new Error(payload.message || "Unable to chat right now.");
          }
        }
      }

      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantId ? { ...message, pending: false } : message
        )
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Unable to chat right now.";

      setError(message);
      setStatusText("");
      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                pending: false,
                content:
                  "I could not finish that reply right now. Please try again in a moment.",
              }
            : message
        )
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#e0f2fe,transparent_32%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_100%)] px-4 py-8">
      <div className="mx-auto flex max-w-5xl flex-col gap-6">
        <section className="rounded-[32px] border border-slate-200 bg-white/80 p-6 shadow-[0_20px_80px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="max-w-2xl">
              <div className="mb-3 inline-flex rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-700">
                Conversational Health Assistant
              </div>
              <h1 className="text-4xl font-semibold tracking-tight text-slate-900">
                Chat about symptoms and get a safer, clearer next-step answer.
              </h1>
              <p className="mt-3 max-w-xl text-sm leading-7 text-slate-600">
                The assistant now checks your recent conversation history first,
                uses trusted medical references for fresh answers, and streams the
                reply in real time.
              </p>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs leading-6 text-slate-500">
              Replies are for general guidance only.
              <br />
              Urgent symptoms like chest pain, severe bleeding, fainting, or trouble
              breathing should get immediate medical care.
            </div>
          </div>
        </section>

        <section className="overflow-hidden rounded-[32px] border border-slate-200 bg-white shadow-[0_24px_100px_rgba(15,23,42,0.08)]">
          <div className="border-b border-slate-200 bg-slate-950 px-6 py-4 text-slate-50">
            <div className="text-sm font-semibold">Health Chat</div>
            <div className="mt-1 text-xs text-slate-300">
              Ask a question naturally, like you would in a normal chat.
            </div>
          </div>

          <div className="max-h-[62vh] overflow-y-auto bg-[linear-gradient(180deg,#f8fafc_0%,#ffffff_25%,#f8fafc_100%)] px-4 py-5 md:px-6">
            <div className="space-y-4">
              {messages.map((message) =>
                message.role === "assistant" ? (
                  <AssistantBubble key={message.id} message={message} />
                ) : (
                  <UserBubble key={message.id} message={message} />
                )
              )}
              <div ref={bottomRef} />
            </div>
          </div>

          <div className="border-t border-slate-200 bg-white px-4 py-4 md:px-6">
            {statusText && (
              <div className="mb-3 rounded-2xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-medium text-sky-700">
                {statusText}
              </div>
            )}

            {error && (
              <div className="mb-3 rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    e.currentTarget.form?.requestSubmit();
                  }
                }}
                rows={4}
                placeholder="Example: I have had a sore throat and low fever since last night. Should I worry?"
                className="w-full rounded-[24px] border border-slate-300 bg-slate-50 px-4 py-3 text-sm leading-7 text-slate-800 outline-none transition focus:border-sky-400 focus:bg-white focus:ring-4 focus:ring-sky-100"
              />

              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <p className="text-xs leading-6 text-slate-500">
                  Add symptom details, duration, fever, pain level, medications, or existing conditions for a better answer.
                </p>

                <button
                  type="submit"
                  // disabled={loading || !input.trim()}
                  className="inline-flex items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {loading ? "Streaming reply..." : "Send message"}
                </button>
              </div>
            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
