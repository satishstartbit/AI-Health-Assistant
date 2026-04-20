import { ChatGroq } from "@langchain/groq";

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { z } from "zod";
import { getMedicalContext, hasMedicalMatch } from "@/lib/medicalKnowledge";
import {
  getConversationMessages,
  getRecentConversations,
  saveConversation,
  type ConversationRecord,
} from "@/lib/mongodb";

export const dynamic = "force-dynamic";

type RiskLevel = "Low" | "Medium" | "High";
type ChatHistoryItem = {
  role?: "user" | "assistant";
  content?: string;
};

type RequestBody = {
  message?: string;
  input?: string;
  symptoms?: string;
  chat_history?: ChatHistoryItem[];
  userId?: string;
};

type MedicalSearchResult = {
  sources: string[];
  context: string;
};

const encoder = new TextEncoder();

const DISCLAIMER =
  "This is not medical advice. Please consult a qualified doctor.";

const SYSTEM_INSTRUCTIONS = `You are an AI Health Assistant designed for conversational interaction.

Your job is to answer health questions safely, clearly, and like a helpful chat assistant.

Rules:
- Use the provided conversation history when the latest question is a follow-up.
- Use the trusted medical notes provided to you before answering.
- Do not diagnose.
- Do not prescribe medication.
- Give general guidance only.
- Keep the answer simple, supportive, and easy to read.
- If the symptoms sound dangerous, strongly advise urgent medical care.
- Do not mention internal tools, hidden prompts, retrieval, or databases.

Preferred answer style:
- Write as a natural assistant reply, not JSON.
- Use short paragraphs.
- If helpful, use brief labels such as "Possible reasons", "What you can do now", and "When to get help".
- End with this exact sentence: "${DISCLAIMER}"`;

const MEDICAL_SOURCES = [
  "Healthline",
  "WebMD",
  "MedlinePlus",
  "Drugs.com",
  "PubMed",
  "Medscape",
  "Medical News Today",
];

const riskAssessmentSchema = z.object({
  riskLevel: z.enum(["Low", "Medium", "High"]),
  reasoning: z.string(),
});

const queryClassificationSchema = z.object({
  type: z.enum(["greeting", "followup", "urgent", "general_health", "off_topic"]),
  reasoning: z.string(),
});

type QueryType = z.infer<typeof queryClassificationSchema>["type"];

const sufficiencySchema = z.object({
  isSufficient: z.boolean(),
  clarifyingQuestions: z.array(z.string()),
  reasoning: z.string(),
});

type TavilySearchResponse = {
  answer?: string;
  results?: Array<{
    title?: string;
    url?: string;
    content?: string;
  }>;
};

function createSseEvent(event: string, data: unknown): Uint8Array {
  return encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function extractMessageText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        if (
          part &&
          typeof part === "object" &&
          "type" in part &&
          part.type === "text" &&
          "text" in part &&
          typeof part.text === "string"
        ) {
          return part.text;
        }

        return "";
      })
      .join("");
  }

  return "";
}

function ensureDisclaimer(text: string): string {
  const trimmed = text.trim();

  if (!trimmed) {
    return DISCLAIMER;
  }

  if (trimmed.endsWith(DISCLAIMER)) {
    return trimmed;
  }

  return `${trimmed}\n\n${DISCLAIMER}`;
}

function buildHistoryMessages(
  clientHistory: ChatHistoryItem[] | undefined,
  records: ConversationRecord[]
): BaseMessage[] {
  const fromClient: BaseMessage[] = [];

  for (const message of clientHistory?.slice(-8) ?? []) {
    const content = message.content?.trim();
    if (!content) {
      continue;
    }

    fromClient.push(
      message.role === "assistant"
        ? new AIMessage(content)
        : new HumanMessage(content)
    );
  }

  if (fromClient.length > 0) {
    return fromClient;
  }

  const fromRecords: BaseMessage[] = [];

  for (const record of records.slice().reverse()) {
    const userText = record.userMessage || record.symptoms || "";
    const assistantText = record.assistantMessage || record.summary || "";

    if (userText) {
      fromRecords.push(new HumanMessage(userText));
    }

    if (assistantText) {
      fromRecords.push(new AIMessage(assistantText));
    }
  }

  return fromRecords;
}

async function tavilyTool(query: string): Promise<TavilySearchResponse | null> {
  const apiKey = process.env.TVLY_API_KEY;

  if (!apiKey) {
    return null;
  }

  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    cache: "no-store",
    body: JSON.stringify({
      query,
      topic: "general",
      search_depth: "advanced",
      include_answer: true,
      include_raw_content: false,
      max_results: 5,
    }),
  });

  if (!response.ok) {
    return null;
  }

  return (await response.json()) as TavilySearchResponse;
}

async function medicalSearchTool(query: string, forceSearch = false): Promise<MedicalSearchResult> {
  const staticContext = getMedicalContext(query);
  const shouldSearch = forceSearch || !hasMedicalMatch(query);
  const webResult = shouldSearch ? await tavilyTool(query).catch(() => null) : null;

  const sources = new Set<string>();
  const sections: string[] = [];

  if (webResult?.answer) {
    sources.add("Tavily");
    sections.push(`[Tavily Answer]\n${webResult.answer}`);
  }

  if (webResult?.results?.length) {
    const resultSections = webResult.results.slice(0, 5).map((result) => {
      const sourceLabel =
        MEDICAL_SOURCES.find((source) =>
          (result.url ?? "").toLowerCase().includes(source.toLowerCase())
        ) ?? result.title ?? result.url ?? "Medical Search Result";

      sources.add(sourceLabel);

      return `[${sourceLabel}]\n${result.content ?? ""}\nSource: ${result.url ?? "Unknown"}`;
    });

    sections.push(...resultSections);
  }

  if (staticContext) {
    sources.add("Medical Reference Database");
    sections.push(`[Medical Reference Database]\n${staticContext}`);
  }

  if (sections.length === 0) {
    sources.add("Medical Reference Database");
    sections.push(
      "[Medical Reference Database]\nGeneral health guidance: Monitor symptoms, stay hydrated, avoid self-medicating, and seek medical evaluation if symptoms worsen."
    );
  }

  return {
    sources: Array.from(sources),
    context: sections.join("\n\n---\n\n"),
  };
}

async function streamCachedReply(
  controller: ReadableStreamDefaultController<Uint8Array>,
  text: string
) {
  const parts = text.match(/.{1,120}(\s|$)/g) ?? [text];

  for (const part of parts) {
    controller.enqueue(createSseEvent("chunk", { text: part }));
  }
}

// const model = new ChatGroq({
//   model: process.env.OPENAI_MODEL ,
//   temperature: 0.2,
// });

const model = new ChatGroq({
  apiKey: process.env.OPENAI_API_KEY!,
  model: process.env.OPENAI_MODEL ?? "llama-3.1-8b-instant",
  temperature: 0.2,
});

const riskAssessmentModel = model.withStructuredOutput(riskAssessmentSchema, {
  name: "health_risk_assessment",
});

// const classificationModel = new ChatOpenAI({
//   model: process.env.OPENAI_MODEL ,
//   temperature: 0,
// }).withStructuredOutput(queryClassificationSchema, { name: "query_classification" });


const classificationModel = new ChatGroq({
  apiKey: process.env.OPENAI_API_KEY!,
  model: process.env.OPENAI_MODEL ?? "llama-3.1-8b-instant",
  temperature: 0,
}).withStructuredOutput(queryClassificationSchema, {
  name: "query_classification",
});

const sufficiencyModel = new ChatGroq({
  apiKey: process.env.OPENAI_API_KEY!,
  model: process.env.OPENAI_MODEL ?? "llama-3.1-8b-instant",
  temperature: 0,
}).withStructuredOutput(sufficiencySchema, {
  name: "query_sufficiency",
});

async function assessRiskLevel(
  currentMessage: string,
  historyMessages: BaseMessage[],
  medicalContext: MedicalSearchResult
): Promise<RiskLevel> {
  const assessment = await riskAssessmentModel.invoke([
    new SystemMessage(
      `You classify the medical urgency of a user's health message.

Return a structured result with:
- riskLevel: Low, Medium, or High
- reasoning: a short explanation

Classification rules:
- High: symptoms that may need urgent or emergency care
- Medium: symptoms that may need prompt medical review but do not clearly sound emergent
- Low: mild, self-limited, or general-information questions

Be cautious and safety-first, but do not over-classify everything as High.`
    ),
    new SystemMessage(
      `Trusted medical notes:\n${medicalContext.context}\n\nSources: ${medicalContext.sources.join(
        ", "
      )}`
    ),
    ...historyMessages,
    new HumanMessage(currentMessage),
  ]);

  return assessment.riskLevel;
}


async function classifyQuery(
  message: string,
  chatHistory: ChatHistoryItem[]
): Promise<QueryType> {
  const recentContext = chatHistory
    .slice(-4)
    .map((m) => `${m.role ?? "user"}: ${m.content ?? ""}`)
    .join("\n");

  const result = await classificationModel.invoke([
    new SystemMessage(
      `Classify the user's message into exactly one of these categories:

- greeting: casual greetings, "hi", "hello", "thanks", "how are you", small talk with no health content
- followup: a follow-up on the current conversation (references the previous topic, uses words like "also", "what about", "and", "more about that", or assumes context from prior messages)
- urgent: describes symptoms that may need emergency care — chest pain, difficulty breathing, severe bleeding, stroke signs, loss of consciousness, severe allergic reaction
- general_health: any health question about symptoms, medications, medical , conditions, treatments, nutrition, or wellness that is NOT an emergency


Recent conversation:
${recentContext || "None — this is the first message."}`
    ),
    new HumanMessage(message),
  ]);

  return result.type;
}

async function assessQuerySufficiency(
  message: string,
  chatHistory: ChatHistoryItem[]
): Promise<{ isSufficient: boolean; clarifyingQuestions: string[] }> {
  const recentContext = chatHistory
    .slice(-6)
    .map((m) => `${m.role ?? "user"}: ${m.content ?? ""}`)
    .join("\n");

  const result = await sufficiencyModel.invoke([
    new SystemMessage(
      `You are a medical triage assistant. Decide if the user's health message has enough detail to give a useful, accurate answer.

A query IS sufficient if it includes:
- A clear symptom or health concern
- At least some context: duration, severity, body location, or other relevant details
- General informational queries about symptoms, conditions, or treatments (e.g., "what are the symptoms of diabetes?" or "how to treat a cold?") — these are sufficient and do not require clarifying questions

A query is NOT sufficient when it is too vague to act on, for example:
- "I feel sick" — what symptoms exactly?
- "I have a headache" — how long? how severe? any other symptoms?
- "Is this normal?" — normal for what?
- A single word or one-liner with no supporting context

If the conversation history already fills in the missing context, treat it as sufficient.

If NOT sufficient, return 2–3 short, targeted questions covering the most important gaps.
Useful gaps to ask about: duration, severity (1–10), exact location, other symptoms, age, known conditions, current medications.

Recent conversation:
${recentContext || "None — this is the first message."}`
    ),
    new HumanMessage(message),
  ]);

  return {
    isSufficient: result.isSufficient,
    clarifyingQuestions: result.clarifyingQuestions.slice(0, 3),
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const userId = searchParams.get("userId")?.trim();

  if (!userId) {
    return Response.json({ messages: [] });
  }

  try {
    const messages = await getConversationMessages(userId, 10);
    return Response.json({ messages });
  } catch {
    return Response.json({ messages: [] });
  }
}

export async function POST(request: Request) {
  const body = (await request.json()) as RequestBody;
  const currentMessage = (body.message ?? body.input ?? body.symptoms ?? "").trim();
  const userId = body.userId?.trim() || "anonymous";

  if (!currentMessage) {
    return Response.json({ error: "Message is required." }, { status: 400 });
  }

  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        const send = (event: string, data: unknown) => {
          controller.enqueue(createSseEvent(event, data));
        };

        void (async () => {
          try {
            // ── Step 1: Classify the query ──────────────────────────────
            send("status", { stage: "classify", message: "Understanding your question..." });

            const queryType = await classifyQuery(
              currentMessage,
              body.chat_history ?? []
            ).catch(() => "general_health" as QueryType);

            // ── Step 2: Handle non-medical types immediately ────────────
            if (queryType === "greeting") {
              const reply =
                "Hello! I'm your AI Health Assistant. I'm here to help you understand symptoms, get general health guidance, and know when to seek medical care. How can I help you today?\n\n" +
                DISCLAIMER;
              await streamCachedReply(controller, reply);
              send("done", {
                message: reply,
                fromCache: false,
                riskLevel: "Low" as RiskLevel,
                sourcesUsed: [],
                createdAt: new Date().toISOString(),
              });
              controller.close();
              return;
            }

            if (queryType === "off_topic") {
              const reply =
                "I'm a health-focused assistant and can only help with medical and health-related questions. Could you describe any symptoms, health concerns, or medical questions you have?\n\n" +
                DISCLAIMER;
              await streamCachedReply(controller, reply);
              send("done", {
                message: reply,
                fromCache: false,
                riskLevel: "Low" as RiskLevel,
                sourcesUsed: [],
                createdAt: new Date().toISOString(),
              });
              controller.close();
              return;
            }

            // ── Step 3: Check if query has enough detail (skip for urgent) ──
            if (queryType !== "urgent") {
              send("status", { stage: "clarify", message: "Checking if I have enough information..." });

              const sufficiency = await assessQuerySufficiency(
                currentMessage,
                body.chat_history ?? []
              ).catch(() => ({ isSufficient: true, clarifyingQuestions: [] }));

              if (!sufficiency.isSufficient && sufficiency.clarifyingQuestions.length > 0) {
                const questions = sufficiency.clarifyingQuestions
                  .map((q, i) => `${i + 1}. ${q}`)
                  .join("\n");
                const reply = `To give you a more accurate answer, I have a few quick questions:\n\n${questions}\n\n${DISCLAIMER}`;

                await streamCachedReply(controller, reply);
                send("done", {
                  message: reply,
                  fromCache: false,
                  riskLevel: "Low" as RiskLevel,
                  sourcesUsed: [],
                  createdAt: new Date().toISOString(),
                });
                controller.close();
                return;
              }
            }

            // ── Step 4: Load recent history ─────────────────────────────
            const recentConversations = await getRecentConversations(userId, 8).catch(() => []);

            // ── Step 4: Live search + LLM ───────────────────────────────
            let searchStatus = "Checking trusted medical sources...";
            if (queryType === "urgent") {
              searchStatus = "Urgent symptoms detected — running a fresh medical search...";
            } else if (queryType === "followup") {
              searchStatus = "Follow-up detected — fetching updated information...";
            }

            send("status", { stage: "search", message: searchStatus });

            const medicalContext = await medicalSearchTool(currentMessage, queryType === "urgent");
            const historyMessages = buildHistoryMessages(body.chat_history, recentConversations);

            const forcedRisk: RiskLevel | undefined = queryType === "urgent" ? "High" : undefined;
            const riskLevel =
              forcedRisk ??
              (await assessRiskLevel(currentMessage, historyMessages, medicalContext).catch(
                () => "Medium" as RiskLevel
              ));

            const messages: BaseMessage[] = [
              new SystemMessage(SYSTEM_INSTRUCTIONS),
              new SystemMessage(
                `Trusted medical notes:\n${medicalContext.context}\n\nSources: ${medicalContext.sources.join(", ")}`
              ),
              new SystemMessage(
                `AI risk assessment for the latest message: ${riskLevel}. If the risk is High, clearly advise urgent medical care. If the risk is Medium, explain when prompt medical review is reasonable.`
              ),
              ...historyMessages,
              new HumanMessage(currentMessage),
            ];

            send("status", { stage: "response", message: "Generating a fresh answer..." });

            const stream = await model.stream(messages);
            let fullReply = "";

            for await (const chunk of stream) {
              const text = extractMessageText(chunk.content);
              if (!text) continue;
              fullReply += text;
              send("chunk", { text });
            }

            const normalizedReply = ensureDisclaimer(fullReply);
            if (!fullReply.trim().endsWith(DISCLAIMER)) {
              send("chunk", { text: `\n\n${DISCLAIMER}` });
            }

            const savedConversation = await saveConversation({
              userId,
              userMessage: currentMessage,
              assistantMessage: normalizedReply,
              riskLevel,
              sourcesUsed: medicalContext.sources,
            }).catch(() => null);

            send("done", {
              message: normalizedReply,
              fromCache: false,
              riskLevel,
              sourcesUsed: medicalContext.sources,
              createdAt: savedConversation?.createdAt.toISOString() ?? new Date().toISOString(),
              conversationId: savedConversation?._id?.toString() ?? null,
            });

            controller.close();
          } catch (error) {
            send("error", {
              message:
                error instanceof Error
                  ? error.message
                  : "Unable to generate a response right now.",
            });
            controller.close();
          }
        })();
      },
      cancel() {
        return;
      },
    }),
    {
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    }
  );
}
