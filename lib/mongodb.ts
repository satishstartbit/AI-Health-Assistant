import { MongoClient, Db, ObjectId } from "mongodb";
import { generateEmbedding } from "@/lib/embeddings";

const uri = process.env.DATABASE_URL!;

let client: MongoClient;
let db: Db;

type RiskLevel = "Low" | "Medium" | "High";
type ChatRole = "user" | "assistant";

export type ConversationMessage = {
  role: ChatRole;
  content: string;
};

export type EmbeddingPayload = number[];

export type ConversationRecord = {
  _id?: ObjectId;
  userId: string;
  userMessage: string;
  assistantMessage: string;
  riskLevel: RiskLevel;
  sourcesUsed: string[];
  messages: ConversationMessage[];
  embeddingText: string;
  embeddingPayload: EmbeddingPayload;
  queryText: string;
  searchText: string;
  symptoms?: string;
  risk_level?: RiskLevel;
  summary?: string;
  reuseCount?: number;
  createdAt: Date;
};

export type SavedConversationInput = {
  userId: string;
  userMessage: string;
  assistantMessage: string;
  riskLevel: RiskLevel;
  sourcesUsed: string[];
  messages?: ConversationMessage[];
};

export type RelevantConversationMatch = ConversationRecord & {
  score: number;
};

type FindRelevantConversationOptions = {
  userId?: string;
  limit?: number;
};

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "how",
  "i",
  "in",
  "is",
  "it",
  "me",
  "my",
  "of",
  "on",
  "or",
  "the",
  "to",
  "what",
  "when",
  "with",
  "you",
  "your",
]);

async function getDb(): Promise<Db> {
  if (!db) {
    client = new MongoClient(uri);
    await client.connect();
    db = client.db("health_advisor");
  }
  return db;
}

function normalizeSearchText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function generateSafeId() {
  const cryptoObject =
    typeof globalThis !== "undefined" ? globalThis.crypto : undefined;

  if (cryptoObject?.randomUUID) {
    return cryptoObject.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function extractKeywords(text: string): string[] {
  return Array.from(
    new Set(
      normalizeSearchText(text)
        .split(" ")
        .filter((token) => token.length > 2 && !STOP_WORDS.has(token))
    )
  );
}

function buildEmbeddingText(
  userMessage: string,
  assistantMessage: string,
  riskLevel: RiskLevel,
  sourcesUsed: string[]
): string {
  const sources = sourcesUsed.length > 0 ? sourcesUsed.join(", ") : "No explicit sources";
  return [
    `User: ${userMessage}`,
    `Assistant: ${assistantMessage}`,
    `Risk Level: ${riskLevel}`,
    `Sources: ${sources}`,
  ].join("\n");
}

function getUserText(record: ConversationRecord): string {
  return record.userMessage || record.symptoms || "";
}

function getAssistantText(record: ConversationRecord): string {
  return record.assistantMessage || record.summary || "";
}

function getStoredEmbedding(record: ConversationRecord): number[] {
  if (Array.isArray(record.embeddingPayload)) {
    return record.embeddingPayload;
  }

  const legacyPayload = record.embeddingPayload as
    | { vector?: number[] }
    | undefined;

  return Array.isArray(legacyPayload?.vector) ? legacyPayload.vector : [];
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    return 0;
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function calculateKeywordScore(query: string, record: ConversationRecord): number {
  const normalizedQuery = normalizeSearchText(query);
  const candidateText =
    record.searchText ||
    normalizeSearchText(`${getUserText(record)} ${getAssistantText(record)} ${record.embeddingText}`);

  if (!normalizedQuery || !candidateText) {
    return 0;
  }

  if (candidateText.includes(normalizedQuery)) {
    return 1;
  }

  const queryKeywords = extractKeywords(query);
  if (queryKeywords.length === 0) {
    return 0;
  }

  const candidateKeywords = new Set(extractKeywords(candidateText));
  const overlap = queryKeywords.filter((keyword) => candidateKeywords.has(keyword)).length;

  if (overlap === 0) {
    return 0;
  }

  const coverage = overlap / queryKeywords.length;
  const density = overlap / Math.max(candidateKeywords.size, 1);

  return coverage * 0.8 + density * 0.2;
}

export async function getRecentConversations(
  userId: string,
  limit = 8
): Promise<ConversationRecord[]> {
  const database = await getDb();
  return database
    .collection<ConversationRecord>("conversations")
    .find({ userId })
    .sort({ createdAt: -1 })
    .limit(limit)
    .toArray();
}

export async function findExactQuestionMatch(
  query: string
): Promise<RelevantConversationMatch | null> {
  const database = await getDb();
  const normalizedQuery = normalizeSearchText(query);

  if (!normalizedQuery) {
    return null;
  }

  const match = await database
    .collection<ConversationRecord>("conversations")
    .findOne(
      { queryText: normalizedQuery },
      { sort: { createdAt: -1 } }
    );

  if (!match) {
    const fallbackCandidates = await database
      .collection<ConversationRecord>("conversations")
      .find({})
      .sort({ createdAt: -1 })
      .limit(200)
      .toArray();

    const fallbackMatch = fallbackCandidates.find((candidate) => {
      const candidateQuery = normalizeSearchText(
        candidate.userMessage || candidate.symptoms || ""
      );
      return candidateQuery === normalizedQuery;
    });

    if (!fallbackMatch) {
      return null;
    }

    return {
      ...fallbackMatch,
      score: 1,
    };
  }

  return {
    ...match,
    score: 1,
  };
}

async function getConversationCandidates(
  options: FindRelevantConversationOptions
): Promise<ConversationRecord[]> {
  const database = await getDb();
  const { userId, limit = 30 } = options;

  return database
    .collection<ConversationRecord>("conversations")
    .find(userId ? { userId } : {})
    .sort({ createdAt: -1 })
    .limit(limit)
    .toArray();
}

export async function getConversationMessages(userId: string, limit = 8) {
  const records = await getRecentConversations(userId, limit);

  return records
    .slice()
    .reverse()
    .flatMap((record) => {
      const createdAt = record.createdAt instanceof Date
        ? record.createdAt.toISOString()
        : new Date(record.createdAt).toISOString();

      const storedMessages = Array.isArray(record.messages) && record.messages.length > 0
        ? record.messages
        : [
            {
              role: "user" as const,
              content: getUserText(record),
            },
            {
              role: "assistant" as const,
              content: getAssistantText(record),
            },
          ];

      return storedMessages.map((message, index) => ({
        id: `${record._id?.toString() ?? generateSafeId()}-${message.role}-${index}`,
        role: message.role,
        content: message.content,
        createdAt,
        ...(message.role === "assistant"
          ? {
              meta: {
                riskLevel: record.riskLevel || record.risk_level || "Medium",
                sourcesUsed: record.sourcesUsed ?? [],
                fromCache: false,
              },
            }
          : {}),
      }));
    });
}

export async function findRelevantConversation(
  query: string,
  options: FindRelevantConversationOptions = {}
): Promise<RelevantConversationMatch | null> {
  const [candidates, queryVector] = await Promise.all([
    getConversationCandidates(options),
    generateEmbedding(query).catch(() => []),
  ]);

  let bestMatch: RelevantConversationMatch | null = null;

  for (const candidate of candidates) {
    const candidateVector = getStoredEmbedding(candidate);
    const vectorScore = cosineSimilarity(queryVector, candidateVector);
    const keywordScore = calculateKeywordScore(query, candidate);
    const score =
      vectorScore > 0 ? vectorScore * 0.9 + keywordScore * 0.1 : keywordScore;

    if (!bestMatch || score > bestMatch.score) {
      bestMatch = { ...candidate, score };
    }
  }

  const threshold = options.userId
    ? queryVector.length > 0
      ? 0.78
      : 0.62
    : queryVector.length > 0
      ? 0.88
      : 0.82;

  if (!bestMatch || bestMatch.score < threshold) {
    return null;
  }

  return bestMatch;
}

export async function saveConversation(
  input: SavedConversationInput
): Promise<ConversationRecord> {
  const database = await getDb();
  const createdAt = new Date();
  const embeddingText = buildEmbeddingText(
    input.userMessage,
    input.assistantMessage,
    input.riskLevel,
    input.sourcesUsed
  );
  const embeddingVector = await generateEmbedding(input.userMessage).catch(() => []);
  const queryText = normalizeSearchText(input.userMessage);

  const record: ConversationRecord = {
    userId: input.userId,
    userMessage: input.userMessage,
    assistantMessage: input.assistantMessage,
    riskLevel: input.riskLevel,
    sourcesUsed: input.sourcesUsed,
    messages:
      input.messages && input.messages.length > 0
        ? input.messages
        : [
            { role: "user", content: input.userMessage },
            { role: "assistant", content: input.assistantMessage },
          ],
    embeddingText,
    embeddingPayload: embeddingVector,
    queryText,
    searchText: normalizeSearchText(embeddingText),
    symptoms: input.userMessage,
    risk_level: input.riskLevel,
    summary: input.assistantMessage,
    createdAt,
  };

  const result = await database
    .collection<ConversationRecord>("conversations")
    .insertOne(record);

  return {
    ...record,
    _id: result.insertedId,
  };
}

export async function incrementConversationReuseCount(id: ObjectId): Promise<void> {
  const database = await getDb();
  await database
    .collection<ConversationRecord>("conversations")
    .updateOne({ _id: id }, { $inc: { reuseCount: 1 } });
}
