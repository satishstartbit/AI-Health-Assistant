import { env, pipeline } from "@xenova/transformers";

env.allowLocalModels = false;

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

type FeatureExtractor = Awaited<ReturnType<typeof pipeline>>;

let extractorPromise: Promise<FeatureExtractor> | null = null;

async function getExtractor(): Promise<FeatureExtractor> {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", EMBEDDING_MODEL);
  }

  return extractorPromise;
}

export async function generateEmbedding(text: string): Promise<number[]> {
  const normalizedText = text.trim();

  if (!normalizedText) {
    return [];
  }

  const extractor = await getExtractor();
  const output = await (extractor as (
    input: string,
    options: { pooling: "mean"; normalize: boolean }
  ) => Promise<{ data: Float32Array }>)(normalizedText, {
    pooling: "mean",
    normalize: true,
  });

  return Array.from(output.data as Float32Array);
}

export function getEmbeddingModelName() {
  return EMBEDDING_MODEL;
}
