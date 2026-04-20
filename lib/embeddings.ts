const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

type FeatureExtractor = Awaited<ReturnType<typeof import("@xenova/transformers").pipeline>>;

let extractorPromise: Promise<FeatureExtractor> | null = null;

async function getExtractor(): Promise<FeatureExtractor> {
  if (!extractorPromise) {
    const transformers = await import("@xenova/transformers").catch(() => {
      throw new Error("Embeddings not available");
    });
    transformers.env.allowLocalModels = false;
    extractorPromise = transformers.pipeline("feature-extraction", EMBEDDING_MODEL);
  }

  return extractorPromise;
}

export async function generateEmbedding(text: string): Promise<number[]> {
  const normalizedText = text.trim();

  if (!normalizedText) {
    return [];
  }

  try {
    const extractor = await getExtractor();
    const output = await (extractor as (
      input: string,
      options: { pooling: "mean"; normalize: boolean }
    ) => Promise<{ data: Float32Array }>)(normalizedText, {
      pooling: "mean",
      normalize: true,
    });

    return Array.from(output.data as Float32Array);
  } catch {
    // If embeddings fail (e.g., on Vercel), return empty array
    return [];
  }
}

export function getEmbeddingModelName() {
  return EMBEDDING_MODEL;
}
