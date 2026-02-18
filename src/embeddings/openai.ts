import type { EmbeddingProvider } from '../types.js';

export interface OpenAIEmbeddingConfig {
  apiKey: string;
  model?: string;
  dimensions?: number;
}

export class OpenAIEmbedding implements EmbeddingProvider {
  readonly dimensions: number;
  private apiKey: string;
  private model: string;

  constructor(config: OpenAIEmbeddingConfig) {
    this.apiKey = config.apiKey;
    this.model = config.model ?? 'text-embedding-3-small';
    this.dimensions = config.dimensions ?? 1536;
  }

  async embed(text: string): Promise<Float32Array> {
    const result = await this.embedBatch([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const { default: OpenAI } = await import('openai');
    const client = new OpenAI({ apiKey: this.apiKey });

    const response = await client.embeddings.create({
      model: this.model,
      input: texts,
      dimensions: this.dimensions,
    });

    return response.data.map(
      (item) => new Float32Array(item.embedding),
    );
  }
}
