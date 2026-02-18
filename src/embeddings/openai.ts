import type { EmbeddingProvider } from '../types.js';

const log = (...args: unknown[]) => console.log('[lsm-ei:openai]', ...args);
const logError = (...args: unknown[]) => console.error('[lsm-ei:openai]', ...args);

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
    log(`initialized model=${this.model} dimensions=${this.dimensions}`);
  }

  async embed(text: string): Promise<Float32Array> {
    const result = await this.embedBatch([text]);
    return result[0];
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    log(`embedBatch count=${texts.length} totalChars=${texts.reduce((s, t) => s + t.length, 0)}`);
    try {
      const { default: OpenAI } = await import('openai');
      const client = new OpenAI({ apiKey: this.apiKey });

      const response = await client.embeddings.create({
        model: this.model,
        input: texts,
        dimensions: this.dimensions,
      });

      log(`embedBatch complete vectors=${response.data.length} usage=${JSON.stringify(response.usage)}`);

      return response.data.map(
        (item) => new Float32Array(item.embedding),
      );
    } catch (err) {
      logError(`embedBatch failed for ${texts.length} text(s)`, err);
      throw err;
    }
  }
}
