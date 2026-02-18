export interface EmbeddingProvider {
  dimensions: number;
  embed(text: string): Promise<Float32Array>;
  embedBatch?(texts: string[]): Promise<Float32Array[]>;
}

export interface MemoryConfig {
  dir: string;
  embedding: EmbeddingProvider;
  chunkSize?: number;
}

export interface QueryResult {
  text: string;
  score: number;
  offset: number;
  length: number;
}
