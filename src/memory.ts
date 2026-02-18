import { mkdirSync } from 'node:fs';
import { AppendLog } from './log.js';
import { EmbeddingIndex } from './embedding-index.js';
import { topKSimilarity } from './similarity.js';
import type { EmbeddingProvider, MemoryConfig, QueryResult } from './types.js';

const DEFAULT_CHUNK_SIZE = 10_240; // 10 KB

export class Memory {
  private log: AppendLog;
  private index: EmbeddingIndex;
  private embedding: EmbeddingProvider;
  private chunkSize: number;

  /** Byte offset where the current (un-embedded) chunk starts. */
  private chunkStartOffset: number;
  /** Cumulative bytes written since the last embed. */
  private bytesSinceLastEmbed: number;

  constructor(config: MemoryConfig) {
    mkdirSync(config.dir, { recursive: true });

    this.embedding = config.embedding;
    this.chunkSize = config.chunkSize ?? DEFAULT_CHUNK_SIZE;
    this.log = new AppendLog(config.dir);
    this.index = new EmbeddingIndex(config.dir, config.embedding.dimensions);

    // The chunk starts at the current end of the log file.
    // On a fresh start this is 0; on resume it's wherever we left off.
    this.chunkStartOffset = this.log.position;
    this.bytesSinceLastEmbed = 0;
  }

  /** Append a conversational turn and flush a chunk if the threshold is reached. */
  async append(turn: { role: string; text: string }): Promise<void> {
    const line = JSON.stringify(turn);
    const { length } = this.log.append(line);
    this.bytesSinceLastEmbed += length;

    if (this.bytesSinceLastEmbed >= this.chunkSize) {
      await this.flushChunk();
    }
  }

  /** Query the memory for the top-k most similar chunks. */
  async query(text: string, opts?: { topK?: number }): Promise<QueryResult[]> {
    const topK = opts?.topK ?? 5;
    const queryVec = await this.embedding.embed(text);
    const { vectors, offsets, lengths } = this.index.readAll();

    if (vectors.length === 0) return [];

    const hits = topKSimilarity(queryVec, vectors, topK);

    return hits.map((h) => ({
      text: this.log.read(offsets[h.index], lengths[h.index]),
      score: h.score,
      offset: offsets[h.index],
      length: lengths[h.index],
    }));
  }

  /** Flush any pending chunk and close file handles. */
  async close(): Promise<void> {
    if (this.bytesSinceLastEmbed > 0) {
      await this.flushChunk();
    }
    this.index.close();
    this.log.close();
  }

  private async flushChunk(): Promise<void> {
    const offset = this.chunkStartOffset;
    const length = this.bytesSinceLastEmbed;
    const chunkText = this.log.read(offset, length);

    const vector = await this.embedding.embed(chunkText);
    this.index.append([{ vector, offset, length }]);

    this.chunkStartOffset = this.log.position;
    this.bytesSinceLastEmbed = 0;
  }
}
