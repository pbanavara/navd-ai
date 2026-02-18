import { mkdirSync } from 'node:fs';
import { AppendLog } from './log.js';
import { EmbeddingIndex } from './embedding-index.js';
import { topKSimilarity } from './similarity.js';
import type { EmbeddingProvider, MemoryConfig, QueryResult } from './types.js';

const log = (...args: unknown[]) => console.log('[lsm-ei:memory]', ...args);
const logError = (...args: unknown[]) => console.error('[lsm-ei:memory]', ...args);

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
    log(`initializing dir=${config.dir} chunkSize=${config.chunkSize ?? DEFAULT_CHUNK_SIZE} embeddingDim=${config.embedding.dimensions}`);
    try {
      mkdirSync(config.dir, { recursive: true });
    } catch (err) {
      logError(`failed to create directory ${config.dir}`, err);
      throw err;
    }

    this.embedding = config.embedding;
    this.chunkSize = config.chunkSize ?? DEFAULT_CHUNK_SIZE;
    this.log = new AppendLog(config.dir);
    this.index = new EmbeddingIndex(config.dir, config.embedding.dimensions);

    // Detect un-indexed tail: if the log has bytes beyond the last indexed chunk,
    // a previous session crashed after appending to the log but before embedding.
    // Set the chunk start to where the index left off so close() will backfill.
    const lastIndexedEnd = this.index.lastIndexedEnd();
    const logEnd = this.log.position;

    if (lastIndexedEnd < logEnd) {
      this.chunkStartOffset = lastIndexedEnd;
      this.bytesSinceLastEmbed = logEnd - lastIndexedEnd;
      if (lastIndexedEnd > 0) {
        log(`recovery: found ${this.bytesSinceLastEmbed} un-indexed bytes in log (index ends at ${lastIndexedEnd}, log ends at ${logEnd})`);
      }
    } else {
      this.chunkStartOffset = logEnd;
      this.bytesSinceLastEmbed = 0;
    }

    log(`ready chunkStartOffset=${this.chunkStartOffset} bytesSinceLastEmbed=${this.bytesSinceLastEmbed}`);
  }

  /** Append a conversational turn and flush a chunk if the threshold is reached. */
  async append(turn: { role: string; text: string }): Promise<void> {
    log(`append role=${turn.role} textLength=${turn.text.length}`);
    const line = JSON.stringify({ ts: Date.now(), ...turn });
    const { length } = this.log.append(line);
    this.bytesSinceLastEmbed += length;
    log(`bytesSinceLastEmbed=${this.bytesSinceLastEmbed}/${this.chunkSize}`);

    if (this.bytesSinceLastEmbed >= this.chunkSize) {
      log('chunk threshold reached, flushing');
      await this.flushChunk();
    }
  }

  /** Query the memory for the top-k most similar chunks. */
  async query(text: string, opts?: { topK?: number }): Promise<QueryResult[]> {
    const topK = opts?.topK ?? 5;
    log(`query topK=${topK} textLength=${text.length}`);

    let queryVec: Float32Array;
    try {
      queryVec = await this.embedding.embed(text);
    } catch (err) {
      logError('query: embedding failed', err);
      throw err;
    }

    const { vectors, norms, offsets, lengths } = this.index.readAll();

    if (vectors.length === 0) {
      log('query: no indexed vectors, returning empty');
      return [];
    }

    const hits = topKSimilarity(queryVec, vectors, norms, topK);
    log(`query: found ${hits.length} hit(s) topScore=${hits.length > 0 ? hits[0].score.toFixed(4) : 'n/a'}`);

    return hits.map((h) => ({
      text: this.log.read(offsets[h.index], lengths[h.index]),
      score: h.score,
      offset: offsets[h.index],
      length: lengths[h.index],
    }));
  }

  /** Flush any pending chunk and close file handles. */
  async close(): Promise<void> {
    log('closing');
    if (this.bytesSinceLastEmbed > 0) {
      log(`flushing remaining chunk bytes=${this.bytesSinceLastEmbed}`);
      await this.flushChunk();
    }
    this.index.close();
    this.log.close();
    log('closed');
  }

  private async flushChunk(): Promise<void> {
    const offset = this.chunkStartOffset;
    const length = this.bytesSinceLastEmbed;
    log(`flushChunk offset=${offset} length=${length}`);

    const chunkText = this.log.read(offset, length);

    let vector: Float32Array;
    try {
      vector = await this.embedding.embed(chunkText);
    } catch (err) {
      logError(`flushChunk: embedding failed for chunk at offset=${offset} length=${length}`, err);
      throw err;
    }

    this.index.append([{ vector, offset, length }]);

    this.chunkStartOffset = this.log.position;
    this.bytesSinceLastEmbed = 0;
    log(`flushChunk complete, new chunkStartOffset=${this.chunkStartOffset}`);
  }
}
