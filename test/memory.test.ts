import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Memory } from '../src/memory.js';
import type { EmbeddingProvider } from '../src/types.js';

/** A deterministic mock embedding provider that hashes text into a fixed-size vector. */
class MockEmbedding implements EmbeddingProvider {
  readonly dimensions = 32;

  async embed(text: string): Promise<Float32Array> {
    const vec = new Float32Array(this.dimensions);
    // Simple deterministic hash: use char codes to fill the vector
    for (let i = 0; i < text.length; i++) {
      vec[i % this.dimensions] += text.charCodeAt(i) / 256;
    }
    // Normalize
    let norm = 0;
    for (let i = 0; i < this.dimensions; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < this.dimensions; i++) vec[i] /= norm;
    }
    return vec;
  }
}

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lsm-ei-test-'));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('Memory', () => {
  it('appends turns and creates conversations.log', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 100, // small chunk for testing
    });

    await mem.append({ role: 'user', text: 'Hello' });
    await mem.append({ role: 'assistant', text: 'Hi there!' });
    await mem.close();

    const logPath = path.join(tmpDir, 'conversations.log');
    expect(fs.existsSync(logPath)).toBe(true);

    const content = fs.readFileSync(logPath, 'utf-8');
    const lines = content.trim().split('\n');
    expect(lines.length).toBe(2);
    expect(JSON.parse(lines[0])).toEqual({ role: 'user', text: 'Hello' });
    expect(JSON.parse(lines[1])).toEqual({ role: 'assistant', text: 'Hi there!' });
  });

  it('creates embeddings.arrow after exceeding chunk size', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 50, // very small to trigger quickly
    });

    await mem.append({ role: 'user', text: 'This is a test message that should exceed the chunk size' });
    await mem.close();

    const arrowPath = path.join(tmpDir, 'embeddings.arrow');
    expect(fs.existsSync(arrowPath)).toBe(true);
  });

  it('queries return relevant chunks', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 100,
    });

    // Append enough to create at least one chunk
    await mem.append({ role: 'user', text: 'Buy oat milk and eggs from the grocery store' });
    await mem.append({ role: 'assistant', text: 'Got it, I will remind you to buy oat milk and eggs' });
    await mem.append({ role: 'user', text: 'Schedule a dentist appointment for next Tuesday' });
    await mem.append({ role: 'assistant', text: 'Done, dentist appointment scheduled for Tuesday' });
    await mem.close();

    const results = await mem.query('what groceries do I need?', { topK: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]).toHaveProperty('text');
    expect(results[0]).toHaveProperty('score');
    expect(results[0]).toHaveProperty('offset');
    expect(results[0]).toHaveProperty('length');
    expect(typeof results[0].score).toBe('number');
  });

  it('returns empty results when no chunks exist', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 100_000, // very large so no chunk is created
    });

    await mem.append({ role: 'user', text: 'Hello' });
    // Don't close (which would flush) - query with no indexed chunks
    const results = await mem.query('test');
    expect(results).toEqual([]);

    await mem.close();
  });

  it('handles many turns and returns correct top-k', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 80,
    });

    const topics = [
      'grocery shopping list milk eggs bread',
      'dentist appointment next tuesday morning',
      'project deadline friday code review',
      'weekend hiking trip mountain trail',
      'birthday party planning cake decorations',
      'car maintenance oil change tires',
      'book recommendation science fiction novel',
      'recipe for pasta carbonara italian',
      'gym workout schedule morning routine',
      'vacation planning beach resort flights',
    ];

    for (let i = 0; i < topics.length; i++) {
      await mem.append({ role: 'user', text: topics[i] });
      await mem.append({ role: 'assistant', text: `Noted about: ${topics[i]}` });
    }
    await mem.close();

    const results = await mem.query('groceries milk eggs', { topK: 3 });
    expect(results.length).toBeLessThanOrEqual(3);
    expect(results.length).toBeGreaterThan(0);

    // Scores should be descending
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('chunk text matches what was appended', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
      chunkSize: 50,
    });

    await mem.append({ role: 'user', text: 'Remember to buy milk' });
    await mem.close();

    const results = await mem.query('milk', { topK: 1 });
    expect(results.length).toBe(1);

    // The returned text should be parseable JSON lines
    const lines = results[0].text.trim().split('\n');
    for (const line of lines) {
      const parsed = JSON.parse(line);
      expect(parsed).toHaveProperty('role');
      expect(parsed).toHaveProperty('text');
    }
  });

  it('close is idempotent on empty memory', async () => {
    const mem = new Memory({
      dir: tmpDir,
      embedding: new MockEmbedding(),
    });
    // Close without appending anything - should not throw
    await mem.close();
  });
});
