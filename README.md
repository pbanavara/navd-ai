# LSM-EI: Log-Structured Memory with Embedding Index

Persistent conversational memory for AI agents. Two files, zero databases, pluggable embeddings.

Raw conversations are the source of truth. Embeddings serve as a spatial index over byte ranges in an append-only log. No summarization, no information loss, no vector database.

## Install

```bash
npm install lsm-ei
```

For the built-in OpenAI adapter, also install the peer dependency:

```bash
npm install openai
```

## Quick Start

```ts
import { Memory, OpenAIEmbedding } from 'lsm-ei';

const mem = new Memory({
  dir: './memory-data',
  embedding: new OpenAIEmbedding({ apiKey: process.env.OPENAI_API_KEY! }),
});

// Write — append conversational turns
await mem.append({ role: 'user', text: 'Buy oat milk on the way home' });
await mem.append({ role: 'assistant', text: "Got it, I'll remind you" });

// Read — semantic search over past conversations
const results = await mem.query('what groceries do I need?', { topK: 5 });
for (const r of results) {
  console.log(`[score=${r.score.toFixed(3)}] ${r.text}`);
}

// Cleanup — flush pending chunks and close file handles
await mem.close();
```

## How It Works

### Storage

LSM-EI uses exactly two files:

```
memory-data/
  conversations.log    ← append-only JSONL, every turn ever recorded
  embeddings.arrow     ← Apache Arrow IPC file: vector → (offset, length)
```

### Write Path

1. `mem.append()` serializes the turn as a JSON line and appends it to `conversations.log`
2. Bytes accumulate in a buffer counter
3. When the buffer exceeds `chunkSize` (default 10 KB), the chunk is embedded and a `(vector, offset, length)` row is appended to `embeddings.arrow`
4. `mem.close()` flushes any remaining partial chunk

### Read Path

1. `mem.query()` embeds the query text using the same provider
2. Loads all vectors from `embeddings.arrow`
3. Brute-force cosine similarity, returns top-k matches
4. For each hit, reads the raw chunk bytes from `conversations.log` at `(offset, length)`

## API

### `new Memory(config)`

```ts
interface MemoryConfig {
  dir: string;                   // directory for the two data files
  embedding: EmbeddingProvider;  // embedding model adapter
  chunkSize?: number;            // bytes per chunk (default 10240)
}
```

### `mem.append(turn)`

```ts
await mem.append({ role: 'user', text: '...' });
```

Appends a turn to the log. Triggers an embed + index write when the chunk size threshold is crossed.

### `mem.query(text, opts?)`

```ts
const results = await mem.query('search text', { topK: 5 });
```

Returns an array of `QueryResult`:

```ts
interface QueryResult {
  text: string;   // raw conversation text from the log
  score: number;  // cosine similarity score
  offset: number; // byte offset in conversations.log
  length: number; // byte length of chunk
}
```

### `mem.close()`

Flushes any pending partial chunk and closes file handles. Always call this when done.

## Custom Embedding Providers

Implement the `EmbeddingProvider` interface to use any embedding model:

```ts
import { Memory, type EmbeddingProvider } from 'lsm-ei';

class MyEmbedding implements EmbeddingProvider {
  dimensions = 384;

  async embed(text: string): Promise<Float32Array> {
    // call your model here
    return new Float32Array(this.dimensions);
  }

  // optional batch method
  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map(t => this.embed(t)));
  }
}

const mem = new Memory({
  dir: './data',
  embedding: new MyEmbedding(),
});
```

### Built-in: `OpenAIEmbedding`

```ts
import { OpenAIEmbedding } from 'lsm-ei';

const embedding = new OpenAIEmbedding({
  apiKey: '...',
  model: 'text-embedding-3-small',  // default
  dimensions: 1536,                  // default
});
```

## Design Principles

1. **Append-only** — never mutate historical data
2. **Embeddings are an index, not storage** — the log is the source of truth
3. **No LLM in the storage path** — only the embedding model touches writes
4. **No vector database** — brute-force cosine similarity is fast enough at personal-agent scale
5. **Lazy retrieval** — only fetch chunks when a query needs them
6. **Rebuildable** — the Arrow index can be regenerated from the log + embedding model

## Scale Characteristics

| Metric | Value |
|--------|-------|
| Vectors for 1 year of heavy use | ~5,000-50,000 |
| Arrow file size at 50k vectors (1536D) | ~300 MB |
| Brute-force search time at 50k vectors | < 10 ms |
| Log file size at 50k chunks of 10 KB | ~500 MB |

## License

MIT
