# LSM-EI: Log-Structured Memory with Embedding Index

## Overview

A minimal, zero-compaction memory system for AI agents. Raw conversations are the source of truth. Embeddings serve as a spatial index over byte ranges in an append-only log. No summarization, no information loss, no vector database.

## Architecture

```
conversations.log    (append-only, every conversation turn ever)
embeddings.arrow     (Apache Arrow file: embedding vector → file offset + length)
```

## Write Path

1. Message arrives (user or assistant turn)
2. Append raw turn to `conversations.log`, record byte offset
3. Every N bytes (e.g. 10KB) of accumulated content, embed that chunk
4. Append `(vector, offset, length)` to `embeddings.arrow`

No LLM summarization. No compaction. No information loss.

## Read Path (Query Time)

1. Incoming query → embed the query text
2. Cosine similarity over vectors in `embeddings.arrow` (brute-force dot product)
3. Top-k results → each gives a `(file_offset, length)` pointer
4. Seek to offset in `conversations.log` → read `length` bytes
5. Inject retrieved chunks into the LLM context as memory

## Storage Format

### conversations.log

Append-only flat file. Each entry is a raw conversation turn with minimal framing:

```
[offset 0]      {"ts": 1708000000, "role": "user", "text": "Buy oat milk on the way home"}
[offset 128]    {"ts": 1708000001, "role": "assistant", "text": "Got it, I'll remind you"}
...
```

JSONL or length-prefixed binary. The offset is implicit (byte position in the file).

### embeddings.arrow

Apache Arrow file with three columns:

| Column     | Type          | Description                              |
|------------|---------------|------------------------------------------|
| vector     | float32[D]    | Embedding vector (D = model dimension)   |
| offset     | uint64        | Byte offset into conversations.log       |
| length     | uint32        | Byte length of the chunk                 |

v1 uses `readFileSync` (full copy into Node.js heap). At personal-agent scale (~8MB per user) this is fine. A future version could use mmap via a native addon (e.g. `mmap-io`) for true zero-copy reads and OS page cache benefits.

## Why No Compaction

Traditional agent memory systems use an LLM to "compact" conversations into summaries before storing them. This:

- **Loses information** — the LLM decides what's important, discards the rest
- **Adds latency** — an extra LLM call on every session end or context overflow
- **Introduces error** — summarization is a judgment call that can lose nuance
- **Costs money** — every compaction is a billable LLM invocation

With context windows at 200k+ tokens, the justification for aggressive compression is weak. Retrieve raw chunks, let the LLM read the original conversation. The model is already good at extracting what's relevant from verbose logs.

## Why No Vector Database

At personal assistant scale (thousands to low tens of thousands of memory chunks):

- Brute-force cosine similarity over mmap'd floats completes in microseconds
- HNSW/IVF indexing adds complexity for zero practical benefit at this scale
- No server process, no network, no serialization overhead
- The embedding model does the hard work; the "database" is just a dot product and a sort

A standalone vector DB becomes relevant at millions of vectors. For single-user agents, it's unnecessary abstraction.

## Design Principles

1. **Append-only** — never mutate historical data
2. **Embeddings are an index, not storage** — the log is the source of truth
3. **No LLM in the storage path** — only the embedding model touches writes
4. **Minimal I/O** — v1 uses positioned reads; mmap is a future optimization
5. **Lazy retrieval** — only fetch chunks when a query needs them
6. **File pointers as values** — embedding maps to (offset, length), not to copied content

## Implementation Notes

- Embedding model: any model that produces fixed-dimension vectors (e.g. OpenAI `text-embedding-3-small` at 1536D, or a local model)
- Chunk boundary: fixed byte size (e.g. 10KB) or semantic boundary (e.g. per-session or per N turns)
- Arrow file can be rebuilt from the log + embedding model if corrupted (the log is the only source of truth)
- For multi-session use: include a session_id column in the Arrow file for optional filtering before similarity search

## Scale Characteristics

| Metric | Value |
|--------|-------|
| Vectors for 1 year of heavy use | ~5,000-50,000 |
| Arrow file size at 50k vectors (1536D) | ~300MB |
| Brute-force search time at 50k vectors | < 10ms |
| Log file size at 50k chunks of 10KB | ~500MB |
| Log read latency (positioned read) | microseconds |
