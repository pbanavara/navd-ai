import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  Schema,
  Field,
  FixedSizeList,
  Float32,
  Float64,
  Int64,
  Struct,
  makeData,
  RecordBatch,
  Table,
  RecordBatchStreamWriter,
  RecordBatchReader,
} from 'apache-arrow';

const log = (...args: unknown[]) => console.log('[lsm-ei:index]', ...args);
const logError = (...args: unknown[]) => console.error('[lsm-ei:index]', ...args);

export interface IndexEntry {
  vector: Float32Array;
  offset: number;
  length: number;
}

export interface IndexData {
  vectors: Float32Array[];
  norms: number[];
  offsets: number[];
  lengths: number[];
}

function buildSchema(dim: number): Schema {
  return new Schema([
    new Field(
      'vector',
      new FixedSizeList(dim, new Field('value', new Float32(), false)),
      false,
    ),
    new Field('norm', new Float64(), false),
    new Field('offset', new Int64(), false),
    new Field('length', new Int64(), false),
  ]);
}

function entriesToBatch(entries: IndexEntry[], schema: Schema): RecordBatch {
  const dim = (schema.fields[0].type as FixedSizeList).listSize;
  const n = entries.length;

  const flatFloats = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    flatFloats.set(entries[i].vector, i * dim);
  }

  const childData = makeData({
    type: new Float32(),
    length: n * dim,
    data: flatFloats,
  });

  const vectorData = makeData({
    type: new FixedSizeList(dim, new Field('value', new Float32(), false)),
    length: n,
    child: childData,
  });

  // Pre-compute L2 norms at write time so queries only need dot product + divide
  const normValues = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const vec = entries[i].vector;
    for (let j = 0; j < vec.length; j++) sum += vec[j] * vec[j];
    normValues[i] = Math.sqrt(sum);
  }

  const offsetValues = new BigInt64Array(n);
  const lengthValues = new BigInt64Array(n);
  for (let i = 0; i < n; i++) {
    offsetValues[i] = BigInt(entries[i].offset);
    lengthValues[i] = BigInt(entries[i].length);
  }

  const normData = makeData({ type: new Float64(), length: n, data: normValues });
  const offsetData = makeData({ type: new Int64(), length: n, data: offsetValues });
  const lengthData = makeData({ type: new Int64(), length: n, data: lengthValues });

  const structType = new Struct(schema.fields);
  const structData = makeData({
    type: structType,
    length: n,
    children: [vectorData, normData, offsetData, lengthData],
  });

  return new RecordBatch(schema, structData);
}

/** Size of the Arrow IPC stream end-of-stream marker (continuation + 0-length metadata). */
const EOS_SIZE = 8;

/**
 * Compute the byte size of the schema header in an Arrow IPC stream.
 * We serialize an empty table with the schema, which produces [schema_msg + EOS],
 * then subtract the EOS marker size.
 */
function schemaHeaderSize(schema: Schema): number {
  const emptyTable = new Table(schema, []);
  const bytes = RecordBatchStreamWriter.writeAll(emptyTable).toUint8Array(true);
  return bytes.byteLength - EOS_SIZE;
}

export class EmbeddingIndex {
  readonly filePath: string;
  private schema: Schema;
  private pendingBatches: RecordBatch[] = [];
  private cachedSchemaSize: number | null = null;

  constructor(dir: string, dim: number) {
    this.filePath = path.join(dir, 'embeddings.arrow');
    this.schema = buildSchema(dim);
    log(`initialized dim=${dim} file=${this.filePath}`);
  }

  private getSchemaSize(): number {
    if (this.cachedSchemaSize === null) {
      this.cachedSchemaSize = schemaHeaderSize(this.schema);
      log(`computed schema header size=${this.cachedSchemaSize} bytes`);
    }
    return this.cachedSchemaSize;
  }

  /** Buffer a new batch in memory. Call flush() or close() to persist. */
  append(entries: IndexEntry[]): void {
    if (entries.length === 0) return;
    log(`buffering batch entries=${entries.length} pending=${this.pendingBatches.length + 1}`);
    this.pendingBatches.push(entriesToBatch(entries, this.schema));
  }

  /**
   * Flush pending batches to the Arrow IPC stream file.
   *
   * - If the file doesn't exist: writes a complete stream (schema + batches + EOS).
   * - If the file exists: truncates the EOS marker, appends only the new batch
   *   messages + a new EOS marker. No read-modify-write.
   */
  flush(): void {
    if (this.pendingBatches.length === 0) {
      log('flush: nothing pending, skipping');
      return;
    }

    log(`flush: writing ${this.pendingBatches.length} pending batch(es)`);

    try {
      // Serialize the pending batches as a full stream (schema + batches + EOS)
      const table = new Table(this.schema, this.pendingBatches);
      const fullStream = RecordBatchStreamWriter.writeAll(table).toUint8Array(true);
      log(`flush: serialized stream size=${fullStream.byteLength} bytes`);

      if (!fs.existsSync(this.filePath)) {
        // First write: write the complete stream
        log(`flush: creating new file ${this.filePath}`);
        fs.writeFileSync(this.filePath, fullStream);
      } else {
        // Append: strip schema header from new bytes, truncate old EOS, append
        const schemaSize = this.getSchemaSize();
        const batchBytesWithEOS = fullStream.slice(schemaSize);

        // Remove the old EOS marker from the existing file
        const stat = fs.statSync(this.filePath);
        const newSize = stat.size - EOS_SIZE;
        log(`flush: appending to existing file size=${stat.size} truncating to=${newSize} appending=${batchBytesWithEOS.byteLength} bytes`);
        fs.truncateSync(this.filePath, newSize);

        // Append the new batch messages + new EOS
        fs.appendFileSync(this.filePath, batchBytesWithEOS);
      }

      this.pendingBatches = [];
      log('flush: complete');
    } catch (err) {
      logError('flush failed', err);
      throw err;
    }
  }

  /** Read all entries from the Arrow IPC stream file plus any unflushed in-memory batches. */
  readAll(): IndexData {
    const vectors: Float32Array[] = [];
    const norms: number[] = [];
    const offsets: number[] = [];
    const lengths: number[] = [];

    // Read persisted batches from disk.
    // v1 simplification: readFileSync copies the file into a Node.js Buffer.
    // The design doc mentions mmap for zero-copy reads — at personal-agent scale
    // (~8MB per user) this doesn't matter, but for larger deployments a native
    // mmap addon (e.g. mmap-io) would avoid the copy.
    if (fs.existsSync(this.filePath)) {
      log(`readAll: reading from ${this.filePath}`);
      try {
        const bytes = fs.readFileSync(this.filePath);
        const reader = RecordBatchReader.from(bytes);
        let diskBatches = 0;
        for (const batch of reader) {
          extractBatch(batch, vectors, norms, offsets, lengths);
          diskBatches++;
        }
        log(`readAll: read ${diskBatches} batch(es) ${vectors.length} vector(s) from disk`);
      } catch (err) {
        logError(`readAll: failed to read ${this.filePath}`, err);
        throw err;
      }
    } else {
      log('readAll: no file on disk');
    }

    // Include unflushed in-memory batches
    if (this.pendingBatches.length > 0) {
      const beforeCount = vectors.length;
      for (const batch of this.pendingBatches) {
        extractBatch(batch, vectors, norms, offsets, lengths);
      }
      log(`readAll: added ${vectors.length - beforeCount} vector(s) from ${this.pendingBatches.length} in-memory batch(es)`);
    }

    log(`readAll: total vectors=${vectors.length}`);
    return { vectors, norms, offsets, lengths };
  }

  /** Flush pending batches to disk. */
  close(): void {
    log('closing embedding index');
    this.flush();
  }
}

function extractBatch(
  batch: RecordBatch,
  vectors: Float32Array[],
  norms: number[],
  offsets: number[],
  lengths: number[],
): void {
  const vectorCol = batch.getChild('vector')!;
  const normCol = batch.getChild('norm')!;
  const offsetCol = batch.getChild('offset')!;
  const lengthCol = batch.getChild('length')!;

  for (let i = 0; i < batch.numRows; i++) {
    const inner = vectorCol.get(i)!;
    vectors.push(new Float32Array(inner.toArray() as Float32Array));
    norms.push(normCol.get(i) as number);
    offsets.push(Number(offsetCol.get(i) as bigint));
    lengths.push(Number(lengthCol.get(i) as bigint));
  }
}
