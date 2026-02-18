import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  Schema,
  Field,
  FixedSizeList,
  Float32,
  Int64,
  Struct,
  makeData,
  RecordBatch,
  Table,
  RecordBatchStreamWriter,
  RecordBatchReader,
} from 'apache-arrow';

export interface IndexEntry {
  vector: Float32Array;
  offset: number;
  length: number;
}

export interface IndexData {
  vectors: Float32Array[];
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

  const offsetValues = new BigInt64Array(n);
  const lengthValues = new BigInt64Array(n);
  for (let i = 0; i < n; i++) {
    offsetValues[i] = BigInt(entries[i].offset);
    lengthValues[i] = BigInt(entries[i].length);
  }

  const offsetData = makeData({ type: new Int64(), length: n, data: offsetValues });
  const lengthData = makeData({ type: new Int64(), length: n, data: lengthValues });

  const structType = new Struct(schema.fields);
  const structData = makeData({
    type: structType,
    length: n,
    children: [vectorData, offsetData, lengthData],
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
  }

  private getSchemaSize(): number {
    if (this.cachedSchemaSize === null) {
      this.cachedSchemaSize = schemaHeaderSize(this.schema);
    }
    return this.cachedSchemaSize;
  }

  /** Buffer a new batch in memory. Call flush() or close() to persist. */
  append(entries: IndexEntry[]): void {
    if (entries.length === 0) return;
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
    if (this.pendingBatches.length === 0) return;

    // Serialize the pending batches as a full stream (schema + batches + EOS)
    const table = new Table(this.schema, this.pendingBatches);
    const fullStream = RecordBatchStreamWriter.writeAll(table).toUint8Array(true);

    if (!fs.existsSync(this.filePath)) {
      // First write: write the complete stream
      fs.writeFileSync(this.filePath, fullStream);
    } else {
      // Append: strip schema header from new bytes, truncate old EOS, append
      const schemaSize = this.getSchemaSize();
      const batchBytesWithEOS = fullStream.slice(schemaSize);

      // Remove the old EOS marker from the existing file
      const stat = fs.statSync(this.filePath);
      fs.truncateSync(this.filePath, stat.size - EOS_SIZE);

      // Append the new batch messages + new EOS
      fs.appendFileSync(this.filePath, batchBytesWithEOS);
    }

    this.pendingBatches = [];
  }

  /** Read all entries from the Arrow IPC stream file plus any unflushed in-memory batches. */
  readAll(): IndexData {
    const vectors: Float32Array[] = [];
    const offsets: number[] = [];
    const lengths: number[] = [];

    // Read persisted batches from disk
    if (fs.existsSync(this.filePath)) {
      const bytes = fs.readFileSync(this.filePath);
      const reader = RecordBatchReader.from(bytes);
      for (const batch of reader) {
        extractBatch(batch, vectors, offsets, lengths);
      }
    }

    // Include unflushed in-memory batches
    for (const batch of this.pendingBatches) {
      extractBatch(batch, vectors, offsets, lengths);
    }

    return { vectors, offsets, lengths };
  }

  /** Flush pending batches to disk. */
  close(): void {
    this.flush();
  }
}

function extractBatch(
  batch: RecordBatch,
  vectors: Float32Array[],
  offsets: number[],
  lengths: number[],
): void {
  const vectorCol = batch.getChild('vector')!;
  const offsetCol = batch.getChild('offset')!;
  const lengthCol = batch.getChild('length')!;

  for (let i = 0; i < batch.numRows; i++) {
    const inner = vectorCol.get(i)!;
    vectors.push(new Float32Array(inner.toArray() as Float32Array));
    offsets.push(Number(offsetCol.get(i) as bigint));
    lengths.push(Number(lengthCol.get(i) as bigint));
  }
}
