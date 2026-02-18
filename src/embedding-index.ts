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
  tableToIPC,
  tableFromIPC,
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

export class EmbeddingIndex {
  readonly filePath: string;
  private dim: number;

  constructor(dir: string, dim: number) {
    this.filePath = path.join(dir, 'embeddings.arrow');
    this.dim = dim;
  }

  /** Append entries to the Arrow IPC file. */
  append(entries: IndexEntry[]): void {
    if (entries.length === 0) return;

    const schema = buildSchema(this.dim);
    const newBatch = entriesToBatch(entries, schema);
    const newTable = new Table(schema, [newBatch]);

    let merged: Table;
    if (fs.existsSync(this.filePath)) {
      const existing = tableFromIPC(fs.readFileSync(this.filePath));
      merged = existing.concat(newTable);
    } else {
      merged = newTable;
    }

    fs.writeFileSync(this.filePath, tableToIPC(merged, 'file'));
  }

  /** Read all entries from the Arrow IPC file. */
  readAll(): IndexData {
    if (!fs.existsSync(this.filePath)) {
      return { vectors: [], offsets: [], lengths: [] };
    }

    const table = tableFromIPC(fs.readFileSync(this.filePath));
    const numRows = table.numRows;

    const vectorCol = table.getChild('vector')!;
    const offsetCol = table.getChild('offset')!;
    const lengthCol = table.getChild('length')!;

    const vectors: Float32Array[] = [];
    const offsets: number[] = [];
    const lengths: number[] = [];

    for (let i = 0; i < numRows; i++) {
      const inner = vectorCol.get(i)!;
      vectors.push(new Float32Array(inner.toArray() as Float32Array));
      offsets.push(Number(offsetCol.get(i) as bigint));
      lengths.push(Number(lengthCol.get(i) as bigint));
    }

    return { vectors, offsets, lengths };
  }
}
