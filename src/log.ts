import * as fs from 'node:fs';
import * as path from 'node:path';

export class AppendLog {
  private fd: number;
  private currentOffset: number;
  readonly filePath: string;

  constructor(dir: string) {
    this.filePath = path.join(dir, 'conversations.log');
    this.fd = fs.openSync(this.filePath, 'a+');
    const stat = fs.fstatSync(this.fd);
    this.currentOffset = stat.size;
  }

  /** Append a line and return { offset, length } of the written bytes. */
  append(line: string): { offset: number; length: number } {
    const data = line + '\n';
    const buf = Buffer.from(data, 'utf-8');
    const offset = this.currentOffset;
    fs.writeSync(this.fd, buf);
    this.currentOffset += buf.byteLength;
    return { offset, length: buf.byteLength };
  }

  /** Read a chunk of bytes from the log at (offset, length). */
  read(offset: number, length: number): string {
    const buf = Buffer.alloc(length);
    const readFd = fs.openSync(this.filePath, 'r');
    try {
      fs.readSync(readFd, buf, 0, length, offset);
    } finally {
      fs.closeSync(readFd);
    }
    return buf.toString('utf-8');
  }

  /** Current byte position (end of file). */
  get position(): number {
    return this.currentOffset;
  }

  close(): void {
    fs.closeSync(this.fd);
  }
}
