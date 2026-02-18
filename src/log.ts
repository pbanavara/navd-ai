import * as fs from 'node:fs';
import * as path from 'node:path';

const log = (...args: unknown[]) => console.log('[lsm-ei:log]', ...args);
const logError = (...args: unknown[]) => console.error('[lsm-ei:log]', ...args);

export class AppendLog {
  private fd: number;
  private currentOffset: number;
  readonly filePath: string;

  constructor(dir: string) {
    this.filePath = path.join(dir, 'conversations.log');
    log(`opening ${this.filePath}`);
    try {
      this.fd = fs.openSync(this.filePath, 'a+');
      const stat = fs.fstatSync(this.fd);
      this.currentOffset = stat.size;
      log(`opened at offset=${this.currentOffset}`);
    } catch (err) {
      logError(`failed to open ${this.filePath}`, err);
      throw err;
    }
  }

  /** Append a line and return { offset, length } of the written bytes. */
  append(line: string): { offset: number; length: number } {
    const data = line + '\n';
    const buf = Buffer.from(data, 'utf-8');
    const offset = this.currentOffset;
    try {
      fs.writeSync(this.fd, buf);
    } catch (err) {
      logError(`write failed at offset=${offset} length=${buf.byteLength}`, err);
      throw err;
    }
    this.currentOffset += buf.byteLength;
    log(`append offset=${offset} length=${buf.byteLength} position=${this.currentOffset}`);
    return { offset, length: buf.byteLength };
  }

  /** Read a chunk of bytes from the log at (offset, length). */
  read(offset: number, length: number): string {
    log(`read offset=${offset} length=${length}`);
    const buf = Buffer.alloc(length);
    let readFd: number;
    try {
      readFd = fs.openSync(this.filePath, 'r');
    } catch (err) {
      logError(`failed to open for read ${this.filePath}`, err);
      throw err;
    }
    try {
      fs.readSync(readFd, buf, 0, length, offset);
    } catch (err) {
      logError(`read failed at offset=${offset} length=${length}`, err);
      throw err;
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
    log(`closing ${this.filePath} at position=${this.currentOffset}`);
    try {
      fs.closeSync(this.fd);
    } catch (err) {
      logError(`close failed for ${this.filePath}`, err);
      throw err;
    }
  }
}
