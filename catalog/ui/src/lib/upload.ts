// Upload audio to the store (DATASET_CATALOG_PLAN.md §6.1). Each file goes through these steps:
//
//   hashing -> checking -> uploading -> committing -> (unpacking, for a zip) -> done
//
// The browser hashes the file, asks POST /store/missing, and sends only what the store lacks, in 16 MB chunks to
// PUT /store/audio/{sha256}?offset=. The store answers 409 with error.size when the offset isn't what it has, so
// after a dropped connection the next chunk starts where the store's copy ends, and a file whose upload broke off
// earlier (another visit, a restarted server) resumes from the store's size too. Connection errors are retried with
// a growing delay; after RETRIES the file waits for a person to press Retry. Commit checks the sha256 and probes
// the file with ffprobe; a file ffprobe can't read is "unreadable" and left out of the version. A zip is sent whole
// and unpacked by the catalog service (POST /store/unpack), which lists the files it stored and the ones it skipped.

import { api, ApiError, errorOf } from "./api";
import type { AudioRecord } from "./datasets";
import { hashBlob } from "./sha256";

export const CHUNK = 16 * 1024 * 1024;
export const AUDIO_EXTS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".wma", ".webm"];
const RETRIES = 10;
const DELAYS_S = [1, 2, 4, 8, 15, 30];
const CONCURRENCY = 3;

export type ItemState = "waiting" | "hashing" | "checking" | "uploading" | "retrying" | "committing" | "unpacking" | "done" | "unreadable" | "skipped" | "failed";
export const BUSY: ItemState[] = ["hashing", "checking", "uploading", "retrying", "committing", "unpacking"];

export type ZipEntry = { path: string; sha256: string; duration: number | null; sample_rate: number | null };

export type UploadItem = {
  id: number;
  path: string;
  file: Blob;
  size: number;
  zip: boolean;
  state: ItemState;
  hashed: number;
  sent: number;
  sha256?: string;
  /** The store had the whole file already, so nothing was sent. */
  had?: boolean;
  /** Bytes the store already had when this upload started: a resumed upload. */
  resumedAt?: number;
  record?: AudioRecord;
  entries?: ZipEntry[];
  skipped?: { path: string; error: string }[];
  /** Why the file failed, or why it waits (a dropped connection). */
  error?: string;
  attempt?: number;
};

export type PickedFile = { file: Blob; path: string };

export interface Transport {
  missing(shas: string[]): Promise<string[]>;
  /** Append bytes at `offset`; the store's size afterwards. Throws ApiError 409 (body.size) on a wrong offset. */
  put(sha: string, ext: string, offset: number, body: Blob, signal: AbortSignal): Promise<number>;
  commit(sha: string): Promise<AudioRecord>;
  unpack(sha: string): Promise<{ files: (AudioRecord & { path: string })[]; skipped: { path: string; error: string }[] }>;
}

export const http: Transport = {
  missing: async (shas) => (await api.post<{ missing: string[] }>("/store/missing", { sha256: shas })).missing,
  put: async (sha, ext, offset, body, signal) => {
    const r = await fetch(`/api/store/audio/${sha}?ext=${encodeURIComponent(ext)}&offset=${offset}`, { method: "PUT", body, signal });
    if (!r.ok) throw await errorOf(r);
    return ((await r.json()) as { size: number }).size;
  },
  commit: (sha) => api.post<AudioRecord>(`/store/audio/${sha}/commit`),
  unpack: (sha) => api.post(`/store/unpack`, { sha256: sha }),
};

export function extOf(path: string): string {
  const m = /\.[^./]+$/.exec(path);
  return m ? m[0].toLowerCase() : "";
}

export function accepted(path: string): boolean {
  const ext = extOf(path);
  return ext === ".zip" || AUDIO_EXTS.includes(ext);
}

/** A dropped connection or a server that is restarting: worth trying again. */
export function retryable(e: unknown): boolean {
  if (e instanceof ApiError) return e.status >= 500 || e.status === 0 || e.body.type === "upload_busy";
  return e instanceof TypeError; // fetch's network error
}

const readable = (r: { duration: number | null; sample_rate: number | null }) => r.duration !== null && r.sample_rate !== null;

/** Wait `ms`, or less when `signal` aborts first. */
function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((r) => {
    if (signal?.aborted) return r();
    const done = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", done);
      r();
    };
    const timer = setTimeout(done, ms);
    signal?.addEventListener("abort", done);
  });
}

type Options = { chunk?: number; concurrency?: number; retries?: number; sleep?: (ms: number, signal?: AbortSignal) => Promise<void>; hash?: typeof hashBlob };

export class Uploader {
  private items: UploadItem[] = [];
  private listeners = new Set<() => void>();
  private next = 1;
  private running = 0;
  private abort = new AbortController();
  private shas = new Map<string, Promise<AudioRecord>>();
  private readonly chunk: number;
  private readonly concurrency: number;
  private readonly retries: number;
  private readonly sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  private readonly hash: typeof hashBlob;

  constructor(
    private t: Transport = http,
    o: Options = {},
  ) {
    this.chunk = o.chunk ?? CHUNK;
    this.concurrency = o.concurrency ?? CONCURRENCY;
    this.retries = o.retries ?? RETRIES;
    this.sleep = o.sleep ?? delay;
    this.hash = o.hash ?? hashBlob;
  }

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getSnapshot = (): UploadItem[] => this.items;

  busy(): boolean {
    return this.items.some((i) => i.state === "waiting" || BUSY.includes(i.state));
  }

  /** Queue files; anything but audio and zips is refused and returned. */
  add(files: PickedFile[]): PickedFile[] {
    const refused = files.filter((f) => !accepted(f.path));
    const fresh = files
      .filter((f) => accepted(f.path))
      .map((f): UploadItem => ({ id: this.next++, path: f.path, file: f.file, size: f.file.size, zip: extOf(f.path) === ".zip", state: "waiting", hashed: 0, sent: 0 }));
    this.items = [...this.items, ...fresh];
    this.emit();
    this.pump();
    return refused;
  }

  retry(id: number): void {
    this.patch(id, { state: "waiting", error: undefined, attempt: undefined });
    this.pump();
  }

  remove(id: number): void {
    const it = this.items.find((i) => i.id === id);
    if (!it || BUSY.includes(it.state)) return;
    this.items = this.items.filter((i) => i.id !== id);
    this.emit();
  }

  /** Stop everything and forget the list. Bytes already sent stay in the store, so the same files resume later. */
  clear(): void {
    this.abort.abort();
    this.abort = new AbortController();
    this.running = 0; // the stopped runs may still be waiting on the store, but they hold no slot
    this.items = [];
    this.shas.clear();
    this.emit();
  }

  /** Wait until no file is waiting or busy (tests). */
  async idle(): Promise<void> {
    while (this.busy()) await new Promise((r) => setTimeout(r, 0));
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  private patch(id: number, p: Partial<UploadItem>): void {
    this.items = this.items.map((i) => (i.id === id ? { ...i, ...p } : i));
    this.emit();
  }

  private get(id: number): UploadItem | undefined {
    return this.items.find((i) => i.id === id);
  }

  private pump(): void {
    while (this.running < this.concurrency) {
      const it = this.items.find((i) => i.state === "waiting");
      if (!it) return;
      this.running++;
      const abort = this.abort;
      this.patch(it.id, { state: "hashing" });
      void this.run(it.id, abort.signal).finally(() => {
        if (abort !== this.abort) return; // cleared since: clear() started the count again
        this.running--;
        this.pump();
      });
    }
  }

  private async run(id: number, signal: AbortSignal): Promise<void> {
    try {
      const it = this.get(id)!;
      const sha = it.sha256 ?? (await this.hash(it.file, (n) => !signal.aborted && this.patch(id, { hashed: n }), signal));
      if (signal.aborted) return;
      this.patch(id, { sha256: sha, hashed: it.size, state: "checking" });
      const twin = this.items.find((i) => i.id < id && i.zip && it.zip && i.sha256 === sha && i.state !== "skipped");
      if (twin) {
        this.patch(id, { state: "skipped", error: `the same zip as ${twin.path}` });
        return;
      }
      let first = this.shas.get(sha);
      if (!first) {
        first = this.store(id, sha, signal);
        this.shas.set(sha, first);
        first.catch(() => this.shas.delete(sha)); // a later Retry starts again
      }
      const rec = await first;
      if (signal.aborted) return;
      if (it.zip) {
        this.patch(id, { state: "unpacking" });
        const out = await this.retrying(id, signal, () => this.t.unpack(sha));
        if (signal.aborted) return;
        this.patch(id, { state: "done", entries: out.files.map((f) => ({ path: f.path, sha256: f.sha256, duration: f.duration, sample_rate: f.sample_rate })), skipped: out.skipped });
      } else if (!readable(rec)) {
        this.patch(id, { state: "unreadable", record: rec, error: "ffprobe can't read it" });
      } else {
        this.patch(id, { state: "done", record: rec });
      }
    } catch (e) {
      if (signal.aborted) return;
      this.patch(id, { state: "failed", error: e instanceof Error ? e.message : String(e) });
    }
  }

  /** Make sure the store has `sha`, sending what it lacks, and return its record. */
  private async store(id: number, sha: string, signal: AbortSignal): Promise<AudioRecord> {
    const it = this.get(id)!;
    const missing = await this.retrying(id, signal, () => this.t.missing([sha]));
    if (!missing.includes(sha)) {
      this.patch(id, { had: true, sent: it.size, state: "committing" });
      return this.retrying(id, signal, () => this.t.commit(sha)); // the stored record: duration, or unreadable
    }
    for (let again = true; ; again = false) {
      await this.send(id, sha, signal);
      this.patch(id, { state: "committing" });
      try {
        return await this.retrying(id, signal, () => this.t.commit(sha));
      } catch (e) {
        // The bytes didn't hash to sha256 (a partial file left by something else): the store removed them, so
        // start once more from the beginning.
        if (again && e instanceof ApiError && e.status === 422 && e.body.type === "hash_mismatch") {
          this.patch(id, { sent: 0, resumedAt: undefined });
          continue;
        }
        throw e;
      }
    }
  }

  private async send(id: number, sha: string, signal: AbortSignal): Promise<void> {
    const it = this.get(id)!;
    const ext = extOf(it.path);
    let offset = 0;
    let fails = 0;
    this.patch(id, { state: "uploading", sent: 0 });
    do {
      signal.throwIfAborted();
      const end = Math.min(offset + this.chunk, it.size);
      try {
        const size = await this.t.put(sha, ext, offset, it.file.slice(offset, end), signal);
        if (size < end && end > offset) throw new Error(`the store kept ${size} bytes of ${end}`);
        offset = size;
        fails = 0;
        this.patch(id, { state: "uploading", sent: offset, error: undefined, attempt: undefined });
      } catch (e) {
        if (signal.aborted) throw e;
        const size = e instanceof ApiError && e.status === 409 ? e.body.size : undefined;
        if (typeof size === "number" && e instanceof ApiError && e.body.type !== "upload_busy") {
          // The store's copy ends elsewhere: a chunk that broke off part way, or an earlier upload. Go on from there.
          if (offset === 0 && size > 0 && this.get(id)?.resumedAt === undefined) this.patch(id, { resumedAt: size });
          offset = size;
          this.patch(id, { sent: offset });
          continue;
        }
        if (!retryable(e) || ++fails > this.retries) throw e;
        await this.wait(id, fails, e, signal);
      }
    } while (offset < it.size);
  }

  /** Call `fn`, retrying connection errors the way chunks are retried. */
  private async retrying<T>(id: number, signal: AbortSignal, fn: () => Promise<T>): Promise<T> {
    const state = this.get(id)!.state;
    for (let fails = 1; ; fails++) {
      try {
        const out = await fn();
        if (fails > 1) this.patch(id, { state, error: undefined, attempt: undefined });
        return out;
      } catch (e) {
        if (signal.aborted || !retryable(e) || fails > this.retries) throw e;
        await this.wait(id, fails, e, signal);
        this.patch(id, { state });
      }
    }
  }

  private async wait(id: number, fails: number, e: unknown, signal: AbortSignal): Promise<void> {
    const s = DELAYS_S[Math.min(fails, DELAYS_S.length) - 1];
    const why = e instanceof ApiError && e.body.type === "upload_busy" ? "Another upload of the same file is running" : "Connection lost";
    this.patch(id, { state: "retrying", attempt: fails, error: `${why}. Trying again in ${s} s (attempt ${fails} of ${this.retries}).` });
    await this.sleep(s * 1000, signal);
    signal.throwIfAborted();
  }
}

/** The page's uploads. One per tab, so leaving the page and coming back keeps the list and the uploads going. */
export const uploads = new Uploader();

// ----------------------------------------------------------------------------- the version's files

export type Planned = { path: string; sha256: string; duration: number };
export type LeftOut = { path: string; reason: string };

function stemOf(path: string): string {
  const name = path.split("/").pop() ?? path;
  const dot = name.lastIndexOf(".");
  return (dot > 0 ? name.slice(0, dot) : name).toLowerCase();
}

/** The files a version gets from finished uploads, in order, and the ones left out with the reason. The catalog
 *  refuses two files with the same name stem (a.wav and calls/a.flac), because the apps' references name files by
 *  stem, so only the first is kept. `existing` is the latest version's files when adding to a dataset: a file
 *  already there at the same path is kept quietly, and the catalog merges it. */
export function planVersion(items: UploadItem[], existing: { path: string; sha256: string }[] = []): { files: Planned[]; leftOut: LeftOut[]; already: number } {
  const files: Planned[] = [];
  const leftOut: LeftOut[] = [];
  const paths = new Map(existing.map((f) => [f.path, f.sha256]));
  const old = new Set(existing.map((f) => f.path));
  const stems = new Map(existing.map((f) => [stemOf(f.path), f.path]));
  let already = 0;
  const take = (path: string, sha: string, duration: number | null, sampleRate: number | null) => {
    if (duration === null || sampleRate === null) return leftOut.push({ path, reason: "ffprobe can't read it" });
    if (paths.has(path)) {
      if (paths.get(path) !== sha) return leftOut.push({ path, reason: "another file already has this path" });
      if (old.has(path)) already++;
      return;
    }
    const stem = stemOf(path);
    if (stems.has(stem)) return leftOut.push({ path, reason: `same name as ${stems.get(stem)}` });
    paths.set(path, sha);
    stems.set(stem, path);
    files.push({ path, sha256: sha, duration });
  };
  for (const it of items) {
    if (it.state === "unreadable" || it.state === "skipped") leftOut.push({ path: it.path, reason: it.error ?? "ffprobe can't read it" });
    if (it.state !== "done") continue;
    if (it.zip) {
      for (const e of it.entries ?? []) take(e.path, e.sha256, e.duration, e.sample_rate);
      for (const s of it.skipped ?? []) leftOut.push({ path: `${it.path}: ${s.path}`, reason: s.error });
    } else if (it.record && it.sha256) {
      take(it.path, it.sha256, it.record.duration, it.record.sample_rate);
    }
  }
  return { files, leftOut, already };
}

// ----------------------------------------------------------------------------- picking files

export function pickedFromInput(list: FileList | null): PickedFile[] {
  return Array.from(list ?? []).map((f) => ({ file: f, path: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name }));
}

/** Files from a drag and drop, walking dropped folders. */
export async function pickedFromDrop(e: React.DragEvent): Promise<PickedFile[]> {
  const items = Array.from(e.dataTransfer?.items ?? []);
  const entries = items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
  if (!entries.length) return Array.from(e.dataTransfer?.files ?? []).map((f) => ({ file: f, path: f.name }));
  const out: PickedFile[] = [];
  const walk = async (entry: FileSystemEntry, prefix: string): Promise<void> => {
    if (entry.isFile) {
      const f = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
      out.push({ file: f, path: prefix + entry.name });
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
  };
  for (const en of entries) await walk(en, "");
  return out;
}
