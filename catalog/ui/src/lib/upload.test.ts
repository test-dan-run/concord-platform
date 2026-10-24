// The upload state machine against a fake store with the catalog's rules (catalog/datasets/store.py): a PUT at the
// wrong offset is a 409 with the store's size, commit checks the sha256, and a connection can drop part way
// through a chunk, after the store kept some of its bytes.

import { createHash, randomBytes } from "node:crypto";
import { describe, expect, it } from "vitest";
import { ApiError } from "./api";
import type { AudioRecord } from "./datasets";
import { planVersion, Uploader, type Transport, type UploadItem } from "./upload";

const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

class FakeStore implements Transport {
  files = new Map<string, { bytes: Uint8Array; ready: boolean; ext: string }>();
  puts: { sha: string; offset: number; size: number }[] = [];
  /** Drop the connection on these PUTs (1-based), after the store kept half of the chunk. */
  drops = new Set<number>();
  /** Every PUT fails before reaching the store while this is set. */
  down = false;
  unreadable = new Set<string>();
  zips = new Map<string, { files: { path: string; bytes: Uint8Array }[]; skipped: { path: string; error: string }[] }>();

  async missing(shas: string[]) {
    return shas.filter((s) => !this.files.get(s)?.ready);
  }

  async put(s: string, ext: string, offset: number, body: Blob) {
    if (this.down) throw new TypeError("Failed to fetch");
    this.puts.push({ sha: s, offset, size: body.size });
    const f = this.files.get(s) ?? { bytes: new Uint8Array(), ready: false, ext };
    this.files.set(s, f);
    if (f.ready) return f.bytes.length;
    if (offset !== f.bytes.length) throw new ApiError(409, "offset", { type: "offset_mismatch", size: f.bytes.length });
    let data = new Uint8Array(await body.arrayBuffer());
    const drop = this.drops.has(this.puts.length);
    if (drop) data = data.subarray(0, Math.floor(data.length / 2));
    const next = new Uint8Array(f.bytes.length + data.length);
    next.set(f.bytes);
    next.set(data, f.bytes.length);
    f.bytes = next;
    if (drop) throw new TypeError("Failed to fetch");
    return f.bytes.length;
  }

  record(s: string): AudioRecord {
    const f = this.files.get(s)!;
    const ok = !this.unreadable.has(s);
    return { sha256: s, size: f.bytes.length, ext: f.ext, duration: ok ? 1.5 : null, sample_rate: ok ? 16000 : null, channels: ok ? 1 : null, state: "ready", created_at: "" };
  }

  async commit(s: string) {
    const f = this.files.get(s);
    if (!f) throw new ApiError(404, "no upload");
    if (!f.ready && sha(f.bytes) !== s) {
      this.files.delete(s);
      throw new ApiError(422, "mismatch", { type: "hash_mismatch" });
    }
    f.ready = true;
    return this.record(s);
  }

  async unpack(s: string) {
    const z = this.zips.get(s)!;
    const files = z.files.map((f) => {
      const h = sha(f.bytes);
      this.files.set(h, { bytes: f.bytes, ready: true, ext: ".wav" });
      return { path: f.path, ...this.record(h) };
    });
    return { files, skipped: z.skipped };
  }
}

function setup(o: { chunk?: number; retries?: number } = {}) {
  const store = new FakeStore();
  const sleeps: number[] = [];
  const states: string[] = [];
  const up = new Uploader(store, { chunk: o.chunk ?? 1000, retries: o.retries ?? 3, sleep: async (ms) => void sleeps.push(ms) });
  up.subscribe(() => states.push(up.getSnapshot().map((i) => i.state).join(",")));
  return { store, up, sleeps, states };
}

const blob = (b: Uint8Array) => new Blob([new Uint8Array(b)]);
const only = (up: Uploader): UploadItem => up.getSnapshot()[0];

describe("Uploader", () => {
  it("hashes a file, sends it in chunks and commits it", async () => {
    const { store, up, states } = setup();
    const data = randomBytes(3500);
    up.add([{ file: blob(data), path: "calls/a.wav" }]);
    await up.idle();
    const it0 = only(up);
    expect(it0.state).toBe("done");
    expect(it0.sha256).toBe(sha(data));
    expect(it0.sent).toBe(3500);
    expect(it0.record?.duration).toBe(1.5);
    expect(store.puts.map((p) => [p.offset, p.size])).toEqual([[0, 1000], [1000, 1000], [2000, 1000], [3000, 500]]);
    expect(store.files.get(sha(data))!.ext).toBe(".wav");
    expect(states.join(";")).toMatch(/hashing.*checking.*uploading.*committing.*done/);
  });

  it("sends nothing for a file the store already has", async () => {
    const { store, up } = setup();
    const data = randomBytes(2000);
    store.files.set(sha(data), { bytes: data, ready: true, ext: ".wav" });
    up.add([{ file: blob(data), path: "a.wav" }]);
    await up.idle();
    expect(only(up)).toMatchObject({ state: "done", had: true, sent: 2000 });
    expect(store.puts).toEqual([]);
  });

  it("resumes after the connection drops part way through a chunk", async () => {
    const { store, up, sleeps, states } = setup();
    const data = randomBytes(3500);
    store.drops.add(2); // the second chunk: the store keeps 500 of its 1000 bytes, then the connection drops
    up.add([{ file: blob(data), path: "a.wav" }]);
    await up.idle();
    expect(only(up).state).toBe("done");
    expect(sleeps).toEqual([1000]);
    expect(states.some((s) => s === "retrying")).toBe(true);
    // Retried at 1000, told the store has 1500, then went on from 1500.
    expect(store.puts.map((p) => p.offset)).toEqual([0, 1000, 1000, 1500, 2500]);
    expect(sha(store.files.get(sha(data))!.bytes)).toBe(sha(data));
  });

  it("resumes an upload that broke off earlier, from the store's size", async () => {
    const { store, up } = setup();
    const data = randomBytes(3500);
    store.files.set(sha(data), { bytes: data.subarray(0, 1700), ready: false, ext: ".wav" });
    up.add([{ file: blob(data), path: "a.wav" }]);
    await up.idle();
    expect(only(up)).toMatchObject({ state: "done", resumedAt: 1700, sent: 3500 });
    expect(store.puts.map((p) => p.offset)).toEqual([0, 1700, 2700]);
  });

  it("waits for Retry after too many dropped connections, then goes on where the store's copy ends", async () => {
    const store = new FakeStore();
    const sleeps: number[] = [];
    // The first chunk drops part way, and the server stays down through both retries.
    const up = new Uploader(store, { chunk: 1000, retries: 2, sleep: async (ms) => void (sleeps.push(ms), (store.down = true)) });
    const data = randomBytes(2500);
    store.drops.add(1);
    up.add([{ file: blob(data), path: "a.wav" }]);
    await up.idle();
    expect(only(up)).toMatchObject({ state: "failed", sent: 0 });
    expect(only(up).error).toMatch(/Failed to fetch/);
    expect(sleeps).toEqual([1000, 2000]);
    store.down = false;
    up.retry(only(up).id);
    await up.idle();
    expect(only(up)).toMatchObject({ state: "done", sent: 2500 });
    expect(store.puts.map((p) => p.offset)).toEqual([0, 0, 500, 1500]);
  });

  it("starts again from the beginning once when the stored bytes don't match", async () => {
    const { store, up } = setup();
    const data = randomBytes(1500);
    store.files.set(sha(data), { bytes: randomBytes(600), ready: false, ext: ".wav" });
    up.add([{ file: blob(data), path: "a.wav" }]);
    await up.idle();
    expect(only(up).state).toBe("done");
    expect(store.puts.map((p) => p.offset)).toEqual([0, 600, 0, 1000]);
  });

  it("marks a file ffprobe can't read, and refuses files that aren't audio", async () => {
    const { store, up } = setup();
    const data = randomBytes(100);
    store.unreadable.add(sha(data));
    const refused = up.add([
      { file: blob(data), path: "broken.wav" },
      { file: blob(randomBytes(10)), path: "notes.txt" },
    ]);
    expect(refused.map((r) => r.path)).toEqual(["notes.txt"]);
    await up.idle();
    expect(only(up)).toMatchObject({ state: "unreadable", error: "ffprobe can't read it" });
  });

  it("sends a zip whole and lists what the store unpacked and skipped", async () => {
    const { store, up } = setup();
    const zip = randomBytes(1200);
    const inner = randomBytes(300);
    store.zips.set(sha(zip), { files: [{ path: "calls/b.wav", bytes: inner }], skipped: [{ path: "calls/readme.txt", error: "not an audio file" }] });
    up.add([
      { file: blob(zip), path: "batch.zip" },
      { file: blob(zip), path: "copy/batch.zip" },
    ]);
    await up.idle();
    const [a, b] = up.getSnapshot();
    expect(a).toMatchObject({ state: "done", entries: [{ path: "calls/b.wav", sha256: sha(inner), duration: 1.5 }] });
    expect(b).toMatchObject({ state: "skipped", error: "the same zip as batch.zip" });
    expect(store.puts.filter((p) => p.sha === sha(zip)).length).toBe(2);
  });

  it("sends the same bytes once when two files share them", async () => {
    const { store, up } = setup({ chunk: 10_000 });
    const data = randomBytes(800);
    up.add([
      { file: blob(data), path: "a/x.wav" },
      { file: blob(data), path: "b/y.wav" },
    ]);
    await up.idle();
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["done", "done"]);
    expect(store.puts.length).toBe(1);
  });

  it("forgets the list on clear", async () => {
    const { up } = setup();
    up.add([{ file: blob(randomBytes(10)), path: "a.wav" }]);
    up.clear();
    expect(up.getSnapshot()).toEqual([]);
    expect(up.busy()).toBe(false);
  });
});

describe("planVersion", () => {
  const rec = (duration: number | null) => ({ sha256: "", size: 1, ext: ".wav", duration, sample_rate: duration === null ? null : 16000, channels: 1, state: "ready" as const, created_at: "" });
  const item = (p: Partial<UploadItem>): UploadItem => ({ id: 1, path: "", file: new Blob(), size: 1, zip: false, state: "done", hashed: 1, sent: 1, ...p });

  it("keeps the first of two files with the same name stem and lists what it leaves out", () => {
    const out = planVersion([
      item({ path: "a.wav", sha256: "1", record: rec(2) }),
      item({ path: "calls/A.flac", sha256: "2", record: rec(3) }),
      item({ path: "bad.wav", state: "unreadable", error: "ffprobe can't read it" }),
      item({ path: "x.zip", zip: true, entries: [{ path: "z/c.wav", sha256: "3", duration: 1, sample_rate: 8000 }], skipped: [{ path: "z/n.txt", error: "not an audio file" }] }),
      item({ path: "busy.wav", state: "uploading" }),
    ]);
    expect(out.files).toEqual([
      { path: "a.wav", sha256: "1", duration: 2 },
      { path: "z/c.wav", sha256: "3", duration: 1 },
    ]);
    expect(out.leftOut).toEqual([
      { path: "calls/A.flac", reason: "same name as a.wav" },
      { path: "bad.wav", reason: "ffprobe can't read it" },
      { path: "x.zip: z/n.txt", reason: "not an audio file" },
    ]);
  });

  it("checks the files against the dataset's latest version", () => {
    const out = planVersion(
      [
        item({ path: "a.wav", sha256: "1", record: rec(2) }),
        item({ path: "b.wav", sha256: "9", record: rec(2) }),
        item({ path: "new/c.wav", sha256: "3", record: rec(2) }),
        item({ path: "d.wav", sha256: "4", record: rec(2) }),
      ],
      [
        { path: "a.wav", sha256: "1" },
        { path: "b.wav", sha256: "2" },
        { path: "old/c.wav", sha256: "5" },
      ],
    );
    expect(out.files.map((f) => f.path)).toEqual(["d.wav"]);
    expect(out.already).toBe(1);
    expect(out.leftOut).toEqual([
      { path: "b.wav", reason: "another file already has this path" },
      { path: "new/c.wav", reason: "same name as old/c.wav" },
    ]);
  });
});
