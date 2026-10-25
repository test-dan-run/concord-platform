// "Stop and clear", then the same file again: when the upload from before the clear fails, it must not forget
// the new upload of that file, or a third copy of the file starts a second upload of the same bytes beside it.

import { describe, expect, it } from "vitest";
import type { AudioRecord } from "./datasets";
import { Uploader, type Transport } from "./upload";

const SHA = "a".repeat(64);
const rec = (s: string): AudioRecord => ({ sha256: s, size: 1, ext: ".wav", duration: 1, sample_rate: 16000, channels: 1, state: "ready", created_at: "" });
const file = (name: string) => [{ file: new Blob(["same bytes"]), path: `${name}.wav` }];
const ticks = async (n: number) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

describe("Uploader after a clear", () => {
  it("keeps sharing the new upload of a file when the old one fails", async () => {
    const calls: { ok: (shas: string[]) => void; fail: (e: Error) => void }[] = [];
    const t: Transport = {
      missing: () => new Promise((ok, fail) => void calls.push({ ok, fail })), // the store answers when told to
      put: async (_sha, _ext, _offset, body) => body.size,
      commit: async (s) => rec(s),
      unpack: async () => ({ files: [], skipped: [] }),
    };
    const up = new Uploader(t, { concurrency: 3, hash: async () => SHA });
    up.add(file("a"));
    await ticks(5);
    expect(calls.length).toBe(1);
    up.clear();
    up.add(file("a"));
    await ticks(5);
    expect(calls.length).toBe(2); // the new upload of the file asks the store itself
    calls[0].fail(new TypeError("Failed to fetch")); // the old call ends after the clear
    await ticks(5);
    up.add(file("b")); // the same bytes under another name: it shares the new upload
    await ticks(5);
    expect(calls.length).toBe(2);
    calls[1].ok([]);
    await up.idle();
    expect(up.getSnapshot().map((i) => [i.path, i.state])).toEqual([["a.wav", "done"], ["b.wav", "done"]]);
  });
});
