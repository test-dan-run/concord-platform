// "Stop and clear" while every upload slot waits on something that ignores the abort: a backoff that has not
// ended, or a call to the store that is still out. Files added after the clear must still start.

import { describe, expect, it, vi } from "vitest";
import type { AudioRecord } from "./datasets";
import { Uploader, type Transport } from "./upload";

const rec = (s: string): AudioRecord => ({ sha256: s, size: 1, ext: ".wav", duration: 1, sample_rate: 16000, channels: 1, state: "ready", created_at: "" });
const files = (names: string[]) => names.map((n) => ({ file: new Blob([n]), path: `${n}.wav` }));
const ticks = async (n: number) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setTimeout(r, 0));
};

describe("Uploader.clear", () => {
  it("starts files added after a clear while the old uploads still wait", async () => {
    let down = true;
    const signals: (AbortSignal | undefined)[] = [];
    const t: Transport = {
      missing: async (shas) => {
        if (down) throw new TypeError("Failed to fetch");
        return shas;
      },
      put: async (_sha, _ext, _offset, body) => body.size,
      commit: async (s) => rec(s),
      unpack: async () => ({ files: [], skipped: [] }),
    };
    // A backoff that never ends on its own.
    const sleep = (_ms: number, signal?: AbortSignal) => new Promise<void>(() => void signals.push(signal));
    let n = 0;
    const up = new Uploader(t, { concurrency: 3, retries: 5, sleep, hash: async () => String(++n).padStart(64, "0") });
    up.add(files(["a", "b", "c"]));
    await ticks(20);
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["retrying", "retrying", "retrying"]);
    up.clear();
    down = false;
    up.add(files(["d", "e", "f", "g"]));
    await ticks(50);
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["done", "done", "done", "done"]);
    expect(up.busy()).toBe(false);
    expect(signals.slice(0, 3).every((s) => s?.aborted)).toBe(true); // the old backoffs were told to stop
  });

  it("keeps to the concurrency when an upload from before the clear ends", async () => {
    const pending: (() => void)[] = [];
    const t: Transport = {
      missing: (shas) => new Promise((r) => pending.push(() => r(shas))), // the store answers when told to
      put: async (_sha, _ext, _offset, body) => body.size,
      commit: async (s) => rec(s),
      unpack: async () => ({ files: [], skipped: [] }),
    };
    let n = 0;
    const up = new Uploader(t, { concurrency: 1, hash: async () => String(++n).padStart(64, "0") });
    up.add(files(["a"]));
    await ticks(5);
    expect(pending.length).toBe(1);
    up.clear();
    up.add(files(["b", "c"]));
    await ticks(5);
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["checking", "waiting"]);
    pending[0](); // the old call returns
    await ticks(5);
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["checking", "waiting"]);
    pending[1]();
    await ticks(10);
    pending[2]();
    await up.idle();
    expect(up.getSnapshot().map((i) => i.state)).toEqual(["done", "done"]);
  });

  it("ends a backoff at once when the uploads are cleared", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const t: Transport = {
        missing: async () => {
          throw new TypeError("Failed to fetch");
        },
        put: async () => 0,
        commit: async (s) => rec(s),
        unpack: async () => ({ files: [], skipped: [] }),
      };
      const up = new Uploader(t, { retries: 5, hash: async () => "a".repeat(64) }); // the real backoff, 1 s first
      up.add(files(["a"]));
      for (let i = 0; i < 20; i++) await Promise.resolve();
      expect(up.getSnapshot()[0].state).toBe("retrying");
      expect(vi.getTimerCount()).toBe(1);
      up.clear();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});
