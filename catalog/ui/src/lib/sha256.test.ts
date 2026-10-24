import { createHash, randomBytes } from "node:crypto";
import { describe, expect, it } from "vitest";
import { hashBlob, Sha256 } from "./sha256";

const node = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

describe("sha256", () => {
  it("matches the standard test vectors", () => {
    expect(new Sha256().hex()).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(new Sha256().update(new TextEncoder().encode("abc")).hex()).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("gives the same digest however the bytes are split", () => {
    const data = randomBytes(5000);
    for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 4999]) {
      const parts = [data.subarray(0, n), data.subarray(n, n + 17), data.subarray(n + 17)];
      const h = new Sha256();
      for (const p of parts) h.update(p);
      expect(h.hex()).toBe(node(data));
    }
    for (const len of [55, 56, 57, 63, 64, 65, 111, 112, 119, 120]) expect(new Sha256().update(data.subarray(0, len)).hex()).toBe(node(data.subarray(0, len)));
  });

  it("hashes a blob in slices with progress", async () => {
    const data = randomBytes(9 * 1024 * 1024 + 3);
    const seen: number[] = [];
    expect(await hashBlob(new Blob([data]), (d) => seen.push(d))).toBe(node(data));
    expect(seen.at(-1)).toBe(data.length);
    expect(seen.length).toBe(3);
  });
});
