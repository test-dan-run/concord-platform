// Incremental SHA-256 (FIPS 180-4) in plain TypeScript. The browser hashes each file before upload, so files the
// store already has aren't sent again. crypto.subtle can't hash in parts, so it would need the whole file in
// memory, and it only exists on https or localhost.

const K = new Int32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
  0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256 {
  private h = new Int32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  private w = new Int32Array(64);
  private buf = new Uint8Array(64);
  private fill = 0;
  private length = 0;

  update(data: Uint8Array): this {
    let i = 0;
    this.length += data.length;
    if (this.fill) {
      const take = Math.min(64 - this.fill, data.length);
      this.buf.set(data.subarray(0, take), this.fill);
      this.fill += take;
      i = take;
      if (this.fill < 64) return this;
      this.block(this.buf, 0);
      this.fill = 0;
    }
    for (; i + 64 <= data.length; i += 64) this.block(data, i);
    if (i < data.length) {
      this.buf.set(data.subarray(i));
      this.fill = data.length - i;
    }
    return this;
  }

  /** The hex digest. The hash can't be updated afterwards. */
  hex(): string {
    const bits = this.length * 8;
    const pad = new Uint8Array((this.fill < 56 ? 56 : 120) - this.fill + 8);
    pad[0] = 0x80;
    const n = pad.length;
    const hi = Math.floor(bits / 0x100000000);
    for (let k = 0; k < 4; k++) {
      pad[n - 1 - k] = (bits >>> (8 * k)) & 0xff;
      pad[n - 5 - k] = (hi >>> (8 * k)) & 0xff;
    }
    this.update(pad);
    let out = "";
    for (const x of this.h) out += (x >>> 0).toString(16).padStart(8, "0");
    return out;
  }

  private block(d: Uint8Array, o: number): void {
    const w = this.w;
    for (let t = 0; t < 16; t++, o += 4) w[t] = (d[o] << 24) | (d[o + 1] << 16) | (d[o + 2] << 8) | d[o + 3];
    for (let t = 16; t < 64; t++) {
      const a = w[t - 15];
      const b = w[t - 2];
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) | 0;
    }
    const h = this.h;
    let a = h[0], b = h[1], c = h[2], e = h[4], f = h[5], g = h[6], x = h[3], y = h[7];
    for (let t = 0; t < 64; t++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const t1 = (y + S1 + ((e & f) ^ (~e & g)) + K[t] + w[t]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      y = g;
      g = f;
      f = e;
      e = (x + t1) | 0;
      x = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }
    h[0] = (h[0] + a) | 0;
    h[1] = (h[1] + b) | 0;
    h[2] = (h[2] + c) | 0;
    h[3] = (h[3] + x) | 0;
    h[4] = (h[4] + e) | 0;
    h[5] = (h[5] + f) | 0;
    h[6] = (h[6] + g) | 0;
    h[7] = (h[7] + y) | 0;
  }
}

const SLICE = 4 * 1024 * 1024;

/** The hex sha256 of a file, read 4 MB at a time, so a large file never sits in memory whole and the page keeps
 *  answering between slices. */
export async function hashBlob(blob: Blob, onProgress?: (done: number) => void, signal?: AbortSignal): Promise<string> {
  const h = new Sha256();
  for (let off = 0; off < blob.size; off += SLICE) {
    signal?.throwIfAborted();
    h.update(new Uint8Array(await blob.slice(off, off + SLICE).arrayBuffer()));
    onProgress?.(Math.min(blob.size, off + SLICE));
  }
  return h.hex();
}
