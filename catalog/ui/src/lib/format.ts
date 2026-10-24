// Number and time formatting. Every value rendered with these goes inside a `.num` element.

export const DASH = "—";

export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return `${(v * 100).toFixed(digits)}%`;
}

/** Signed percentage-point delta, e.g. "+1.2" / "−0.4". */
export function deltaPts(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  const x = v * 100;
  if (Math.abs(x) < 0.05) return "±0.0";
  return `${x > 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function secs(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return DASH;
  return `${v.toFixed(digits)} s`;
}

/** mm:ss.mmm */
export function clock(t: number | null | undefined, ms = true): string {
  if (t === null || t === undefined || Number.isNaN(t)) return DASH;
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  const ss = ms ? s.toFixed(3).padStart(6, "0") : Math.floor(s).toString().padStart(2, "0");
  return `${m.toString().padStart(2, "0")}:${ss}`;
}

export function hours(h: number | null | undefined): string {
  if (h === null || h === undefined) return DASH;
  if (h < 1) return `${(h * 60).toFixed(h * 60 < 10 ? 1 : 0)} min`;
  return `${h.toFixed(1)} h`;
}

/** Sizes on the Export page: GB with one decimal, MB below 1 GB. */
export function size(n: number | null | undefined): string {
  if (n === null || n === undefined) return DASH;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.max(1, Math.round(n / 1e6))} MB`;
  return n ? `${Math.max(1, Math.round(n / 1e3))} KB` : "0 MB";
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const t = new Date(iso.endsWith("Z") ? iso : `${iso}Z`).getTime();
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

export function rtf(v: number | null | undefined): string {
  if (v === null || v === undefined) return DASH;
  return v < 0.01 ? v.toFixed(4) : v.toFixed(3);
}

export function capLabel(c: string): string {
  return { asr: "ASR", vad: "VAD", diar: "Diarisation", kws: "KWS", enhance: "Speech enhancement", itn: "ITN" }[c] ?? c;
}

export function timingLabel(t: string | null | undefined): string {
  return { word: "word times", segment: "segment times", text: "text only" }[t ?? ""] ?? DASH;
}
