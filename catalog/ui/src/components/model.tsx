// Pieces every model view shares: state, capabilities, the verification report, the GPU memory bar.

import type { Capabilities, Model, PlatformInfo } from "../lib/api";
import { capLabel, pct, timingLabel } from "../lib/format";
import { Chip, cx, Fact, StatusDot } from "./ui";

const STATE_LABEL: Record<string, string> = {
  running: "running",
  stopped: "stopped",
  starting: "starting",
  stopping: "stopping",
  queued: "waiting for compute",
  failed: "failed to start",
  hosted: "hosted",
};

export function StateCell({ m, compact }: { m: Model; compact?: boolean }) {
  return (
    <span className="flex flex-col gap-0.5 min-w-0">
      <span className="flex items-center gap-1.5 text-[13px]">
        <StatusDot state={m.state} />
        {STATE_LABEL[m.state] ?? m.state}
        {m.in_flight > 0 && <span className="num text-[11px] text-muted">· {m.in_flight} in flight</span>}
      </span>
      {!compact && m.hosted && <span className="text-[11px] text-muted num truncate">{m.hosted.host}</span>}
      {!compact && m.leased_by.length > 0 && <span className="text-[11px] text-muted num truncate">leased by {m.leased_by.join(", ")}</span>}
      {!compact && m.state_detail && (
        <span className={cx("text-[11px] truncate", m.state === "failed" || m.hosted ? "text-bad" : "text-muted")} title={m.state_detail}>
          {m.state_detail.split("\n")[0]}
        </span>
      )}
    </span>
  );
}

const VERIFY_TONE = { verified: "good", failed: "bad", stale: "warn", pending: "default", running: "info" } as const;
const VERIFY_LABEL = { verified: "verified", failed: "failed", stale: "stale: new build", pending: "queued", running: "verifying" } as const;

export function VerifyChip({ c }: { c: (Pick<Capabilities, "status"> & Partial<Pick<Capabilities, "verified_at">>) | null }) {
  if (!c) return <Chip>not verified</Chip>;
  return (
    <Chip tone={VERIFY_TONE[c.status]} title={c.verified_at ? `Verified ${c.verified_at}` : undefined}>
      <StatusDot state={c.status} />
      {VERIFY_LABEL[c.status]}
    </Chip>
  );
}

export function CapabilityChips({ c }: { c: Capabilities | null }) {
  if (!c || !c.tasks.length) return <span className="text-xs text-subtle">—</span>;
  return (
    <span className="flex flex-wrap gap-1">
      {c.tasks.map((t) => (
        <Chip key={t} tone="good">{capLabel(t)}</Chip>
      ))}
      {c.timing_level && <Chip>{timingLabel(c.timing_level)}</Chip>}
      {c.tasks.includes("itn") && c.output_detail.includes("spans") && <Chip>spans</Chip>}
    </span>
  );
}

/** The mark and colour of a check line's status. */
export const GLYPH: Record<string, { c: string; t: string }> = {
  pass: { c: "text-good", t: "✓" },
  warn: { c: "text-warn", t: "!" },
  fail: { c: "text-bad", t: "✕" },
  info: { c: "text-muted", t: "i" },
};

export function Report({ c }: { c: Capabilities }) {
  if (!c.report) return null;
  return (
    <div className="flex flex-col gap-1.5">
      {c.report.lines.map((l, k) => (
        <div key={k} className="flex items-start gap-2.5 text-[13px]">
          <span className={cx("num w-4 text-center font-semibold shrink-0", GLYPH[l.status]?.c)} aria-label={l.status}>
            {GLYPH[l.status]?.t}
          </span>
          <span className="w-40 shrink-0 text-muted truncate" title={l.name}>{l.name}</span>
          <span className="min-w-0 break-words">{l.reason}</span>
        </div>
      ))}
      {c.report.sample?.transcript && (
        <span className="text-xs text-muted pt-1">
          Transcript of the clip: <span className="text-fg">“{c.report.sample.transcript}”</span>
        </span>
      )}
      <EnhancementSample c={c} />
      {c.report.sample?.itn && (
        <span className="text-xs text-muted pt-1 flex flex-wrap gap-x-3">
          {c.report.sample.itn.map((x) => (
            <span key={x.input}>
              “{x.input}” → <span className="text-fg num">“{x.output}”</span>
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

/** The speech enhancement probe's numbers: SI-SDR of the degraded input and of the output, and the length difference. */
function EnhancementSample({ c }: { c: Capabilities }) {
  const s = c.report?.sample;
  if (s?.si_sdr_input_db === undefined && s?.length_diff_samples === undefined) return null;
  const diff = s.length_diff_samples ?? 0;
  return (
    <div className="grid gap-x-6 sm:grid-cols-3 pt-1">
      <Fact k="SI-SDR, degraded input" v={s.si_sdr_input_db === undefined ? "—" : `${s.si_sdr_input_db.toFixed(1)} dB`} />
      <Fact k="SI-SDR, output" v={s.si_sdr_output_db === undefined ? "—" : `${s.si_sdr_output_db.toFixed(1)} dB`} />
      <Fact k="Length difference" v={`${diff} samples (${(Math.abs(diff) / 16).toFixed(1)} ms)`} />
    </div>
  );
}

const CAT = ["var(--color-cat-1)", "var(--color-cat-2)", "var(--color-cat-3)", "var(--color-cat-4)", "var(--color-cat-5)", "var(--color-cat-6)"];

/** Running GPU models as segments of the usable memory, with the host reserve at the end. */
export function GpuBar({ p }: { p: PlatformInfo }) {
  const shown = p.gpu_models.slice(0, 5);
  const other = p.gpu_models.slice(5).reduce((a, m) => a + m.fraction, 0);
  const segs = [...shown.map((m, i) => ({ id: m.id, f: m.fraction, color: CAT[i] })), ...(other ? [{ id: "other", f: other, color: CAT[5] }] : [])];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex h-3 rounded overflow-hidden bg-track" role="img" aria-label={`${pct(p.used_fraction)} of the GPU in use`}>
        {segs.map((s) => (
          <span key={s.id} title={`${s.id}: ${(s.f * p.gpu_total_gb).toFixed(1)} GB`} style={{ width: `${s.f * 100}%`, background: s.color }} />
        ))}
        <span className="ml-auto bg-line/60" style={{ width: `${(1 - p.util_limit) * 100}%` }} title={`Host reserve: ${p.reserve_gb} GB`} />
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
        {segs.map((s) => (
          <span key={s.id} className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm" style={{ background: s.color }} />
            {s.id} <span className="num">{(s.f * p.gpu_total_gb).toFixed(1)} GB</span>
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-sm bg-line/60" />
          host reserve <span className="num">{p.reserve_gb} GB</span>
        </span>
        {!segs.length && <span>No GPU models running.</span>}
      </div>
    </div>
  );
}

export function gpuLabel(m: Model): string {
  if (m.hosted) return "remote";
  return m.gpu ? `${m.gpu.gb.toFixed(1)} GB` : "CPU";
}
