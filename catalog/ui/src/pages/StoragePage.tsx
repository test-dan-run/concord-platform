// Storage (DATASET_CATALOG_PLAN.md §4.3, §6): how big the audio store is, how much of its audio versions list, apps
// hold, or nothing needs, and each app's holds. Clean-up always shows a dry run first; it deletes only audio that
// no version lists, no app holds, and that has been in the store for more than 24 hours.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { appLabel, plural, type GcResult, type Group, type StoreStats } from "../lib/datasets";
import { ago, bytes, num, pct } from "../lib/format";
import { Bar, Button, Empty, ErrorNote, Fact, PageHeader, Panel, Spinner } from "../components/ui";

const PARTS = [
  { key: "in_versions", label: "In versions", note: "listed by at least one version", color: "var(--color-primary)" },
  { key: "held_only", label: "Held only", note: "in no version, but an app keeps it in one of its datasets", color: "var(--color-secondary)" },
  { key: "neither", label: "In neither", note: "in no version and held by no app: clean-up deletes it after 24 hours", color: "var(--color-warn)" },
] as const;

function Stacked({ s }: { s: StoreStats }) {
  const total = Math.max(1, s.audio.bytes);
  return (
    <span className="flex h-2.5 rounded bg-track overflow-hidden" aria-hidden>
      {PARTS.map((p) => (
        <span key={p.key} style={{ width: `${(100 * s[p.key].bytes) / total}%`, background: p.color }} />
      ))}
    </span>
  );
}

function Line({ label, note, g, color }: { label: string; note: string; g: Group; color?: string }) {
  return (
    <div className="flex items-start gap-3 py-1.5 border-b border-track last:border-0 text-[13px]">
      <span className="size-2.5 rounded-sm mt-1 shrink-0" style={{ background: color ?? "var(--color-subtle)" }} />
      <span className="flex flex-col flex-1 min-w-0">
        <span>{label}</span>
        <span className="text-xs text-muted">{note}</span>
      </span>
      <span className="num text-right">
        {bytes(g.bytes)}
        <span className="block text-[11px] text-muted">{plural(g.files, "file")}</span>
      </span>
    </div>
  );
}

export function StoragePage() {
  const qc = useQueryClient();
  const stats = useQuery({ queryKey: ["store-stats"], queryFn: () => api.get<StoreStats>("/store/stats"), refetchInterval: 30_000 });
  const [dry, setDry] = useState<GcResult | null>(null);
  const [done, setDone] = useState<GcResult | null>(null);
  const check = useMutation({
    mutationFn: () => api.post<GcResult>("/store/gc", { dry_run: true }),
    onSuccess: (r) => (setDry(r), setDone(null)),
  });
  const clean = useMutation({
    mutationFn: () => api.post<GcResult>("/store/gc", { dry_run: false }),
    onSuccess: (r) => {
      setDone(r);
      setDry(null);
      void qc.invalidateQueries({ queryKey: ["store-stats"] });
    },
  });
  const s = stats.data;
  const used = s ? (s.disk.total - s.disk.free) / s.disk.total : 0;
  return (
    <>
      <PageHeader
        crumbs={<><Link to="/datasets">Datasets</Link> / Storage</>}
        title="Storage"
        subtitle="The audio store keeps each original audio file once, under the sha256 of its bytes. Versions list files; apps hold the files their own datasets use."
      />
      {stats.isLoading && <Spinner />}
      <ErrorNote error={stats.error} />
      {s && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px] items-start">
          <div className="flex flex-col gap-5 min-w-0">
            <Panel title="Audio" subtitle={`${bytes(s.audio.bytes)} in ${plural(s.audio.files, "file")}`}>
              <Stacked s={s} />
              <div>
                {PARTS.map((p) => (
                  <Line key={p.key} label={p.label} note={p.note} g={s[p.key]} color={p.color} />
                ))}
                {!!s.partial.files && <Line label="Partial uploads" note="uploads that haven't finished; clean-up deletes them 24 hours after their last write" g={s.partial} />}
              </div>
            </Panel>
            <Panel title="Holds per app" subtitle="Audio an app keeps in one of its datasets, whether or not a version lists it">
              {s.holds.length ? (
                s.holds.map((h) => <Fact key={h.app} k={appLabel(h.app)} v={`${bytes(h.bytes)} · ${plural(h.files, "file")}`} />)
              ) : (
                <Empty>No app holds any audio yet.</Empty>
              )}
            </Panel>
            <Panel title="Clean-up" subtitle="Deletes audio that no version lists and no app holds, once it is more than 24 hours old. A dry run comes first, every time.">
              <span className="flex flex-wrap gap-2">
                <Button onClick={() => check.mutate()} disabled={check.isPending || clean.isPending}>
                  {dry ? "Check again" : "Check what clean-up would delete"}
                </Button>
              </span>
              <ErrorNote error={check.error ?? clean.error} />
              {done && (
                <span role="status" className="border border-good/50 bg-good/10 text-[13px] rounded-md px-3 py-2">
                  Deleted {plural(done.files, "audio file")}, {bytes(done.bytes)}.
                </span>
              )}
              {dry && !dry.files && <span className="text-[13px] text-muted">Nothing to clean up. Every file is in a version, held by an app, or less than 24 hours old.</span>}
              {dry && !!dry.files && (
                <div className="flex flex-col gap-2">
                  <span className="text-[13px]">
                    Clean-up would delete {plural(dry.files, "audio file")}, {bytes(dry.bytes)}.
                  </span>
                  <div className="overflow-x-auto scroll-thin">
                    <table className="w-full border-collapse text-[13px]">
                      <caption className="sr-only">Files clean-up would delete</caption>
                      <thead>
                        <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                          <th scope="col" className="font-medium px-3 py-2 border-b border-line">sha256</th>
                          <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Size</th>
                          <th scope="col" className="font-medium px-3 py-2 border-b border-line">State</th>
                          <th scope="col" className="font-medium px-3 py-2 border-b border-line">Stored</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dry.sample.map((r) => (
                          <tr key={r.sha256} className="border-b border-track">
                            <td className="px-3 py-1.5 num text-[11px]" title={r.sha256}>
                              {r.sha256.slice(0, 16)}
                              {r.ext}
                            </td>
                            <td className="px-3 py-1.5 num text-right">{bytes(r.size)}</td>
                            <td className="px-3 py-1.5">{r.state === "partial" ? "partial upload" : "stored"}</td>
                            <td className="px-3 py-1.5 text-xs text-muted">{ago(r.created_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {dry.files > dry.sample.length && <span className="text-xs text-muted">and {num(dry.files - dry.sample.length)} more</span>}
                  <span>
                    <Button variant="danger" disabled={clean.isPending} onClick={() => clean.mutate()}>
                      Delete {plural(dry.files, "file")}
                    </Button>
                  </span>
                  <span className="text-xs text-muted">The rules are checked again when you press it, so audio a version or a hold picked up since the dry run is kept.</span>
                </div>
              )}
            </Panel>
          </div>
          <Panel title="Store">
            <Fact k="Folder" v={<span className="text-[11px] break-all">{s.root}</span>} />
            <Fact k="Datasets" v={num(s.datasets)} />
            <Fact k="Versions" v={num(s.versions)} />
            <Fact k="Audio files" v={num(s.audio.files)} />
            <Fact k="Audio size" v={bytes(s.audio.bytes)} />
            <div className="flex flex-col gap-1.5 pt-2">
              <span className="flex justify-between text-xs text-muted">
                <span>Disk</span>
                <span className="num">
                  {bytes(s.disk.free)} free of {bytes(s.disk.total)}
                </span>
              </span>
              <Bar value={used} color={used > 0.9 ? "var(--color-warn)" : "var(--color-primary)"} />
              <span className="text-[11px] text-muted num">{pct(used)} used</span>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}
