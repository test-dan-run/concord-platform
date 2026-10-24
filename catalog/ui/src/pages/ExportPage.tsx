// Export: pack platform models with their exact builds into a bundle, or push their images to a registry.
// The job runs in the catalog's background; this page shows its progress, failure and finished bundles.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type ExportJob, type ExportModel } from "../lib/api";
import { ago, size } from "../lib/format";
import * as I from "../components/icons";
import { VerifyChip } from "../components/model";
import { AnchorButton, Bar, Button, cx, Empty, ErrorNote, Fact, PageHeader, Panel, Spinner } from "../components/ui";

type Options = { destination: "bundle" | "registry"; registry: string; images: boolean; weights: boolean; verification: boolean };

const LIVE = ["queued", "running"];

function elapsed(from: string, to?: string | null): string {
  const s = Math.max(0, Math.round(((to ? new Date(to).getTime() : Date.now()) - new Date(from).getTime()) / 1000));
  return `${Math.floor(s / 60)} min ${s % 60} s`;
}

function tree(name: string, picked: ExportModel[], o: Options): string {
  const n = picked.length;
  const reg = o.registry.trim().replace(/\/+$/, "") || "registry.example.internal/models";
  const imgs = picked.reduce((a, m) => a + (m.image_size ?? 0), 0);
  let t = `${name}/\n  manifest.json           ${n} model${n === 1 ? "" : "s"}, build keys\n`;
  if (o.images && n) t += o.destination === "bundle" ? `  images.tar              up to ${size(imgs)}, shared layers once\n` : `  images → ${reg}  ${n} pushed\n`;
  for (const m of picked) {
    t += `  models/${m.id}/\n    service.yaml\n    registration.json\n`;
    if (o.weights && m.weights_size && !m.weights_note) t += `    weights/              ${size(m.weights_size)}\n`;
    if (o.verification) t += "    verification.json\n";
  }
  if (!n) t += "  (no models selected)";
  return t;
}

export function ExportPage() {
  const qc = useQueryClient();
  const [params] = useSearchParams();
  const models = useQuery({ queryKey: ["export-models"], queryFn: () => api.get<ExportModel[]>("/export/models") });
  const jobs = useQuery({
    queryKey: ["exports"],
    queryFn: () => api.get<ExportJob[]>("/exports"),
    refetchInterval: (q) => ((q.state.data ?? []).some((j) => LIVE.includes(j.state)) ? 1000 : 15_000),
  });
  const [sel, setSel] = useState<Record<string, boolean>>(() => (params.get("model") ? { [params.get("model")!]: true } : {}));
  const [o, setO] = useState<Options>({ destination: "bundle", registry: "", images: true, weights: true, verification: true });
  const [dismissed, setDismissed] = useState<string | null>(null);
  const [, tick] = useState(0);
  const running = (jobs.data ?? []).find((j) => LIVE.includes(j.state));
  const last = (jobs.data ?? []).find((j) => !LIVE.includes(j.state));
  const failed = last && last.state === "failed" && last.id !== dismissed ? last : null;
  const done = last && last.state === "done" && last.available && last.id !== dismissed && Date.now() - new Date(last.finished_at ?? 0).getTime() < 3_600_000 ? last : null;
  const bundles = (jobs.data ?? []).filter((j) => j.state === "done" && j.available);
  const locked = !!running;
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [running]);
  const refresh = () => qc.invalidateQueries({ queryKey: ["exports"] });
  const start = useMutation({ mutationFn: (b: unknown) => api.post<ExportJob>("/exports", b), onSuccess: refresh });
  const remove = useMutation({ mutationFn: (id: string) => api.del(`/exports/${id}`), onSuccess: refresh });

  const list = (models.data ?? []).filter((m) => m.registered || sel[m.id]); // registered models, as on the Models page
  const picked = list.filter((m) => sel[m.id] && m.exportable);
  const n = picked.length;
  const bundle = o.destination === "bundle";
  const imgSum = picked.reduce((a, m) => a + (m.image_size ?? 0), 0);
  const wSum = picked.reduce((a, m) => a + (m.weights_note ? 0 : m.weights_size), 0);
  const total = (bundle && o.images ? imgSum : 0) + (o.weights ? wSum : 0);
  const hint = !n ? "Select at least one model." : !bundle && o.images && !o.registry.trim() ? "Enter the target registry to push to." : "";
  const set = (patch: Partial<Options>) => !locked && setO((x) => ({ ...x, ...patch }));
  const pick = (ids: string[]) => !locked && setSel(Object.fromEntries(ids.map((id) => [id, true])));
  const run = (b: unknown) => {
    setDismissed(null);
    start.mutate(b);
  };
  const name = running?.name ?? `catalog-export-${new Date().toISOString().slice(0, 10)}`;

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/">Models</Link> / Export</>}
        title="Export models"
        subtitle="Pick the models to move to another environment. Each one goes with its exact build: image, weights, registration and verification record."
      />
      {done && (
        <section role="status" aria-labelledby="done-title" className="flex flex-wrap items-center gap-3 border border-good/50 bg-good/10 rounded-lg p-4">
          <span className="text-good inline-flex"><I.Check size={20} /></span>
          <div className="flex flex-col gap-0.5 min-w-0">
            <h2 id="done-title" className="m-0 text-sm font-semibold">
              {done.request.destination === "bundle"
                ? `Bundle ready · ${done.request.models.length} model${done.request.models.length === 1 ? "" : "s"}`
                : `Pushed ${done.request.images ? done.request.models.length : 0} image${done.request.models.length === 1 ? "" : "s"} · manifest ready`}
            </h2>
            <span className="text-xs text-muted num">
              {done.name}.tar · {size(done.bytes_written)} written{done.request.destination === "bundle" && done.request.images ? " (shared layers once)" : ""}
            </span>
          </div>
          <span className="ml-auto flex gap-2">
            <Button variant="ghost" onClick={() => setDismissed(done.id)}>Dismiss</Button>
            <AnchorButton href={`/api/exports/${done.id}/download`} variant="primary" download><I.Download /> Download bundle</AnchorButton>
          </span>
        </section>
      )}
      {failed && (
        <section role="alert" aria-labelledby="fail-title" className="flex flex-wrap items-start gap-3 border border-bad/50 bg-bad/10 rounded-lg p-4">
          <span className="text-bad inline-flex"><I.Alert size={20} /></span>
          <div className="flex flex-col gap-1 min-w-0 flex-1">
            <h2 id="fail-title" className="m-0 text-sm font-semibold">Export failed while {failed.failed_phase ?? "starting"}</h2>
            <span className="text-[13px] break-words">{failed.error}</span>
            <span className="text-xs text-muted">
              {failed.phase_count ? `Phase ${failed.phase_index} of ${failed.phase_count}. ` : ""}The partial bundle was removed. Free some space, or set CATALOG_EXPORT_DIR to a larger disk, then retry.
            </span>
          </div>
          <span className="flex gap-2">
            <Button onClick={() => setDismissed(failed.id)}>Dismiss</Button>
            <Button variant="primary" disabled={locked || start.isPending} onClick={() => run(failed.request)}>Retry</Button>
          </span>
        </section>
      )}
      <ErrorNote error={models.error ?? jobs.error ?? start.error ?? remove.error} />
      <div className="grid gap-5 xl:grid-cols-[1fr_360px] items-start">
        <div className="flex flex-col gap-5 min-w-0">
          <Panel
            title="Models"
            subtitle={`${n} of ${list.filter((m) => m.exportable).length} selected`}
            actions={
              <>
                <Button size="sm" disabled={locked} onClick={() => pick(list.filter((m) => m.exportable).map((m) => m.id))}>Select all</Button>
                <Button size="sm" disabled={locked} onClick={() => pick(list.filter((m) => m.exportable && m.state === "running").map((m) => m.id))}>Running only</Button>
                <Button size="sm" disabled={locked} onClick={() => pick([])}>Clear</Button>
              </>
            }
          >
            {models.isLoading && <Spinner label="Reading image and weights sizes" />}
            {!!list.length && (
              <div className="overflow-x-auto scroll-thin">
                <table className="w-full border-collapse text-[13px]">
                  <caption className="sr-only">Models to export</caption>
                  <thead>
                    <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line w-10">Include</th>
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line">Model</th>
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line">Image</th>
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Image size</th>
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Weights</th>
                      <th scope="col" className="font-medium px-3 py-2 border-b border-line">Verification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((m) => (
                      <tr key={m.id} className={cx("border-b border-track align-top", sel[m.id] && m.exportable && "bg-primary/5", !m.exportable && "opacity-60")} title={m.reason ?? undefined}>
                        <td className="px-3 py-2.5">
                          <input
                            type="checkbox"
                            id={`pick-${m.id}`}
                            checked={!!sel[m.id] && m.exportable}
                            disabled={locked || !m.exportable}
                            onChange={() => setSel((x) => ({ ...x, [m.id]: !x[m.id] }))}
                            className="accent-primary size-4"
                          />
                        </td>
                        <td className="px-3 py-2.5">
                          <label htmlFor={`pick-${m.id}`} className="flex flex-col cursor-pointer">
                            <span>{m.display_name}{!m.registered && <span className="text-[11px] text-subtle"> · not registered</span>}</span>
                            <span className="text-[11px] text-subtle num">{m.contract} / {m.api} · {m.build_key ?? "no build key"}</span>
                            {m.reason && <span className="text-[11px] text-warn">{m.reason}</span>}
                          </label>
                        </td>
                        <td className="px-3 py-2.5 text-[11px] num text-muted max-w-[240px] truncate" title={m.image}>{m.image}</td>
                        <td className="px-3 py-2.5 text-right num whitespace-nowrap">{size(m.image_size)}</td>
                        <td className="px-3 py-2.5 text-right num whitespace-nowrap">{m.weights_note ? <span className="text-[11px] text-subtle">{m.weights_note}</span> : size(m.weights_size)}</td>
                        <td className="px-3 py-2.5">{m.verification ? <VerifyChip c={{ status: m.verification }} /> : <span className="text-xs text-subtle">not verified</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <span className="text-xs text-muted">
              Image layers shared between models are written once, so the images file is smaller than the sum of the image sizes. On the target, <span className="num text-fg">python -m catalog import &lt;bundle&gt;</span> loads the bundle and re-verifies each build; a matching build key keeps its existing outputs. Hosted models are not exported: they have no image or weights.
            </span>
          </Panel>
          <Panel title="Destination">
            <fieldset disabled={locked} className="m-0 p-0 border-0 grid gap-3 sm:grid-cols-2">
              <legend className="sr-only">Where to export to</legend>
              {(
                [
                  ["bundle", "Bundle file", "One .tar to carry over, for air-gapped or offline targets"],
                  ["registry", "Push to a registry", "Images are retagged and pushed; the bundle holds only the manifest"],
                ] as const
              ).map(([v, t, d]) => (
                <label key={v} className={cx("flex items-start gap-2.5 rounded-md p-3 cursor-pointer border", o.destination === v ? "border-primary bg-primary/10" : "border-line")}>
                  <input type="radio" name="dest" checked={o.destination === v} onChange={() => set({ destination: v })} className="mt-0.5 accent-primary" />
                  <span className="flex flex-col">
                    <span className="text-[13px]">{t}</span>
                    <span className="text-xs text-muted">{d}</span>
                  </span>
                </label>
              ))}
            </fieldset>
            {!bundle && (
              <>
                <label className="flex flex-col gap-1.5 text-xs text-muted">
                  Target registry
                  <input type="text" value={o.registry} onChange={(e) => set({ registry: e.target.value })} disabled={locked} placeholder="registry.example.internal/models" className="min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg num" />
                </label>
                <span className="text-[11px] text-subtle">Pushing uses this host's Docker login for the registry (~/.docker/config.json, mounted read-only into the catalog).</span>
              </>
            )}
          </Panel>
          <Panel title="Include">
            <div className="flex flex-col gap-1">
              {(
                [
                  ["images", "Container images", `up to ${size(imgSum)}`],
                  ["weights", "Model weights", size(wSum)],
                  ["verification", "Verification records", "< 1 MB"],
                ] as const
              ).map(([k, t, v]) => (
                <label key={k} className="flex items-center gap-2.5 min-h-9 text-[13px] cursor-pointer">
                  <input type="checkbox" checked={o[k]} disabled={locked} onChange={() => set({ [k]: !o[k] })} className="accent-primary size-4" />
                  {t}
                  <span className="ml-auto num text-xs text-muted">{v}</span>
                </label>
              ))}
              <label className="flex items-center gap-2.5 min-h-9 text-[13px] text-muted">
                <input type="checkbox" checked disabled className="accent-primary size-4" />
                Registration, tags and service config
                <span className="ml-auto text-xs">always</span>
              </label>
            </div>
            {!o.weights && n > 0 && <span className="text-xs text-warn">Without weights the target needs them in place before the models start.</span>}
          </Panel>
        </div>
        <div className="flex flex-col gap-5">
          <Panel title="Summary">
            <div>
              <Fact k="Models" v={n} />
              <Fact k="Images" v={o.images ? new Set(picked.map((m) => m.image)).size : 0} />
              <Fact k={bundle ? "Bundle size" : "Bundle size (images pushed separately)"} v={n ? `up to ${size(total)}` : "—"} />
            </div>
            <pre aria-label="Export contents" className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre overflow-x-auto scroll-thin">{tree(name, picked, o)}</pre>
            {running ? (
              <div role="status" aria-live="polite" className="flex flex-col gap-2">
                <span className="flex items-baseline justify-between gap-2 text-[13px]">
                  <span className="first-letter:uppercase">{running.phase}</span>
                  <span className="text-[11px] text-muted num whitespace-nowrap">{running.phase_count ? `Phase ${running.phase_index} of ${running.phase_count}` : "Starting"}</span>
                </span>
                <span role="progressbar" aria-label="Export progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={running.bytes_expected ? Math.round((100 * running.bytes_done) / running.bytes_expected) : 0}>
                  <Bar value={running.bytes_expected ? running.bytes_done / running.bytes_expected : 0} />
                </span>
                <span className="flex justify-between text-[11px] text-muted num">
                  <span>{size(running.bytes_done)} of up to {size(running.bytes_expected)}</span>
                  <span>{elapsed(running.created_at)}</span>
                </span>
                <Button variant="danger" disabled={remove.isPending} onClick={() => remove.mutate(running.id)}>Cancel export</Button>
                <span className="text-[11px] text-subtle">The options are locked until the export finishes. Cancelling deletes the partial bundle.</span>
              </div>
            ) : (
              <>
                <Button variant="primary" disabled={!!hint || start.isPending} onClick={() => run({ models: picked.map((m) => m.id), ...o, registry: o.registry.trim() || null })}>
                  <I.Export /> {bundle ? "Create bundle" : "Push and create manifest"}
                </Button>
                {hint && <span className="text-xs text-muted text-center">{hint}</span>}
              </>
            )}
          </Panel>
          <Panel title="Bundles on this host" subtitle="run/catalog/exports (CATALOG_EXPORT_DIR)">
            {!bundles.length && <Empty>No bundles yet.</Empty>}
            {bundles.map((b) => (
              <div key={b.id} className="flex flex-wrap items-center gap-2 border-b border-track last:border-0 pb-2.5 last:pb-0">
                <span className="flex flex-col min-w-0">
                  <span className="text-[13px] num truncate">{b.name}.tar</span>
                  <span className="text-[11px] text-muted">
                    {b.request.models.length} model{b.request.models.length === 1 ? "" : "s"} · {size(b.bytes_written)} · {ago(b.finished_at)}
                  </span>
                </span>
                <span className="ml-auto flex gap-1.5">
                  <AnchorButton href={`/api/exports/${b.id}/download`} download>Download</AnchorButton>
                  <Button variant="danger" onClick={() => confirm(`Delete ${b.name}? The bundle is removed from this host.`) && remove.mutate(b.id)}>Delete</Button>
                </span>
              </div>
            ))}
          </Panel>
        </div>
      </div>
    </>
  );
}
