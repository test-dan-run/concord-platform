import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, isTextNormaliser, probeFor, type ConnectionTest, type Model, type PlatformInfo } from "../lib/api";
import { ago, rtf, secs } from "../lib/format";
import * as I from "../components/icons";
import { CapabilityChips, gpuLabel, GpuBar, StateCell, VerifyChip } from "../components/model";
import { Button, Chip, cx, Empty, ErrorNote, LinkButton, PageHeader, Panel, Spinner, Toggle } from "../components/ui";

/** Model-type groups, by the probe that applies (capabilities.usable_as, or probeFor before verification);
 * text.transform normalisers join Text normalisation by their declared task (isTextNormaliser). */
const GROUPS = [
  { key: "asr", label: "ASR", desc: "openai / audio.transcriptions · text out", empty: "No ASR models registered.", match: ["asr", "chat-audio"] },
  { key: "seg", label: "VAD and diarisation", desc: "oip-v2 / infer · SEGMENTS out", empty: "No VAD or diarisation models registered.", match: ["segmenter"] },
  { key: "se", label: "Speech enhancement", desc: "oip-v2 / infer · AUDIO out", empty: "No speech enhancement models registered yet.", match: ["enhancer"] },
  { key: "itn", label: "Text normalisation", desc: "openai-inspired / text.itn or text.transform · written form out", empty: "No text normalisation models registered yet.", match: ["itn"] },
] as const;
const OTHER = { key: "other", label: "Other models", desc: "verified on their protocol", empty: "" };

function groupOf(m: Model): string {
  if (isTextNormaliser(m)) return "itn";
  const probe = m.capabilities?.usable_as ?? probeFor(m);
  return GROUPS.find((g) => (g.match as readonly string[]).includes(probe ?? ""))?.key ?? OTHER.key;
}

type Action = "start" | "stop" | "verify" | "test";

export function ModelsPage() {
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const grouped = params.get("group") !== "none";
  const [filter, setFilter] = useState("all");
  const [folded, setFolded] = useState<Record<string, boolean>>({});
  const [tests, setTests] = useState<Record<string, ConnectionTest>>({});
  const models = useQuery({ queryKey: ["models"], queryFn: () => api.get<Model[]>("/models"), refetchInterval: 15_000 });
  const platform = useQuery({ queryKey: ["platform"], queryFn: () => api.get<PlatformInfo>("/platform"), refetchInterval: 15_000 });
  const act = useMutation({
    mutationFn: ({ id, action }: { id: string; action: Action }) => api.post<ConnectionTest>(`/models/${id}/${action}`),
    onSuccess: (data, { id, action }) => {
      if (action === "test") setTests((t) => ({ ...t, [id]: data }));
      void qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
  const rows = (models.data ?? []).filter((m) =>
    filter === "running"
      ? ["running", "starting", "hosted"].includes(m.state)
      : filter === "attention"
        ? m.state === "failed" || ["failed", "stale"].includes(m.capabilities?.status ?? "") || (m.hosted && !!m.state_detail)
        : true,
  );
  const table = (list: Model[], caption: string) => (
    <ModelTable rows={list} caption={caption} tests={tests} pending={act.isPending ? act.variables : undefined} onAction={(id, action) => act.mutate({ id, action })} />
  );

  return (
    <>
      <PageHeader
        crumbs="Models"
        title="Models"
        subtitle="Registered platform services and hosted endpoints. Capabilities come from verification of the exact build; state comes from the scheduler."
        actions={
          <>
            <LinkButton to="/export"><I.Export /> Export</LinkButton>
            <LinkButton to="/add" variant="primary"><I.Plus /> Add model</LinkButton>
          </>
        }
      />
      {platform.data && (
        <Panel title="GPU memory" subtitle={`One GPU model runs at a time and gets the GPU memory free at its start, up to util_limit ${platform.data.util_limit} of ${platform.data.gpu_total_gb} GB. Model swaps so far: ${platform.data.swaps ?? 0}`}>
          <GpuBar p={platform.data} />
        </Panel>
      )}
      <Panel
        actions={
          <>
            <span className="inline-flex items-center gap-2">
              <span aria-hidden className="text-xs text-muted">Group by</span>
              <Toggle
                label="Group by"
                value={grouped ? "type" : "none"}
                onChange={(v) => setParams(v === "none" ? { group: "none" } : {}, { replace: true })}
                options={[
                  { value: "none", label: "None" },
                  { value: "type", label: "Model type" },
                ]}
              />
            </span>
            <Toggle
              label="Filter models"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: "All" },
                { value: "running", label: "Running" },
                { value: "attention", label: "Needs attention" },
              ]}
            />
          </>
        }
        title={`${models.data?.length ?? 0} registered`}
      >
        {models.isLoading && <Spinner />}
        <ErrorNote error={models.error ?? act.error} />
        {models.data?.length === 0 && (
          <Empty>
            No models registered yet. <Link to="/add">Add one from the platform or a hosted endpoint</Link>.
          </Empty>
        )}
        {!!models.data?.length && !grouped && (rows.length ? table(rows, "Registered models") : <Empty>No models match this filter.</Empty>)}
        {!!models.data?.length && grouped && (
          <div className="flex flex-col gap-4">
            {[...GROUPS, OTHER].map((g) => {
              const all = (models.data ?? []).filter((m) => groupOf(m) === g.key);
              if (g.key === OTHER.key && !all.length) return null; // shown only when a service fits no group
              const list = rows.filter((m) => groupOf(m) === g.key);
              const open = !folded[g.key];
              return (
                <div key={g.key} className="flex flex-col">
                  <h3 className="m-0 text-[13px] font-semibold">
                    <button
                      type="button"
                      aria-expanded={open}
                      onClick={() => setFolded((f) => ({ ...f, [g.key]: !f[g.key] }))}
                      className="flex items-center gap-2.5 w-full min-h-10 px-2 border-0 rounded-md bg-canvas/60 text-fg text-[13px] font-semibold text-left cursor-pointer hover:bg-track/60"
                    >
                      <span className="text-muted inline-flex"><I.Chevron open={open} /></span>
                      {g.label}
                      <span className="num inline-flex items-center justify-center min-w-5 px-1.5 rounded-full bg-track text-[11px] font-medium text-muted">{list.length}</span>
                      <span className="num ml-auto text-[11px] font-normal text-subtle hidden sm:inline">{g.desc}</span>
                    </button>
                  </h3>
                  {open && !all.length && (
                    <div className="mt-2 border border-dashed border-line rounded-md p-4 text-center text-[13px] text-muted">
                      {g.empty} <Link to="/add">Add one</Link>
                    </div>
                  )}
                  {open && !!all.length && !list.length && <div className="mt-2 text-xs text-subtle px-2">No {g.label.toLowerCase()} models match this filter.</div>}
                  {open && !!list.length && table(list, `${g.label} models`)}
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </>
  );
}

/** One table of model rows; the grouped and the flat view use the same rows. */
function ModelTable({
  rows,
  caption,
  tests,
  pending,
  onAction,
}: {
  rows: Model[];
  caption: string;
  tests: Record<string, ConnectionTest>;
  pending?: { id: string; action: Action };
  onAction: (id: string, action: Action) => void;
}) {
  const busy = (id: string, action: Action) => pending?.id === id && pending.action === action;
  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="w-full border-collapse text-[13px]">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
            <th scope="col" className="font-medium px-3 py-2 border-b border-line">Model</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line">State</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line">Capabilities</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line">Verification</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">GPU</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Cold start</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">RTF</th>
            <th scope="col" className="font-medium px-3 py-2 border-b border-line">Tags</th>
            <th scope="col" className="px-3 py-2 border-b border-line"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => {
            const test = tests[m.id];
            return (
              <tr key={m.id} className="border-b border-track align-top hover:bg-track/40">
                <td className="px-3 py-2.5">
                  <span className="flex flex-col">
                    <Link to={`/models/${m.id}`} className="!text-fg hover:underline">{m.display_name}</Link>
                    <span className="text-[11px] text-subtle num">
                      {m.id !== m.display_name ? `${m.id} · ` : ""}
                      {m.contract} / {m.api}
                      {m.hosted ? " · hosted" : ""}
                    </span>
                    <span className="text-[11px] text-subtle num" title={m.hosted ? "Build key: URL + model name + contract" : "Build key: image id + config hash + weights hash"}>
                      {m.build_key ?? m.build_pending ?? "—"}
                    </span>
                  </span>
                </td>
                <td className="px-3 py-2.5 max-w-[260px]"><StateCell m={m} /></td>
                <td className="px-3 py-2.5"><CapabilityChips c={m.capabilities} /></td>
                <td className="px-3 py-2.5"><VerifyChip c={m.capabilities} /></td>
                <td className="px-3 py-2.5 text-right num">{gpuLabel(m)}</td>
                <td className="px-3 py-2.5 text-right num" title={m.hosted ? "Hosted: always up" : m.cold_start_measured ? "Median of measured loads" : "Estimate: not measured yet"}>
                  {m.hosted ? "—" : m.cold_start_measured ? secs(m.cold_start_s) : <span className="text-subtle">~{secs(m.cold_start_s, 0)}</span>}
                </td>
                <td className="px-3 py-2.5 text-right num">{rtf(m.capabilities?.rtf)}</td>
                <td className="px-3 py-2.5">
                  <span className="flex flex-wrap gap-1">{m.tags.map((t) => <Chip key={t}>#{t}</Chip>)}</span>
                  {m.last_used && <span className="text-[11px] text-subtle">used {ago(m.last_used)}</span>}
                </td>
                <td className="px-3 py-2.5">
                  <span className="flex gap-1.5 justify-end items-center">
                    {m.hosted ? (
                      <>
                        {test && (
                          <Chip tone={test.ok ? "good" : "bad"} title={test.lines.map((l) => l.reason).join("\n")}>
                            {test.ok ? "reachable" : "test failed"}
                          </Chip>
                        )}
                        <Button size="sm" disabled={busy(m.id, "test")} onClick={() => onAction(m.id, "test")} title="Send one small request with the stored key">
                          Test
                        </Button>
                      </>
                    ) : ["stopped", "failed"].includes(m.state) ? (
                      <Button size="sm" onClick={() => onAction(m.id, "start")} title="Queue a start; it waits its turn like any lease">Start</Button>
                    ) : (
                      <Button
                        size="sm"
                        disabled={m.state !== "running" || m.leased_by.length > 0}
                        title={m.leased_by.length ? `Leased by ${m.leased_by.join(", ")}` : "Stop now (models are also stopped automatically when room is needed)"}
                        onClick={() => onAction(m.id, "stop")}
                      >
                        Stop
                      </Button>
                    )}
                    <Button size="sm" className={cx(busy(m.id, "verify") && "opacity-60")} onClick={() => onAction(m.id, "verify")} title="Verify this build again">
                      Verify
                    </Button>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
