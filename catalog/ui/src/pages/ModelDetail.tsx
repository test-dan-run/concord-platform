import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Model } from "../lib/api";
import { ago, rtf, secs } from "../lib/format";
import * as I from "../components/icons";
import { CapabilityChips, gpuLabel, Report, StateCell, VerifyChip } from "../components/model";
import { TagInput } from "../components/TagInput";
import { Button, Chip, ErrorNote, Fact, LinkButton, PageHeader, Panel, Select, Spinner } from "../components/ui";
import { HostedDetail } from "./HostedDetail";

export function usage(m: Model): string {
  const key = m.aliases[0] ?? m.id;
  if (m.contract === "oip-v2") return `from aicore import client          # AICORE_CATALOG=<catalog URL>\nout = client("${key}").infer({...})`;
  if (m.contract === "openai-inspired" && m.api === "text.itn") return `from aicore import client          # AICORE_CATALOG=<catalog URL>\nclient("${key}").itn("meet at three thirty p m", return_spans=True)`;
  if (m.contract === "openai-inspired") return `from aicore import client\nclient("${key}").transform("hello world")`;
  const named = m.hosted ? `\nmodel_name("${key}")${" ".repeat(Math.max(1, 30 - key.length))}# "${m.served_name}", the provider's name` : "";
  return `from aicore import client, model_name   # AICORE_CATALOG=<catalog URL>\nc = client("${key}")                      # openai.OpenAI via the gateway${named}\n# c.${m.api}.create(model=model_name("${key}"), ...)`;
}

export function ModelDetail() {
  const { id = "" } = useParams();
  const qc = useQueryClient();
  const nav = useNavigate();
  const m = useQuery({ queryKey: ["model", id], queryFn: () => api.get<Model>(`/models/${id}`), refetchInterval: (q) => (["pending", "running"].includes(q.state.data?.capabilities?.status ?? "") || ["starting", "queued"].includes(q.state.data?.state ?? "") ? 3000 : 15_000) });
  const registered = useQuery({ queryKey: ["models"], queryFn: () => api.get<Model[]>("/models") });
  const done = () => qc.invalidateQueries();
  const act = useMutation({ mutationFn: (action: "start" | "stop" | "verify") => api.post(`/models/${id}/${action}`), onSuccess: done });
  const unregister = useMutation({ mutationFn: () => api.del(`/models/${id}/register`), onSuccess: () => { void done(); nav("/"); } });

  if (m.isLoading) return <Spinner />;
  if (!m.data) return <ErrorNote error={m.error ?? "Unknown model"} />;
  if (m.data.hosted) return <HostedDetail m={m.data} models={registered.data ?? []} />;
  const d = m.data;
  const c = d.capabilities;
  return (
    <>
      <PageHeader
        crumbs={<><Link to="/">Models</Link> / {d.id}</>}
        title={d.display_name}
        subtitle={d.description ?? undefined}
        actions={
          <>
            <LinkButton to={`/export?model=${d.id}`}><I.Export /> Export</LinkButton>
            {["stopped", "failed"].includes(d.state) ? <Button onClick={() => act.mutate("start")}>Start</Button> : <Button disabled={d.state !== "running" || d.leased_by.length > 0} onClick={() => act.mutate("stop")}>Stop</Button>}
            <Button variant="primary" onClick={() => act.mutate("verify")} disabled={["pending", "running"].includes(c?.status ?? "")}>Verify again</Button>
          </>
        }
      />
      <ErrorNote error={act.error ?? unregister.error} />
      {!d.registered && (
        <div role="status" className="border border-warn/50 bg-warn/10 text-[13px] rounded-md px-3 py-2">
          This platform service is not registered, so apps do not list it. <Link to={`/add?name=${d.id}`}>Add it</Link>.
        </div>
      )}
      <div className="grid gap-5 xl:grid-cols-[1fr_340px] items-start">
        <div className="flex flex-col gap-5 min-w-0">
          <Panel title="Verification" subtitle={c?.build_key ? <>Build <span className="num">{c.build_key}</span>{c.verified_at && <> · {ago(c.verified_at)}</>}</> : "Not verified yet"} actions={<VerifyChip c={c} />}>
            {c?.status === "stale" && <span className="text-[13px] text-warn">This record is for an earlier build. The current build ({d.build_key ?? d.build_pending}) has not been verified yet.</span>}
            {["pending", "running"].includes(c?.status ?? "") && <Spinner label={c?.status === "running" ? "Verifying: waiting for a lease, starting the model if needed, then probing" : "Queued for verification"} />}
            {c && c.report && <Report c={c} />}
            {c && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs text-muted">Detected:</span>
                <CapabilityChips c={c} />
                {c.output_detail.filter((o) => !o.endsWith("_timestamps")).map((o) => <Chip key={o}>{o.replace("_", " ")}</Chip>)}
              </div>
            )}
          </Panel>
          <Panel title="Calling it" subtitle="Through the catalog's gateway: the same protocol as the service, started on demand">
            <Fact k="Gateway URL" v={<span className="text-[11px] break-all">{d.endpoint.gateway_url}{d.endpoint.base_path}</span>} />
            <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{usage(d)}</pre>
          </Panel>
        </div>
        <div className="flex flex-col gap-5">
          <Panel title="State">
            <StateCell m={d} />
            <Fact k="GPU footprint" v={gpuLabel(d)} />
            <Fact k="Cold start" v={d.cold_start_measured ? secs(d.cold_start_s) : `~${secs(d.cold_start_s, 0)} (estimate)`} />
            <Fact k="Real-time factor" v={rtf(c?.rtf)} />
            <Fact k="Calls" v={d.calls.toLocaleString("en-US")} />
            <Fact k="Last used" v={ago(d.last_used)} />
          </Panel>
          <Panel title="Platform">
            <Fact k="Contract / API" v={`${d.contract} / ${d.api}`} />
            <Fact k="Engine" v={d.engine} />
            <Fact k="Aliases" v={d.aliases.join(", ") || "—"} />
            <Fact k="Image" v={<span className="text-[11px] break-all">{d.image ?? "—"}</span>} />
            <Fact k="Build key" v={d.build_key ?? d.build_pending ?? "—"} />
          </Panel>
          {d.registered && <Registration key={d.id} m={d} models={registered.data ?? []} onRemove={() => confirm(`Unregister ${d.id}? Apps stop listing it; its records and index entries are kept.`) && unregister.mutate()} />}
        </div>
      </div>
    </>
  );
}

export function Registration({ m, models, onRemove, note }: { m: Model; models: Model[]; onRemove: () => void; note?: string }) {
  const qc = useQueryClient();
  const [name, setName] = useState(m.display_name === m.id ? "" : m.display_name);
  const [tags, setTags] = useState(m.tags);
  const [parent, setParent] = useState(m.parent_id ?? "");
  const save = useMutation({
    mutationFn: () => api.patch<Model>(`/models/${m.id}`, { display_name: name || m.id, tags, parent_id: parent || null }),
    onSuccess: () => qc.invalidateQueries(),
  });
  return (
    <Panel title="Registration" subtitle={m.registered_at ? `Added ${ago(m.registered_at)}${note ? ` · ${note}` : ""}` : undefined}>
      <label className="flex flex-col gap-1.5 text-xs text-muted">
        Display name
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder={m.id} className="min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg" />
      </label>
      <Select label="Derived from" value={parent} onChange={(e) => setParent(e.target.value)} className="!min-h-10 flex-1">
        <option value="">None</option>
        {models.filter((x) => x.id !== m.id).map((x) => <option key={x.id} value={x.id}>{x.display_name}</option>)}
      </Select>
      <TagInput value={tags} onChange={setTags} />
      <ErrorNote error={save.error} />
      <span className="flex gap-2">
        <Button variant="primary" disabled={save.isPending} onClick={() => save.mutate()}>Save</Button>
        <Button variant="danger" className="ml-auto" onClick={onRemove} title={m.hosted ? "Removes the endpoint, its encrypted key and its registration" : undefined}>
          Unregister
        </Button>
      </span>
    </Panel>
  );
}
