import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, type Lease, type Usage } from "../lib/api";
import { ago } from "../lib/format";
import { Chip, Empty, ErrorNote, PageHeader, Panel, Spinner, StatusDot } from "../components/ui";

const STATE_TONE = { active: "good", starting: "info", queued: "warn", failed: "bad" } as const;

export function QueuePage() {
  const leases = useQuery({ queryKey: ["leases"], queryFn: () => api.get<Lease[]>("/leases"), refetchInterval: 5_000 });
  const usage = useQuery({ queryKey: ["usage"], queryFn: () => api.get<Usage[]>("/usage"), refetchInterval: 15_000 });
  const granted = (leases.data ?? []).filter((l) => l.state !== "queued");
  const queued = (leases.data ?? []).filter((l) => l.state === "queued");
  const used = (usage.data ?? []).filter((u) => u.calls || u.in_flight).sort((a, b) => (b.last_used ?? "").localeCompare(a.last_used ?? ""));

  return (
    <>
      <PageHeader
        crumbs="Compute"
        title="Queue and leases"
        subtitle="One first-come-first-served queue for every app. A leased model is never stopped; leases expire after three missed 30 s heartbeats."
      />
      <ErrorNote error={leases.error} />
      <div className="grid gap-5 xl:grid-cols-2 items-start">
        <Panel title="Waiting for compute" subtitle="Strict arrival order; a request behind the head passes only if it needs no new GPU memory">
          {leases.isLoading && <Spinner />}
          {!leases.isLoading && !queued.length && <Empty>Nothing is waiting.</Empty>}
          {queued.map((l) => (
            <LeaseRow key={l.id} l={l} />
          ))}
        </Panel>
        <Panel title="Holding models" subtitle="Leases (an app's job) and starts in progress">
          {!leases.isLoading && !granted.length && <Empty>No leases.</Empty>}
          {granted.map((l) => (
            <LeaseRow key={l.id} l={l} />
          ))}
        </Panel>
      </div>
      <Panel title="Gateway usage" subtitle="Calls through the gateway; open calls keep a model from being stopped">
        {!used.length && <Empty>No calls yet.</Empty>}
        {!!used.length && (
          <div className="overflow-x-auto scroll-thin">
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                  <th scope="col" className="font-medium px-3 py-2 border-b border-line">Model</th>
                  <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Calls</th>
                  <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Errors</th>
                  <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">In flight</th>
                  <th scope="col" className="font-medium px-3 py-2 border-b border-line text-right">Last used</th>
                </tr>
              </thead>
              <tbody>
                {used.map((u) => (
                  <tr key={u.id} className="border-b border-track">
                    <td className="px-3 py-2"><Link to={`/models/${u.id}`}>{u.id}</Link></td>
                    <td className="px-3 py-2 text-right num">{u.calls.toLocaleString("en-US")}</td>
                    <td className="px-3 py-2 text-right num">{u.errors ? <span className="text-bad">{u.errors}</span> : 0}</td>
                    <td className="px-3 py-2 text-right num">{u.in_flight}</td>
                    <td className="px-3 py-2 text-right num text-muted">{ago(u.last_used)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}

function LeaseRow({ l }: { l: Lease }) {
  return (
    <div className="flex flex-col gap-1 border-b border-track last:border-0 pb-2.5 last:pb-0">
      <span className="flex flex-wrap items-center gap-2 text-[13px]">
        {l.position && <span className="num text-muted">#{l.position}</span>}
        <span className="num">{l.holder}</span>
        <Chip tone={STATE_TONE[l.state as keyof typeof STATE_TONE] ?? "default"}>
          <StatusDot state={l.state} />
          {l.kind === "start" ? `start · ${l.state}` : l.state}
        </Chip>
        <span className="ml-auto text-[11px] text-subtle num" title={l.expires_at ?? undefined}>
          {l.created_at && `asked ${ago(l.created_at)}`}
        </span>
      </span>
      <span className="flex flex-wrap gap-1">
        {l.models.map((m) => (
          <Link key={m} to={`/models/${m}`} className="text-xs num">{m}</Link>
        ))}
      </span>
      {l.waiting_for && <span className="text-xs text-muted">{l.waiting_for}</span>}
      {l.error && <span className="text-xs text-bad">{l.error}</span>}
    </div>
  );
}
