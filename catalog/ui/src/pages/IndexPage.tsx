import { useQuery } from "@tanstack/react-query";
import { api, type App, type IndexStat } from "../lib/api";
import { Empty, ErrorNote, Fact, PageHeader, Panel, Spinner } from "../components/ui";

export function IndexPage() {
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => api.get<App[]>("/apps"), refetchInterval: 30_000 });
  const stats = useQuery({ queryKey: ["index-stats"], queryFn: () => api.get<IndexStat[]>("/index/stats"), refetchInterval: 30_000 });
  const count = Object.fromEntries((stats.data ?? []).map((s) => [s.owner, s.outputs]));
  const owners = [...new Set([...(apps.data ?? []).map((a) => a.name), ...(stats.data ?? []).map((s) => s.owner)])].sort();
  return (
    <>
      <PageHeader
        crumbs="Compute"
        title="Output index"
        subtitle="Apps keep their model outputs; the catalog keeps only where each one is. A model already run on a file in either app is never run again."
      />
      <ErrorNote error={apps.error ?? stats.error} />
      <Panel title="Apps" subtitle="Where other apps read outputs from: POST {base URL}/api/outputs/batch">
        {(apps.isLoading || stats.isLoading) && <Spinner />}
        {!apps.isLoading && !owners.length && <Empty>No app has registered yet. Apps register at startup with POST /api/apps.</Empty>}
        {owners.map((o) => (
          <Fact
            key={o}
            k={<span className="flex flex-col"><span className="text-fg">{o}</span><span className="text-[11px] num">{(apps.data ?? []).find((a) => a.name === o)?.base_url ?? "not registered"}</span></span>}
            v={`${(count[o] ?? 0).toLocaleString("en-US")} outputs`}
          />
        ))}
      </Panel>
      <Panel title="Key">
        <span className="text-[13px] text-muted leading-relaxed">
          Build key (image id + config hash + weights hash) · sha256 of the original audio file · span in seconds (−1, −1 for the whole file) · the app's
          hash of its request parameters. A rebuilt image, a changed manifest or new weights make a new build, so earlier outputs never match it.
        </span>
      </Panel>
    </>
  );
}
