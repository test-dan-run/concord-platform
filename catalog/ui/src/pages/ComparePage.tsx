// Two versions of a dataset side by side (DATASET_CATALOG_PLAN.md §6): files added and removed, files whose
// references changed (segments added, removed and edited, with a word diff per edited segment), and label
// changes. Versions never change, so a comparison is fetched once.

import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import { labelValue, plural, versionPath, type Comparison, type Dataset } from "../lib/datasets";
import { num } from "../lib/format";
import { SegmentLine, Times, WordDiff } from "../components/dataset";
import { Button, Chip, Empty, ErrorNote, Fact, PageHeader, Panel, Select, Spinner } from "../components/ui";

const SHOW = 50;

/** The first SHOW items, and a button that shows the rest: a comparison can list thousands of files. */
function Capped<T>({ items, children }: { items: T[]; children: (shown: T[]) => ReactNode }) {
  const [all, setAll] = useState(false);
  return (
    <>
      {children(all ? items : items.slice(0, SHOW))}
      {items.length > SHOW && !all && (
        <span>
          <Button size="sm" onClick={() => setAll(true)}>Show all {num(items.length)}</Button>
        </span>
      )}
    </>
  );
}

function FileList({ paths, id, n }: { paths: string[]; id: string; n: number }) {
  return (
    <Capped items={paths}>
      {(shown) => (
        <ul className="m-0 pl-5 flex flex-col gap-0.5 text-[13px] num break-all">
          {shown.map((p) => (
            <li key={p}>
              <Link to={`${versionPath(id, n)}?file=${encodeURIComponent(p)}`}>{p}</Link>
            </li>
          ))}
        </ul>
      )}
    </Capped>
  );
}

function Labels({ side }: { side: { files: Record<string, unknown>; speakers: Record<string, unknown> } }) {
  const rows = [
    ...Object.entries(side.files).map(([k, v]) => [k, labelValue(v)]),
    ...Object.entries(side.speakers).flatMap(([sp, l]) => Object.entries((l ?? {}) as Record<string, unknown>).map(([k, v]) => [`${sp}: ${k}`, labelValue(v)])),
  ];
  if (!rows.length) return <span className="text-subtle">none</span>;
  return (
    <span className="flex flex-col gap-0.5">
      {rows.map(([k, v]) => (
        <span key={k} className="num text-[11px]">
          {k}={v}
        </span>
      ))}
    </span>
  );
}

function Changed({ c, a, b }: { c: Comparison["changed"][number]; a: number; b: number }) {
  const s = c.segments;
  return (
    <div className="border border-track rounded-md p-3 flex flex-col gap-2">
      <span className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-semibold num break-all">{c.path}</span>
        {c.audio_changed && <Chip tone="warn">audio changed</Chip>}
        {s && (
          <span className="text-xs text-muted">
            {[s.added.length && `${s.added.length} added`, s.removed.length && `${s.removed.length} removed`, s.edited.length && `${s.edited.length} edited`].filter(Boolean).join(", ")}
          </span>
        )}
      </span>
      {s?.edited.map((e, i) => {
        const moved = e.a.start !== e.b.start || e.a.end !== e.b.end;
        return (
          <div key={`e${i}`} className="flex flex-col gap-1 border-l-2 border-warn pl-2">
            <span className="flex flex-wrap items-center gap-2">
              <Times s={e.b} />
              {moved && (
                <span className="text-[11px] text-subtle">
                  was <Times s={e.a} />
                </span>
              )}
              {e.a.speaker !== e.b.speaker ? (
                <span className="text-[11px] num">
                  <del className="text-bad">{e.a.speaker ?? "none"}</del> <ins className="text-good no-underline">{e.b.speaker ?? "none"}</ins>
                </span>
              ) : (
                e.b.speaker && <span className="text-[11px] text-focus num">{e.b.speaker}</span>
              )}
            </span>
            <WordDiff ops={e.diff} />
            {JSON.stringify(e.a.labels) !== JSON.stringify(e.b.labels) && (
              <span className="grid grid-cols-[auto_1fr] gap-x-2 text-[11px] text-muted">
                <span>v{a}</span>
                <Labels side={{ files: e.a.labels, speakers: {} }} />
                <span>v{b}</span>
                <Labels side={{ files: e.b.labels, speakers: {} }} />
              </span>
            )}
          </div>
        );
      })}
      {s?.added.map((x, i) => <SegmentLine key={`a${i}`} s={x} tone="good" />)}
      {s?.removed.map((x, i) => <SegmentLine key={`r${i}`} s={x} tone="bad" />)}
      {c.labels && (
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[13px]">
          <span className="text-xs text-muted">Labels in v{a}</span>
          <Labels side={c.labels.a} />
          <span className="text-xs text-muted">Labels in v{b}</span>
          <Labels side={c.labels.b} />
        </div>
      )}
    </div>
  );
}

export function ComparePage() {
  const { id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const ds = useQuery({ queryKey: ["dataset", id], queryFn: () => api.get<Dataset>(`/datasets/${id}`), retry: false });
  const versions = ds.data?.versions ?? [];
  const a = Number(params.get("a") ?? versions[1]?.n ?? 0);
  const b = Number(params.get("b") ?? versions[0]?.n ?? 0);
  const ok = a > 0 && b > 0;
  const c = useQuery({ queryKey: ["compare", id, a, b], queryFn: () => api.get<Comparison>(`/datasets/${id}/compare?a=${a}&b=${b}`), enabled: ok, staleTime: Infinity, retry: false });
  const set = (k: "a" | "b", v: string) => setParams({ a: String(a), b: String(b), [k]: v }, { replace: true });
  const d = c.data;
  const name = ds.data?.name ?? "Dataset";
  return (
    <>
      <PageHeader
        crumbs={<><Link to="/datasets">Datasets</Link> / <Link to={`/datasets/${id}`}>{name}</Link> / Compare</>}
        title={`Compare versions of ${name}`}
        subtitle="What changed from the first version to the second: files, references and labels."
        actions={
          <>
            <Select label="From" value={a || ""} onChange={(e) => set("a", e.target.value)}>
              {versions.map((v) => (
                <option key={v.n} value={v.n}>v{v.n}</option>
              ))}
            </Select>
            <Select label="To" value={b || ""} onChange={(e) => set("b", e.target.value)}>
              {versions.map((v) => (
                <option key={v.n} value={v.n}>v{v.n}</option>
              ))}
            </Select>
          </>
        }
      />
      <ErrorNote error={ds.error ?? c.error} />
      {ds.data && versions.length < 2 && <Empty>This dataset has {plural(versions.length, "version")}, so there is nothing to compare yet.</Empty>}
      {c.isLoading && <Spinner label="Comparing" />}
      {d && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px] items-start">
          <div className="flex flex-col gap-5 min-w-0">
            {!d.segments_compared && (
              <span role="status" className="border border-warn/50 bg-warn/10 text-[13px] rounded-md px-3 py-2">
                Segments weren't compared: one of the versions has no references, or references in a format the catalog doesn't read.
              </span>
            )}
            <Panel title="Changed references" subtitle={d.changed.length ? plural(d.changed.length, "file") : undefined}>
              {d.changed.length ? (
                <Capped items={d.changed}>{(shown) => shown.map((x) => <Changed key={x.path} c={x} a={d.a} b={d.b} />)}</Capped>
              ) : (
                <span className="text-[13px] text-muted">No file in both versions changed.</span>
              )}
            </Panel>
            <Panel title="Files added" subtitle={`In v${d.b}, not in v${d.a}`}>
              {d.files.added.length ? <FileList paths={d.files.added} id={id} n={d.b} /> : <span className="text-[13px] text-muted">None</span>}
            </Panel>
            <Panel title="Files removed" subtitle={`In v${d.a}, not in v${d.b}`}>
              {d.files.removed.length ? <FileList paths={d.files.removed} id={id} n={d.a} /> : <span className="text-[13px] text-muted">None</span>}
            </Panel>
          </div>
          <Panel title="Summary">
            <Fact k="Files added" v={num(d.files.added.length)} />
            <Fact k="Files removed" v={num(d.files.removed.length)} />
            <Fact k="Files changed" v={num(d.changed.length)} />
            <Fact k="Files unchanged" v={num(d.files.same)} />
            <Fact k="Segments added" v={num(d.changed.reduce((x, f) => x + (f.segments?.added.length ?? 0), 0))} />
            <Fact k="Segments removed" v={num(d.changed.reduce((x, f) => x + (f.segments?.removed.length ?? 0), 0))} />
            <Fact k="Segments edited" v={num(d.changed.reduce((x, f) => x + (f.segments?.edited.length ?? 0), 0))} />
            <Fact k="Labels added" v={d.labels.added.join(", ") || "none"} />
            <Fact k="Labels removed" v={d.labels.removed.join(", ") || "none"} />
          </Panel>
        </div>
      )}
    </>
  );
}
