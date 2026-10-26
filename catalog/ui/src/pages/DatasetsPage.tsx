// The Dataset Catalog's home (catalog/README.md "Pages"): every dataset with its latest version, searched and
// filtered by language, label type and app. The filters live in the URL, so a filtered list can be shared.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, type App } from "../lib/api";
import { appLabel, audioOnly, datasetPath, labelType, plural, versionPath, type Dataset } from "../lib/datasets";
import { ago, hours, num } from "../lib/format";
import * as I from "../components/icons";
import { Chip, Empty, ErrorNote, LinkButton, PageHeader, Panel, Select, Spinner } from "../components/ui";

const AUDIO_ONLY = "none";

/** The apps a dataset touches: who made it, who published its latest version, and who made datasets from it. */
function appsOf(d: Dataset): string[] {
  return [d.created_by, d.latest?.published_by, ...(d.latest?.uses ?? []).map((u) => u.app)].filter((a): a is string => !!a);
}

export function LabelTypeChips({ types }: { types: string[] }) {
  if (!types.length) return <Chip tone="info">audio only</Chip>;
  return (
    <span className="flex flex-wrap gap-1">
      {types.map((t) => (
        <Chip key={t}>{labelType(t)}</Chip>
      ))}
    </span>
  );
}

export function LabelNames({ names, max = 4 }: { names: string[]; max?: number }) {
  if (!names.length) return <span className="text-subtle">none</span>;
  return (
    <span className="flex flex-wrap gap-1" title={names.join(", ")}>
      {names.slice(0, max).map((n) => (
        <span key={n} className="inline-flex items-center rounded bg-track px-1.5 py-px text-[11px] text-muted num">{n}</span>
      ))}
      {names.length > max && <span className="text-[11px] text-subtle">+{names.length - max}</span>}
    </span>
  );
}

export function DatasetsPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const language = params.get("language") ?? "";
  const type = params.get("label_type") ?? "";
  const app = params.get("app") ?? "";
  const [text, setText] = useState(q);
  const set = (k: string, v: string) =>
    setParams(
      (p) => {
        const n = new URLSearchParams(p);
        if (v) n.set(k, v);
        else n.delete(k);
        return n;
      },
      { replace: true },
    );
  useEffect(() => {
    if (text.trim() === q) return;
    const t = setTimeout(() => set("q", text.trim()), 250);
    return () => clearTimeout(t);
  }, [text]);

  const server = new URLSearchParams();
  if (q) server.set("q", q);
  if (language) server.set("language", language);
  if (type && type !== AUDIO_ONLY) server.set("label_type", type);
  if (app) server.set("app", app);
  const query = server.toString();
  const all = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<Dataset[]>("/datasets"), refetchInterval: 60_000 });
  const some = useQuery({ queryKey: ["datasets", query], queryFn: () => api.get<Dataset[]>(`/datasets?${query}`), enabled: query !== "" });
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => api.get<App[]>("/apps") });
  const source = query ? some : all;
  const rows = (source.data ?? []).filter((d) => type !== AUDIO_ONLY || audioOnly(d.latest));
  const filtered = query !== "" || type === AUDIO_ONLY;

  const languages = [...new Set((all.data ?? []).map((d) => d.language).filter((l): l is string => !!l))].sort();
  const appNames = [...new Set([...(all.data ?? []).flatMap(appsOf), ...(apps.data ?? []).map((a) => a.name)])].sort();
  const files = rows.reduce((a, d) => a + (d.latest?.counts.files ?? 0), 0);
  const total = rows.reduce((a, d) => a + (d.latest?.counts.hours ?? 0), 0);

  return (
    <>
      <PageHeader
        crumbs="Dataset Catalog"
        title="Datasets"
        subtitle="Versioned datasets that Concord Bench and Concord Label both read. The audio is stored once and the apps never copy it."
        actions={
          <>
            <LinkButton to="/datasets/import"><I.Download /> Import package</LinkButton>
            <LinkButton to="/datasets/upload" variant="primary"><I.Upload /> Upload audio</LinkButton>
          </>
        }
      />
      <Panel
        title={filtered ? `${rows.length} of ${plural(all.data?.length ?? 0, "dataset")}` : plural(all.data?.length ?? 0, "dataset")}
        subtitle={rows.length ? `Latest versions: ${plural(files, "file")}, ${hours(total)} of audio` : undefined}
        actions={
          <>
            <label className="flex items-center gap-2 bg-canvas border border-line rounded-md px-2 min-h-[34px] text-muted">
              <I.Search size={14} />
              <span className="sr-only">Search datasets</span>
              <input
                type="search"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Search names, descriptions, tags"
                className="bg-transparent border-0 outline-none text-xs text-fg w-[200px] max-w-full"
              />
            </label>
            <Select label="Language" value={language} onChange={(e) => set("language", e.target.value)}>
              <option value="">Any</option>
              {languages.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </Select>
            <Select label="Labels" value={type} onChange={(e) => set("label_type", e.target.value)}>
              <option value="">Any</option>
              <option value="asr">ASR</option>
              <option value="diar">Diarisation</option>
              <option value={AUDIO_ONLY}>Audio only</option>
            </Select>
            <Select label="App" value={app} onChange={(e) => set("app", e.target.value)}>
              <option value="">Any</option>
              {appNames.map((a) => (
                <option key={a} value={a}>{appLabel(a)}</option>
              ))}
            </Select>
          </>
        }
      >
        {source.isLoading && <Spinner />}
        <ErrorNote error={all.error ?? some.error} />
        {all.data?.length === 0 && (
          <Empty>
            No datasets yet. <Link to="/datasets/upload">Upload audio</Link>, <Link to="/datasets/import">import a reference package</Link>, or publish one from Concord Label.
          </Empty>
        )}
        {!!all.data?.length && source.data && !rows.length && <Empty>No datasets match these filters.</Empty>}
        {!!rows.length && <DatasetTable rows={rows} />}
      </Panel>
    </>
  );
}

function DatasetTable({ rows }: { rows: Dataset[] }) {
  const th = "font-medium px-3 py-2 border-b border-line";
  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="w-full border-collapse text-[13px]">
        <caption className="sr-only">Datasets</caption>
        <thead>
          <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
            <th scope="col" className={th}>Dataset</th>
            <th scope="col" className={th}>Language</th>
            <th scope="col" className={th}>Latest</th>
            <th scope="col" className={`${th} text-right`}>Files</th>
            <th scope="col" className={`${th} text-right`}>Hours</th>
            <th scope="col" className={th}>Label types</th>
            <th scope="col" className={th}>Labels</th>
            <th scope="col" className={th}>Last published</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => {
            const v = d.latest;
            return (
              <tr key={d.id} className="border-b border-track align-top hover:bg-track/40">
                <td className="px-3 py-2.5 min-w-[200px]">
                  <span className="flex flex-col gap-0.5">
                    <Link to={datasetPath(d.id)} className="!text-fg hover:underline font-medium">{d.name}</Link>
                    {d.description && <span className="text-xs text-muted line-clamp-2">{d.description}</span>}
                    {!!d.tags.length && (
                      <span className="flex flex-wrap gap-1 pt-0.5">
                        {d.tags.map((t) => (
                          <Chip key={t}>#{t}</Chip>
                        ))}
                      </span>
                    )}
                  </span>
                </td>
                <td className="px-3 py-2.5 num">{d.language ?? <span className="text-subtle">not set</span>}</td>
                <td className="px-3 py-2.5 num">{v ? <Link to={versionPath(d.id, v.n)}>v{v.n}</Link> : <span className="text-subtle">no versions</span>}</td>
                <td className="px-3 py-2.5 text-right num">{v ? num(v.counts.files) : ""}</td>
                <td className="px-3 py-2.5 text-right num">{v ? hours(v.counts.hours ?? 0) : ""}</td>
                <td className="px-3 py-2.5">{v && <LabelTypeChips types={v.label_types} />}</td>
                <td className="px-3 py-2.5 max-w-[220px]">{v && <LabelNames names={v.labels} />}</td>
                <td className="px-3 py-2.5 whitespace-nowrap">
                  {v ? (
                    <span className="flex flex-col">
                      <span>{appLabel(v.published_by)}</span>
                      <span className="text-[11px] text-subtle">{ago(v.published_at)}</span>
                    </span>
                  ) : (
                    <span className="text-[11px] text-subtle">made {ago(d.created_at)}</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
