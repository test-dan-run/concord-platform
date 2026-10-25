// One version of a dataset (DATASET_CATALOG_PLAN.md §6): its files with their segments and labels, a player that
// reads the original audio from the store, the references of the file picked, and label statistics. A version
// never changes, so its files and references are fetched once; the version record itself is refreshed, so the
// page notices when the version is deleted.

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, notFound, type App } from "../lib/api";
import {
  allFiles,
  appLabel,
  audioUrl,
  labelBase,
  labelFileUrl,
  labelStats,
  labelType,
  labelValue,
  packageUrl,
  parentLabel,
  plural,
  type Dataset,
  type FileRefs,
  type Segment,
  type Version,
  type VersionFile,
} from "../lib/datasets";
import { ago, clock, hours, num } from "../lib/format";
import * as I from "../components/icons";
import { LabelChips, Times, UseList } from "../components/dataset";
import { AnchorButton, Button, Chip, cx, Empty, ErrorNote, Fact, LinkButton, PageHeader, Panel, Spinner } from "../components/ui";
import { LabelNames } from "./DatasetsPage";

const PAGE = 100;

export function VersionPage() {
  const { id = "", n: nParam = "" } = useParams();
  const n = Number(nParam);
  const [params, setParams] = useSearchParams();
  const sel = params.get("file");
  const ds = useQuery({ queryKey: ["dataset", id], queryFn: () => api.get<Dataset>(`/datasets/${id}`), retry: false });
  const v = useQuery({ queryKey: ["dataset", id, "v", n], queryFn: () => api.get<Version>(`/datasets/${id}/versions/${n}`), retry: false });
  const files = useQuery({ queryKey: ["version-files", id, n], queryFn: () => allFiles(id, n), staleTime: Infinity, enabled: !!v.data });
  const apps = useQuery({ queryKey: ["apps"], queryFn: () => api.get<App[]>("/apps") });
  const label = labelBase(apps.data);
  const name = ds.data?.name ?? "Dataset";
  const crumbs = (
    <>
      <Link to="/datasets">Datasets</Link> / <Link to={`/datasets/${id}`}>{name}</Link> / v{nParam}
    </>
  );
  const picked = files.data?.find((f) => f.path === sel) ?? null;
  const pick = (path: string) => setParams((p) => ({ ...Object.fromEntries(p), file: path }), { replace: true });

  if (v.isLoading) return <Spinner />;
  if (!v.data || notFound(v.error))
    return (
      <>
        <PageHeader crumbs={crumbs} title={`${name} v${nParam}`} />
        <ErrorNote error={v.error ?? "Unknown version"} />
        <span className="text-[13px] text-muted">
          The version may have been deleted. <Link to={`/datasets/${id}`}>See the dataset's versions</Link>.
        </span>
      </>
    );
  const ver = v.data;
  return (
    <>
      <PageHeader
        crumbs={crumbs}
        title={`${name} v${ver.n}`}
        subtitle={`Published by ${appLabel(ver.published_by)}, ${ago(ver.published_at)}. Parent: ${parentLabel(ver)}.`}
        actions={
          <>
            {ver.parent_n !== null && !ver.parent_deleted && <LinkButton to={`/datasets/${id}/compare?a=${ver.parent_n}&b=${ver.n}`}>Compare with v{ver.parent_n}</LinkButton>}
            <AnchorButton href={packageUrl(id, ver.n, false)} download><I.Download /> Package</AnchorButton>
            <AnchorButton href={packageUrl(id, ver.n, true)} download><I.Download /> With audio</AnchorButton>
          </>
        }
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_400px] items-start">
        <div className="flex flex-col gap-5 min-w-0">
          {picked && <FilePanel key={picked.path} id={id} ver={ver} f={picked} label={label} onClose={() => setParams({}, { replace: true })} />}
          <FilesPanel files={files.data} loading={files.isLoading} error={files.error} sel={sel} onPick={pick} total={ver.counts.files} />
        </div>
        <div className="flex flex-col gap-5">
          <VersionFacts ver={ver} />
          <LabelsPanel ver={ver} files={files.data} />
        </div>
      </div>
    </>
  );
}

function FilesPanel({ files, loading, error, sel, onPick, total }: { files?: VersionFile[]; loading: boolean; error: unknown; sel: string | null; onPick: (p: string) => void; total: number }) {
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const rows = useMemo(() => (files ?? []).filter((f) => !q || f.path.toLowerCase().includes(q.toLowerCase())), [files, q]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const shown = rows.slice(page * PAGE, page * PAGE + PAGE);
  const th = "font-medium px-3 py-2 border-b border-line";
  useEffect(() => setPage(0), [q]);
  return (
    <Panel
      title="Files"
      subtitle={`${plural(total, "file")}. Pick one to play it and read its references.`}
      actions={
        <label className="flex items-center gap-2 bg-canvas border border-line rounded-md px-2 min-h-[34px] text-muted">
          <I.Search size={14} />
          <span className="sr-only">Filter files</span>
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter by path" className="bg-transparent border-0 outline-none text-xs text-fg w-[160px]" />
        </label>
      }
    >
      {loading && <Spinner label="Reading the file list" />}
      <ErrorNote error={error} />
      {files && !rows.length && <Empty>No files match.</Empty>}
      {!!shown.length && (
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full border-collapse text-[13px]">
            <caption className="sr-only">Files</caption>
            <thead>
              <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                <th scope="col" className={th}>Path</th>
                <th scope="col" className={`${th} text-right`}>Duration</th>
                <th scope="col" className={`${th} text-right`}>Sample rate</th>
                <th scope="col" className={`${th} text-right`}>Segments</th>
                <th scope="col" className={th}>Speakers</th>
                <th scope="col" className={th}>Labels</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((f) => (
                <tr key={f.path} className={cx("border-b border-track align-top", f.path === sel ? "bg-active" : "hover:bg-track/40")} aria-selected={f.path === sel}>
                  <td className="px-3 py-2 min-w-[200px]">
                    <button type="button" onClick={() => onPick(f.path)} className="bg-transparent border-0 p-0 text-left text-focus hover:underline cursor-pointer text-[13px] break-all inline-flex items-center gap-1.5">
                      <I.Play size={12} className="shrink-0" />
                      {f.path}
                    </button>
                  </td>
                  <td className="px-3 py-2 text-right num whitespace-nowrap">{clock(f.duration, false)}</td>
                  <td className="px-3 py-2 text-right num whitespace-nowrap">{num(f.sample_rate / 1000, f.sample_rate % 1000 ? 1 : 0)} kHz</td>
                  <td className="px-3 py-2 text-right num">{f.segments === null ? <span className="text-subtle" title="References in a format the catalog doesn't read">?</span> : num(f.segments)}</td>
                  <td className="px-3 py-2 num text-xs">{f.speakers?.length ? f.speakers.join(", ") : ""}</td>
                  <td className="px-3 py-2"><LabelChips labels={f.labels} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {pages > 1 && (
        <span className="flex items-center gap-2 text-xs text-muted">
          <Button size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
          <span className="num">
            {page * PAGE + 1} to {Math.min(rows.length, page * PAGE + PAGE)} of {num(rows.length)}
          </span>
          <Button size="sm" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</Button>
        </span>
      )}
    </Panel>
  );
}

/** The file picked: the original audio from the store, and its references. A segment plays from its start. */
function FilePanel({ id, ver, f, label, onClose }: { id: string; ver: Version; f: VersionFile; label: string | null; onClose: () => void }) {
  const audio = useRef<HTMLAudioElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const [t, setT] = useState(0);
  const refs = useQuery({
    queryKey: ["refs", id, ver.n, f.path],
    queryFn: () => api.get<FileRefs>(`/datasets/${id}/versions/${ver.n}/refs?path=${encodeURIComponent(f.path)}`),
    staleTime: Infinity,
    enabled: ver.label_types.length > 0,
  });
  useEffect(() => panel.current?.scrollIntoView({ block: "nearest" }), []);
  const play = (s: Segment) => {
    const a = audio.current;
    if (!a) return;
    a.currentTime = s.start;
    void a.play();
  };
  const segs = refs.data?.segments;
  const speakers = Object.entries(refs.data?.speakers ?? {});
  return (
    <div ref={panel}>
      <Panel
        title={<span className="break-all">{f.path}</span>}
        subtitle={<span className="num">{clock(f.duration)} · {num(f.sample_rate)} Hz · sha256 {f.sha256.slice(0, 12)}</span>}
        actions={
          <>
            {label && (
              <AnchorButton href={labelFileUrl(label, id, f.sha256, t || undefined)} newTab>
                <I.Pen /> Open in Label
              </AnchorButton>
            )}
            <Button variant="ghost" onClick={onClose} aria-label="Close the file">
              <I.X />
            </Button>
          </>
        }
      >
        <audio ref={audio} controls preload="metadata" src={audioUrl(f.sha256)} onTimeUpdate={(e) => setT(e.currentTarget.currentTime)} className="w-full">
          <track kind="captions" />
        </audio>
        {Object.keys(f.labels).length > 0 && (
          <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
            File labels <LabelChips labels={f.labels} />
          </span>
        )}
        {speakers.map(([sp, l]) => (
          <span key={sp} className="flex flex-wrap items-center gap-2 text-xs text-muted">
            Speaker <span className="num text-focus">{sp}</span> <LabelChips labels={l} />
          </span>
        ))}
        {ver.label_types.length === 0 && <span className="text-[13px] text-muted">Audio only: this version has no references yet. Label it in Concord Label.</span>}
        {refs.isLoading && <Spinner label="Reading references" />}
        <ErrorNote error={refs.error} />
        {refs.data && segs === null && (
          <span className="text-[13px] text-muted">
            The references are in a format the catalog doesn't read{refs.data.format ? ` (${refs.data.format})` : ""}. Concord Bench and Concord Label read them.
          </span>
        )}
        {segs && !segs.length && <span className="text-[13px] text-muted">No segments for this file.</span>}
        {!!segs?.length && (
          <ol className="m-0 p-0 list-none flex flex-col max-h-[420px] overflow-y-auto scroll-thin border border-track rounded-md" aria-label="Segments">
            {segs.map((s, i) => {
              const now = t >= s.start && t < s.end;
              return (
                <li key={i} className={cx("border-b border-track last:border-0", now && "bg-active")}>
                  <button type="button" onClick={() => play(s)} className="w-full bg-transparent border-0 text-left text-fg cursor-pointer px-3 py-2 flex flex-col gap-0.5 hover:bg-track/60" aria-current={now || undefined}>
                    <span className="flex flex-wrap items-center gap-2">
                      <Times s={s} />
                      {s.speaker && <span className="text-[11px] text-focus num">{s.speaker}</span>}
                    </span>
                    {s.text && <span className="text-[13px]">{s.text}</span>}
                    <LabelChips labels={s.labels} />
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </Panel>
    </div>
  );
}

function VersionFacts({ ver }: { ver: Version }) {
  const m = ver.manifest;
  const review = m.review ? Object.entries(m.review).filter(([, x]) => x).map(([k, x]) => `${x} ${k.replace("_", " ")}`).join(", ") : null;
  return (
    <Panel title={`Version ${ver.n}`}>
      <Fact k="Published by" v={appLabel(ver.published_by)} />
      <Fact k="Published" v={<span title={ver.published_at}>{ago(ver.published_at)}</span>} />
      <Fact k="Parent" v={parentLabel(ver)} />
      <Fact k="Label types" v={ver.label_types.length ? ver.label_types.map(labelType).join(", ") : "audio only"} />
      <Fact k="Files" v={num(ver.counts.files)} />
      {!!ver.label_types.length && <Fact k="Segments" v={num(ver.counts.segments ?? 0)} />}
      <Fact k="Audio" v={hours(ver.counts.hours ?? 0)} />
      <Fact k="Format version" v={ver.format_version} />
      {typeof m.scope === "string" && <Fact k="Scope" v={m.scope} />}
      {review && <Fact k="Review" v={review} />}
      {!!m.models?.length && <Fact k="Models" v={<span className="text-[11px]">{m.models.join(", ")}</span>} />}
      {typeof m.normaliser === "string" && <Fact k="Normaliser" v={m.normaliser} />}
      <Fact k="Digest" v={<span className="text-[11px]" title={ver.digest}>{ver.digest.slice(0, 16)}</span>} />
      <div className="flex flex-col gap-1 pt-1">
        <span className="text-xs text-muted">Made into</span>
        {ver.uses.length ? <UseList uses={ver.uses} /> : <span className="text-[13px] text-subtle">No app has made a dataset from this version yet.</span>}
      </div>
    </Panel>
  );
}

function LabelsPanel({ ver, files }: { ver: Version; files?: VersionFile[] }) {
  const stats = useMemo(() => labelStats(files ?? []), [files]);
  const speakers = useMemo(() => new Set((files ?? []).flatMap((f) => (f.speakers ?? []).map((s) => `${f.path}\u0000${s}`))).size, [files]);
  return (
    <Panel title="Labels" subtitle="Label names in the references, and the values of each file label">
      <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
        Names <LabelNames names={ver.labels} max={20} />
      </span>
      {!!speakers && <Fact k="Speakers (per file)" v={num(speakers)} />}
      {stats.map(([k, values]) => (
        <div key={k} className="flex flex-col gap-1 border-b border-track last:border-0 pb-2">
          <span className="text-[13px] num">{k}</span>
          <span className="flex flex-wrap gap-1">
            {values.slice(0, 12).map(([val, c]) => (
              <Chip key={val} title={`${c} files`}>
                {labelValue(val)} <span className="num text-subtle">{c}</span>
              </Chip>
            ))}
            {values.length > 12 && <span className="text-[11px] text-subtle">+{values.length - 12} more</span>}
          </span>
        </div>
      ))}
      {files && !stats.length && <span className="text-[13px] text-subtle">No file labels.</span>}
    </Panel>
  );
}
