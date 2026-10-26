// One dataset (catalog/README.md "Pages"): its version history with a short summary of each version's changes
// from its parent, and which app datasets were made from each version. A person edits the name, description and
// tags here, downloads a version as a reference package, and deletes a version or the whole dataset (§5.3).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, notFound } from "../lib/api";
import { appLabel, packageUrl, parentLabel, plural, takesAudio, versionPath, type Comparison, type Dataset, type Version } from "../lib/datasets";
import { ago, hours, num } from "../lib/format";
import * as I from "../components/icons";
import { ConfirmDelete, UseList } from "../components/dataset";
import { TagInput } from "../components/TagInput";
import { Button, Chip, Empty, ErrorNote, Fact, inputClass, LinkButton, PageHeader, Panel, Spinner } from "../components/ui";
import { LabelNames, LabelTypeChips } from "./DatasetsPage";

const small = "inline-flex items-center gap-1.5 min-h-8 px-2.5 rounded-md text-xs no-underline border border-line !text-fg hover:bg-track";

export function compareSummary(c: Comparison): string {
  const parts: string[] = [];
  if (c.files.added.length) parts.push(`${plural(c.files.added.length, "file")} added`);
  if (c.files.removed.length) parts.push(`${plural(c.files.removed.length, "file")} removed`);
  if (c.changed.length) parts.push(`${plural(c.changed.length, "file")} with changed references`);
  if (c.labels.added.length) parts.push(`${plural(c.labels.added.length, "label")} added`);
  if (c.labels.removed.length) parts.push(`${plural(c.labels.removed.length, "label")} removed`);
  return parts.length ? parts.join(", ") : "no changes";
}

function ChangeSummary({ id, v }: { id: string; v: Version }) {
  const parent = v.parent_n;
  const c = useQuery({
    queryKey: ["compare", id, parent, v.n],
    queryFn: () => api.get<Comparison>(`/datasets/${id}/compare?a=${parent}&b=${v.n}`),
    enabled: parent !== null && !v.parent_deleted,
    staleTime: Infinity,
  });
  if (parent === null) return <span className="text-xs text-muted">First version</span>;
  if (v.parent_deleted) return <span className="text-xs text-muted">Based on v{parent}, which was deleted</span>;
  return (
    <span className="text-xs text-muted">
      Changes from <Link to={`/datasets/${id}/compare?a=${parent}&b=${v.n}`}>v{parent}</Link>: {c.data ? compareSummary(c.data) : c.error ? "not available" : "..."}
    </span>
  );
}

function VersionRow({ d, v, latest }: { d: Dataset; v: Version; latest: boolean }) {
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState(false);
  const del = useMutation({
    mutationFn: () => api.del(`/datasets/${d.id}/versions/${v.n}`),
    onSuccess: () => {
      setConfirm(false);
      void qc.invalidateQueries({ queryKey: ["dataset", d.id] });
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      void qc.invalidateQueries({ queryKey: ["store-stats"] });
    },
  });
  return (
    <div className="border border-track rounded-md p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <Link to={versionPath(d.id, v.n)} className="text-[15px] font-semibold num !text-fg hover:underline">v{v.n}</Link>
        {latest && <Chip tone="good">latest</Chip>}
        <LabelTypeChips types={v.label_types} />
        <span className="text-xs text-muted">
          {appLabel(v.published_by)}, {ago(v.published_at)}
        </span>
        <span className="flex flex-wrap gap-1.5 sm:ml-auto">
          <Link to={versionPath(d.id, v.n)} className={small}><I.Eye size={14} /> Open</Link>
          <a href={packageUrl(d.id, v.n, false)} download className={small} title="manifest.json, files.csv and the references, without audio">
            <I.Download size={14} /> Package
          </a>
          <a href={packageUrl(d.id, v.n, true)} download className={small} title="The reference package with every audio file under audio/">
            <I.Download size={14} /> With audio
          </a>
          <Button size="sm" variant="danger" onClick={() => setConfirm(true)} aria-expanded={confirm}>Delete</Button>
        </span>
      </div>
      <span className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted num">
        <span>Parent {parentLabel(v)}</span>
        <span>{plural(v.counts.files, "file")}</span>
        {!!v.label_types.length && <span>{plural(v.counts.segments ?? 0, "segment")}</span>}
        <span>{hours(v.counts.hours ?? 0)}</span>
        <span>format {v.format_version}</span>
      </span>
      <ChangeSummary id={d.id} v={v} />
      {!!v.labels.length && (
        <span className="flex flex-wrap items-center gap-2 text-xs text-muted">
          Labels <LabelNames names={v.labels} max={8} />
        </span>
      )}
      {!!v.uses.length && (
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted">Made into</span>
          <UseList uses={v.uses} />
        </div>
      )}
      {confirm && (
        <ConfirmDelete
          title={`Delete v${v.n}?`}
          uses={v.uses}
          action={`Delete v${v.n}`}
          pending={del.isPending}
          error={del.error}
          onConfirm={() => del.mutate()}
          onCancel={() => setConfirm(false)}
        >
          Its file list and references are removed for good, and the number v{v.n} isn't used again. The audio stays in the store until clean-up, and only files
          that no version lists and no app holds are cleaned up.
        </ConfirmDelete>
      )}
    </div>
  );
}

function Details({ d }: { d: Dataset }) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const [name, setName] = useState(d.name);
  const [description, setDescription] = useState(d.description ?? "");
  const [tags, setTags] = useState(d.tags);
  const [confirm, setConfirm] = useState(false);
  const all = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<Dataset[]>("/datasets") });
  useEffect(() => {
    setName(d.name);
    setDescription(d.description ?? "");
    setTags(d.tags);
  }, [d.updated_at]); // reset when the dataset changes, not on every refetch
  const save = useMutation({
    mutationFn: () => api.patch<Dataset>(`/datasets/${d.id}`, { name: name.trim(), description: description.trim(), tags }),
    onSuccess: (out) => {
      qc.setQueryData(["dataset", d.id], out);
      void qc.invalidateQueries({ queryKey: ["datasets"] });
    },
  });
  const del = useMutation({
    mutationFn: () => api.del(`/datasets/${d.id}`),
    onSuccess: () => {
      qc.removeQueries({ queryKey: ["dataset", d.id] });
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      void qc.invalidateQueries({ queryKey: ["store-stats"] });
      nav("/datasets");
    },
  });
  const changed = name.trim() !== d.name || description.trim() !== (d.description ?? "") || tags.join() !== d.tags.join();
  const uses = (d.versions ?? []).flatMap((v) => v.uses.map((u) => ({ ...u, n: v.n })));
  return (
    <>
      <Panel title="Details">
        <label className="flex flex-col gap-1.5 text-xs text-muted">
          Name
          <input value={name} onChange={(e) => setName(e.target.value)} className={inputClass} />
        </label>
        <label className="flex flex-col gap-1.5 text-xs text-muted">
          Description
          <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} className={`${inputClass} py-2 resize-y`} />
        </label>
        <TagInput value={tags} onChange={setTags} pool={(all.data ?? []).flatMap((x) => x.tags)} placeholder="e.g. call-centre, en-SG" />
        <ErrorNote error={save.error} />
        <span className="flex gap-2">
          <Button variant="primary" disabled={!changed || !name.trim() || save.isPending} onClick={() => save.mutate()}>Save</Button>
          {changed && <Button variant="ghost" onClick={() => (setName(d.name), setDescription(d.description ?? ""), setTags(d.tags))}>Undo</Button>}
        </span>
        <div className="pt-1">
          <Fact k="Language" v={d.language ?? "not set"} />
          <Fact k="Made by" v={appLabel(d.created_by)} />
          <Fact k="Made" v={ago(d.created_at)} />
          <Fact k="Changed" v={ago(d.updated_at)} />
          <Fact k="Dataset id" v={<span className="text-[11px] break-all">{d.id}</span>} />
        </div>
      </Panel>
      <Panel title="Delete dataset" subtitle="Deletes every version. The apps' datasets made from it keep working.">
        {confirm ? (
          <ConfirmDelete
            title={`Delete ${d.name} and its ${plural(d.versions?.length ?? 0, "version")}?`}
            uses={uses}
            action="Delete dataset"
            pending={del.isPending}
            error={del.error}
            onConfirm={() => del.mutate()}
            onCancel={() => setConfirm(false)}
          >
            The dataset, its versions, their file lists and references are removed for good. The audio stays in the store until clean-up.
          </ConfirmDelete>
        ) : (
          <span>
            <Button variant="danger" onClick={() => setConfirm(true)}>Delete dataset</Button>
          </span>
        )}
      </Panel>
    </>
  );
}

export function DatasetDetail() {
  const { id = "" } = useParams();
  const q = useQuery({ queryKey: ["dataset", id], queryFn: () => api.get<Dataset>(`/datasets/${id}`), retry: false });
  if (q.isLoading) return <Spinner />;
  if (!q.data || notFound(q.error))
    return (
      <>
        <PageHeader crumbs={<Link to="/datasets">Datasets</Link>} title="Dataset not found" />
        <ErrorNote error={q.error ?? "Unknown dataset"} />
      </>
    );
  const d = q.data;
  const versions = d.versions ?? [];
  return (
    <>
      <PageHeader
        crumbs={<><Link to="/datasets">Datasets</Link> / {d.name}</>}
        title={d.name}
        subtitle={d.description ?? undefined}
        actions={
          <>
            {versions.length >= 2 && <LinkButton to={`/datasets/${id}/compare?a=${versions[1].n}&b=${versions[0].n}`}>Compare</LinkButton>}
            <LinkButton to={`/datasets/import?dataset=${id}`}><I.Download /> Import a version</LinkButton>
            {takesAudio(d) && <LinkButton to={`/datasets/upload?dataset=${id}`} variant="primary"><I.Upload /> Add audio</LinkButton>}
          </>
        }
      />
      {!takesAudio(d) && (
        <span className="text-xs text-muted -mt-2">
          The latest version has references, so new audio comes in through Concord Label, where it is labelled and published as the next version.
        </span>
      )}
      <div className="grid gap-5 xl:grid-cols-[1fr_340px] items-start">
        <Panel title="Versions" subtitle={versions.length ? `${plural(versions.length, "version")}, newest first` : undefined}>
          {!versions.length && (
            <Empty>
              No versions yet. <Link to={`/datasets/upload?dataset=${id}`}>Add audio</Link> or <Link to={`/datasets/import?dataset=${id}`}>import a reference package</Link>.
            </Empty>
          )}
          {versions.map((v, i) => (
            <VersionRow key={v.n} d={d} v={v} latest={i === 0} />
          ))}
          {!!versions.length && <span className="text-xs text-subtle">{num(versions.reduce((a, v) => a + v.uses.length, 0))} app datasets made from these versions.</span>}
        </Panel>
        <div className="flex flex-col gap-5">
          <Details key={d.id} d={d} />
        </div>
      </div>
    </>
  );
}
