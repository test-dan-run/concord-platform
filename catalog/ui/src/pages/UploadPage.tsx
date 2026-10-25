// Upload audio (DATASET_CATALOG_PLAN.md §6.1): files, a folder or zips of audio go to the store (lib/upload.ts:
// hashed in the browser, only missing files sent, in chunks that resume), then become a version with audio only:
// version 1 of a new dataset, or the next version of a dataset whose latest version has audio only. A dataset with
// references gets new audio through Concord Label instead, so a version never mixes labelled and unlabelled files.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import { allFiles, APP, errorDetail, plural, takesAudio, versionPath, type Dataset, type Version } from "../lib/datasets";
import { bytes, clock, hours, num } from "../lib/format";
import { AUDIO_EXTS, BUSY, pickedFromDrop, pickedFromInput, planVersion, uploads, type PickedFile, type UploadItem } from "../lib/upload";
import * as I from "../components/icons";
import { TagInput } from "../components/TagInput";
import { Bar, Button, cx, Empty, ErrorNote, inputClass, PageHeader, Panel, Select, Toggle } from "../components/ui";

const SHOW = 100;

const finished = (i: UploadItem) => ["done", "unreadable", "skipped"].includes(i.state);

/** A name for a new dataset from what was picked: the folder, or the zip without its extension. */
export function suggestName(items: { path: string }[]): string {
  const first = items[0]?.path ?? "";
  const top = first.includes("/") ? first.split("/")[0] : first.replace(/\.zip$/i, "");
  return items.length === 1 || first.includes("/") || /\.zip$/i.test(first) ? top.replace(/\.[^.]+$/, "") : "";
}

function Status({ it }: { it: UploadItem }) {
  const pct = (x: number) => `${Math.floor((100 * x) / Math.max(1, it.size))}%`;
  switch (it.state) {
    case "waiting":
      return <span className="text-muted">Waiting</span>;
    case "hashing":
      return <span className="text-muted">Hashing {pct(it.hashed)}</span>;
    case "checking":
      return <span className="text-muted">Asking the store</span>;
    case "uploading":
      return (
        <span className="text-muted">
          Sending {bytes(it.sent)} of {bytes(it.size)}
          {it.resumedAt ? `, resumed at ${bytes(it.resumedAt)}` : ""}
        </span>
      );
    case "retrying":
      return <span className="text-warn">{it.error}</span>;
    case "committing":
      return <span className="text-muted">Checking the bytes and reading the audio</span>;
    case "unpacking":
      return <span className="text-muted">Unpacking the zip</span>;
    case "done":
      if (it.zip) return <span className="text-good">{plural(it.entries?.length ?? 0, "audio file")} stored{it.skipped?.length ? `, ${it.skipped.length} left out` : ""}</span>;
      return (
        <span className={it.had ? "text-muted" : "text-good"}>
          {it.had ? "Already in the store" : it.resumedAt ? "Sent, resumed" : "Sent"}
          {it.record?.duration !== null && it.record?.duration !== undefined && <span className="num text-subtle"> · {clock(it.record.duration, false)} · {num(it.record.sample_rate)} Hz</span>}
        </span>
      );
    case "unreadable":
      return <span className="text-bad">ffprobe can't read it, so it is left out</span>;
    case "skipped":
      return <span className="text-muted">Left out: {it.error}</span>;
    case "failed":
      return <span className="text-bad">{it.error}</span>;
  }
}

function ItemRow({ it }: { it: UploadItem }) {
  const progress = it.state === "hashing" ? it.hashed / Math.max(1, it.size) : it.sent / Math.max(1, it.size);
  return (
    <li className="flex flex-col gap-1 py-2 border-b border-track last:border-0">
      <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
        <span className="num break-all flex-1 min-w-[160px]">{it.path}</span>
        <span className="num text-xs text-subtle">{bytes(it.size)}</span>
        {it.state === "failed" && <Button size="sm" onClick={() => uploads.retry(it.id)}><I.Refresh size={14} /> Retry</Button>}
        {!BUSY.includes(it.state) && it.state !== "waiting" && (
          <Button size="sm" variant="ghost" aria-label={`Remove ${it.path} from the list`} onClick={() => uploads.remove(it.id)}>
            <I.X size={14} />
          </Button>
        )}
      </span>
      <span className="text-xs">
        <Status it={it} />
      </span>
      {(BUSY.includes(it.state) || it.state === "retrying") && <Bar value={progress} color={it.state === "retrying" ? "var(--color-warn)" : undefined} />}
      {it.zip && !!it.skipped?.length && (
        <ul className="m-0 pl-5 text-xs text-muted">
          {it.skipped.slice(0, 20).map((s) => (
            <li key={s.path}>
              <span className="num">{s.path}</span>: {s.error}
            </li>
          ))}
          {it.skipped.length > 20 && <li>and {it.skipped.length - 20} more</li>}
        </ul>
      )}
    </li>
  );
}

function Picker({ onFiles }: { onFiles: (f: PickedFile[]) => void }) {
  const files = useRef<HTMLInputElement>(null);
  const folder = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const accept = [...AUDIO_EXTS, ".zip"].join(",");
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={async (e) => {
        e.preventDefault();
        setDrag(false);
        onFiles(await pickedFromDrop(e));
      }}
      className={cx("border border-dashed rounded-md p-6 flex flex-col items-center gap-3 text-center", drag ? "border-primary bg-primary/5" : "border-line")}
    >
      <span className="text-muted"><I.Upload size={24} /></span>
      <span className="text-[13px]">Drop audio files, folders or zips of audio here</span>
      <span className="text-xs text-muted">{AUDIO_EXTS.join(" ")} and .zip. Other files are left out.</span>
      <span className="flex flex-wrap justify-center gap-2">
        <Button onClick={() => files.current?.click()}>Choose files</Button>
        <Button onClick={() => folder.current?.click()}>Choose a folder</Button>
      </span>
      <input ref={files} type="file" multiple accept={accept} className="hidden" aria-label="Choose files" onChange={(e) => (onFiles(pickedFromInput(e.target.files)), (e.target.value = ""))} />
      <input ref={folder} type="file" className="hidden" aria-label="Choose a folder" {...{ webkitdirectory: "" }} onChange={(e) => (onFiles(pickedFromInput(e.target.files)), (e.target.value = ""))} />
    </div>
  );
}

export function UploadPage() {
  const items = useSyncExternalStore(uploads.subscribe, uploads.getSnapshot);
  const [params] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const [refused, setRefused] = useState<string[]>([]);
  const [all, setAll] = useState(false);
  const [mode, setMode] = useState<"new" | "add">(params.get("dataset") ? "add" : "new");
  const [target, setTarget] = useState(params.get("dataset") ?? "");
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const named = useRef(false);
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<Dataset[]>("/datasets") });
  const open = (datasets.data ?? []).filter(takesAudio);
  const chosen = (datasets.data ?? []).find((d) => d.id === target) ?? null;
  const latest = mode === "add" && chosen ? chosen.latest : null;
  const existing = useQuery({ queryKey: ["version-files", target, latest?.n], queryFn: () => allFiles(target, latest!.n), enabled: !!latest, staleTime: Infinity });
  const busy = items.some((i) => i.state === "waiting" || BUSY.includes(i.state));
  const plan = planVersion(items, latest ? (existing.data ?? []) : []);
  const failed = items.filter((i) => i.state === "failed").length;
  const total = items.reduce((a, i) => a + i.size, 0);
  const through = items.reduce((a, i) => a + (finished(i) ? i.size : i.sent), 0);

  useEffect(() => {
    if (!busy) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [busy]);
  useEffect(() => {
    if (!busy && items.length) void qc.invalidateQueries({ queryKey: ["store-stats"] }); // the store has new audio
  }, [busy]); // only when the uploads finish
  useEffect(() => {
    if (named.current || !items.length) return;
    named.current = true;
    setName((n) => n || suggestName(items));
  }, [items]);

  const make = useMutation({
    mutationFn: async (force: boolean) => {
      const files = plan.files.map((f) => ({ path: f.path, sha256: f.sha256 }));
      if (mode === "add") return api.post<Version>(`/datasets/${target}/versions/audio`, { files, parent: latest?.n ?? null, app: APP, force });
      const d = await api.post<Dataset>("/datasets", { name: name.trim(), language: language.trim() || undefined, description: description.trim() || undefined, tags, app: APP });
      try {
        return await api.post<Version>(`/datasets/${d.id}/versions/audio`, { files, parent: null, app: APP });
      } catch (e) {
        await api.del(`/datasets/${d.id}`).catch(() => undefined); // don't leave an empty dataset behind
        throw e;
      }
    },
    onSuccess: (v) => {
      uploads.clear();
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      void qc.invalidateQueries({ queryKey: ["dataset", v.dataset_id] });
      void qc.invalidateQueries({ queryKey: ["store-stats"] });
      nav(versionPath(v.dataset_id, v.n));
    },
  });
  const stale = make.error instanceof ApiError && make.error.body.type === "stale_parent";
  const why =
    busy ? "Wait for the uploads to finish." :
    !plan.files.length ? (plan.already ? "Every file is in the dataset already." : "Add some audio first.") :
    mode === "new" && !name.trim() ? "Give the dataset a name." :
    mode === "add" && !chosen ? "Pick a dataset." :
    mode === "add" && chosen && !takesAudio(chosen) ? "This dataset's latest version has references." :
    mode === "add" && latest && !existing.data ? "Reading the dataset's files." : "";
  const add = (files: PickedFile[]) => {
    const out = uploads.add(files);
    setRefused(out.map((f) => f.path));
    make.reset();
  };

  const order = (i: UploadItem) => (["failed", "retrying", "unreadable"].includes(i.state) ? 0 : BUSY.includes(i.state) ? 1 : i.state === "waiting" ? 2 : 3);
  const sorted = [...items].sort((x, y) => order(x) - order(y) || x.id - y.id);
  const shown = all ? sorted : sorted.slice(0, SHOW);
  const count = (pred: (i: UploadItem) => boolean) => items.filter(pred).length;
  const length = plan.files.reduce((a, f) => a + f.duration, 0) / 3600;

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/datasets">Datasets</Link> / Upload audio</>}
        title="Upload audio"
        subtitle="Audio without references, for labelling in Concord Label. Each file is hashed in your browser first, so files the store already has aren't sent again, and a dropped connection resumes where it stopped."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px] items-start">
        <Panel
          title="Files"
          subtitle={
            items.length
              ? `${plural(items.length, "item")}: ${count((i) => i.state === "done" && !i.had)} sent, ${count((i) => !!i.had && i.state === "done")} already in the store${count((i) => i.state === "unreadable") ? `, ${count((i) => i.state === "unreadable")} unreadable` : ""}${failed ? `, ${failed} failed` : ""}`
              : undefined
          }
          actions={
            items.length ? (
              <Button size="sm" variant="ghost" onClick={() => (uploads.clear(), setRefused([]), make.reset())}>
                {busy ? "Stop and clear" : "Clear list"}
              </Button>
            ) : undefined
          }
        >
          <Picker onFiles={add} />
          {!!refused.length && (
            <span className="text-xs text-muted">
              Left out {plural(refused.length, "file")} that {refused.length === 1 ? "isn't" : "aren't"} audio or a zip: <span className="num">{refused.slice(0, 5).join(", ")}</span>
              {refused.length > 5 && ` and ${refused.length - 5} more`}
            </span>
          )}
          {!!items.length && (
            <span className="flex flex-col gap-1">
              <Bar value={through / Math.max(1, total)} />
              <span className="text-xs text-muted num">
                {bytes(through)} of {bytes(total)} done
              </span>
            </span>
          )}
          {!items.length && <Empty>No files yet. Bytes already sent stay in the store, so picking the same files again resumes them.</Empty>}
          {!!items.length && (
            <ul className="m-0 p-0 list-none flex flex-col" aria-label="Uploads">
              {shown.map((it) => (
                <ItemRow key={it.id} it={it} />
              ))}
            </ul>
          )}
          {sorted.length > SHOW && !all && (
            <span>
              <Button size="sm" onClick={() => setAll(true)}>Show all {num(sorted.length)}</Button>
            </span>
          )}
        </Panel>
        <Panel title="Make a version" subtitle="A version with audio only, which Concord Label can open to start labelling">
          <Toggle
            label="Where the files go"
            value={mode}
            onChange={(v) => (setMode(v as "new" | "add"), make.reset())}
            options={[
              { value: "new", label: "New dataset" },
              { value: "add", label: "Add to a dataset" },
            ]}
            className="self-start"
          />
          {mode === "new" ? (
            <>
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Name
                <input value={name} onChange={(e) => setName(e.target.value)} className={inputClass} />
              </label>
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Language
                <input value={language} onChange={(e) => setLanguage(e.target.value)} placeholder="e.g. en, ms, zh" className={inputClass} />
              </label>
              <label className="flex flex-col gap-1.5 text-xs text-muted">
                Description
                <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} className={`${inputClass} py-2 resize-y`} />
              </label>
              <TagInput value={tags} onChange={setTags} pool={(datasets.data ?? []).flatMap((d) => d.tags)} placeholder="e.g. call-centre, en-SG" />
            </>
          ) : (
            <>
              <Select label="Dataset" value={target} onChange={(e) => (setTarget(e.target.value), make.reset())} className="!min-h-10 flex-1 min-w-0">
                <option value="">Pick one</option>
                {chosen && !takesAudio(chosen) && <option value={chosen.id}>{chosen.name}</option>}
                {open.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} {d.latest ? `(v${d.latest.n}, ${plural(d.latest.counts.files, "file")})` : "(no versions)"}
                  </option>
                ))}
              </Select>
              <span className="text-xs text-muted">
                Only datasets whose latest version has audio only are listed. A dataset with references gets new audio through Concord Label, where the new files are
                labelled and published.
              </span>
              {chosen && !takesAudio(chosen) && <ErrorNote error={`${chosen.name} v${chosen.latest?.n} has references, so audio can't be added here.`} />}
            </>
          )}
          {!!plan.files.length && (
            <span className="text-[13px]">
              {mode === "new" || !latest
                ? `Version 1 with ${plural(plan.files.length, "file")} (${hours(length)}).`
                : `The next version: the ${plural(latest.counts.files, "file")} of v${latest.n} and ${plural(plan.files.length, "new file")} (${hours(length)}).`}
            </span>
          )}
          {!!plan.leftOut.length && (
            <details className="text-xs">
              <summary className="cursor-pointer text-warn">{plural(plan.leftOut.length, "file")} left out</summary>
              <ul className="m-0 mt-1 pl-5 flex flex-col gap-0.5 text-muted">
                {plan.leftOut.slice(0, 50).map((l) => (
                  <li key={l.path}>
                    <span className="num break-all">{l.path}</span>: {l.reason}
                  </li>
                ))}
                {plan.leftOut.length > 50 && <li>and {plan.leftOut.length - 50} more</li>}
              </ul>
            </details>
          )}
          {!!plan.already && <span className="text-xs text-muted">{plural(plan.already, "file")} already in the dataset.</span>}
          <ErrorNote error={make.error} />
          {errorDetail(make.error).length > 0 && <span className="text-xs text-muted num break-all">{errorDetail(make.error).slice(0, 10).join(", ")}</span>}
          <span className="flex flex-wrap items-center gap-2">
            <Button variant="primary" disabled={!!why || make.isPending} onClick={() => make.mutate(false)}>
              {mode === "new" ? "Make dataset" : "Add files"}
            </Button>
            {stale && (
              <Button onClick={() => make.mutate(true)} disabled={make.isPending} title="Someone published a version since this page read the dataset">
                Add anyway
              </Button>
            )}
            {why && <span className="text-xs text-muted">{why}</span>}
          </span>
        </Panel>
      </div>
    </>
  );
}
