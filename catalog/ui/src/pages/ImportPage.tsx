// Import package (DATASET_CATALOG_PLAN.md §6): a reference package zip, from a person or another tool, becomes a
// new dataset or the next version of an existing one. The catalog checks the zip first and shows a preview
// (POST /datasets/import with preview, which stores nothing); the import sends it again and stores its audio/ and
// its references. Audio the package leaves out must already be in the store.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import { appLabel, APP, errorDetail, labelType, plural, postForm, versionPath, type Dataset, type ImportPreview, type Version } from "../lib/datasets";
import { bytes, hours, num } from "../lib/format";
import * as I from "../components/icons";
import { Bar, Button, Chip, cx, ErrorNote, Fact, PageHeader, Panel, Select, Spinner, Toggle } from "../components/ui";

type Checked = { preview: ImportPreview; target: string };

export function ImportPage() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<"new" | "add">(params.get("dataset") ? "add" : "new");
  const [target, setTarget] = useState(params.get("dataset") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<Dataset[]>("/datasets") });
  const chosen = (datasets.data ?? []).find((d) => d.id === target) ?? null;
  const dest = mode === "add" ? target : "";

  const form = (f: File, preview: boolean) => {
    const fd = new FormData();
    fd.append("file", f, f.name);
    if (dest) fd.append("dataset_id", dest);
    fd.append("app", APP);
    fd.append("preview", String(preview));
    return fd;
  };
  const check = useMutation({
    mutationFn: async (f: File): Promise<Checked> => {
      setProgress(0);
      try {
        return { preview: await postForm<ImportPreview>("/datasets/import", form(f, true), setProgress), target: dest };
      } finally {
        setProgress(null);
      }
    },
  });
  const run = useMutation({
    mutationFn: async (f: File) => {
      setProgress(0);
      try {
        return await postForm<Version>("/datasets/import", form(f, false), setProgress);
      } finally {
        setProgress(null);
      }
    },
    onSuccess: (v) => {
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      void qc.invalidateQueries({ queryKey: ["dataset", v.dataset_id] });
      void qc.invalidateQueries({ queryKey: ["store-stats"] });
      nav(versionPath(v.dataset_id, v.n));
    },
  });
  const busy = check.isPending || run.isPending;
  // While a zip is checked or imported, another one dropped on the page is ignored, and changing the target
  // doesn't clear the running import: either would hide it and let a second import start beside it.
  const pick = (f: File | undefined) => {
    if (!f || busy) return;
    setFile(f);
    run.reset();
    check.mutate(f);
  };
  const retarget = () => {
    if (!run.isPending) run.reset();
  };
  const p = check.data?.preview;
  const m = p?.manifest;
  const sameTarget = check.data?.target === dest;
  const why = !p ? "Pick a zip first." : !p.ok ? "Fix the problems in the package." : p.missing.length ? "Audio is missing." : mode === "add" && !chosen ? "Pick a dataset." : "";

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/datasets">Datasets</Link> / Import package</>}
        title="Import package"
        subtitle="A reference package zip from a person or another tool: manifest.json, files.csv, the references, and audio/ unless the store has the audio already."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px] items-start">
        <div className="flex flex-col gap-5 min-w-0">
          <Panel title="Package">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                pick(e.dataTransfer.files[0]);
              }}
              className={cx("border border-dashed rounded-md p-6 flex flex-col items-center gap-3 text-center", drag ? "border-primary bg-primary/5" : "border-line")}
            >
              <span className="text-muted"><I.Download size={24} /></span>
              <span className="text-[13px]">{file ? <span className="num break-all">{file.name}</span> : "Drop a reference package zip here"}</span>
              {file && <span className="text-xs text-muted num">{bytes(file.size)}</span>}
              <Button disabled={busy} onClick={() => input.current?.click()}>{file ? "Choose another zip" : "Choose a zip"}</Button>
              <input ref={input} type="file" accept=".zip" className="hidden" aria-label="Choose a zip" onChange={(e) => (pick(e.target.files?.[0]), (e.target.value = ""))} />
            </div>
            {progress !== null && (
              <span className="flex flex-col gap-1">
                <Bar value={progress} />
                <span className="text-xs text-muted">
                  {progress < 1 ? `Sending ${Math.floor(progress * 100)}%` : check.isPending ? "Checking the package" : "Storing the audio and the references"}
                </span>
              </span>
            )}
            <ErrorNote error={check.error} />
          </Panel>
          {p && <Preview p={p} sameTarget={sameTarget} chosen={chosen} />}
        </div>
        <Panel title="Import" subtitle="Each import makes a version. Versions never change.">
          <Toggle
            label="Import as"
            value={mode}
            onChange={(v) => (setMode(v as "new" | "add"), retarget())}
            options={[
              { value: "new", label: "New dataset" },
              { value: "add", label: "Next version" },
            ]}
            className="self-start"
          />
          {mode === "add" ? (
            <>
              <Select label="Dataset" value={target} onChange={(e) => (setTarget(e.target.value), retarget())} className="!min-h-10 flex-1 min-w-0">
                <option value="">Pick one</option>
                {(datasets.data ?? []).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} {d.latest ? `(v${d.latest.n})` : "(no versions)"}
                  </option>
                ))}
              </Select>
              {chosen && <span className="text-xs text-muted">{chosen.latest ? `The package becomes the next version of ${chosen.name}, based on v${chosen.latest.n}.` : `The package becomes version 1 of ${chosen.name}.`}</span>}
            </>
          ) : (
            <span className="text-xs text-muted">The dataset takes its name, language, description and tags from the package's manifest. You can change them afterwards.</span>
          )}
          <ErrorNote error={run.error} />
          {errorDetail(run.error).length > 0 && <span className="text-xs text-muted num break-all">{errorDetail(run.error).slice(0, 10).join(", ")}</span>}
          <span className="flex flex-wrap items-center gap-2">
            <Button variant="primary" disabled={!!why || busy || !file} onClick={() => file && run.mutate(file)}>
              {mode === "new" ? "Import as a new dataset" : "Import as the next version"}
            </Button>
            {run.isPending && <Spinner label="Importing" />}
            {why && <span className="text-xs text-muted">{why}</span>}
          </span>
          {m && (
            <span className="text-xs text-subtle">
              {mode === "new" ? `Makes the dataset "${m.name ?? file?.name}"` : chosen ? `Adds to ${chosen.name}` : ""}
            </span>
          )}
        </Panel>
      </div>
    </>
  );
}

function Preview({ p, sameTarget, chosen }: { p: ImportPreview; sameTarget: boolean; chosen: Dataset | null }) {
  const m = p.manifest;
  if (!p.ok)
    return (
      <Panel title="Preview" actions={<Chip tone="bad">not valid</Chip>} subtitle="The catalog can't import this package">
        <ul className="m-0 pl-5 flex flex-col gap-1 text-[13px]">
          {p.problems.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </Panel>
    );
  return (
    <Panel title="Preview" actions={<Chip tone={p.missing.length ? "warn" : "good"}>{p.missing.length ? "audio missing" : "ready to import"}</Chip>}>
      {m && (
        <div>
          <Fact k="Name" v={m.name ?? "not set"} />
          <Fact k="Language" v={m.language ?? "not set"} />
          <Fact k="Label types" v={m.label_types?.length ? m.label_types.map(labelType).join(", ") : "audio only"} />
          <Fact k="Files" v={num(p.files)} />
          {!!m.label_types?.length && <Fact k="Segments" v={num(m.counts?.segments ?? 0)} />}
          <Fact k="Audio" v={hours(m.counts?.hours ?? 0)} />
          <Fact k="Audio in the zip" v={m.audio_included ?? "not stated"} />
          <Fact k="Format version" v={m.format_version ?? "not stated"} />
          {m.source?.app && <Fact k="From" v={`${appLabel(m.source.app)}${m.source.dataset_name ? `, ${m.source.dataset_name}` : ""}${m.export_version ? ` v${m.export_version}` : ""}`} />}
          {typeof m.scope === "string" && <Fact k="Scope" v={m.scope} />}
          {!!m.tags?.length && <Fact k="Tags" v={m.tags.map((t) => `#${t}`).join(" ")} />}
        </div>
      )}
      {p.missing.length > 0 && (
        <div role="alert" className="border border-warn/50 bg-warn/10 rounded-md px-3 py-2 text-[13px] flex flex-col gap-1">
          <span>
            {plural(p.missing.length, "audio file")} {p.missing.length === 1 ? "is" : "are"} neither in the package nor in the store. Add them to the zip under audio/, or
            upload them on <Link to="/datasets/upload">Upload audio</Link> first.
          </span>
          <span className="num text-[11px] text-muted break-all">{p.missing.slice(0, 5).join(", ")}{p.missing.length > 5 ? ` and ${p.missing.length - 5} more` : ""}</span>
        </div>
      )}
      {sameTarget && p.same_as_latest && chosen?.latest && (
        <span role="status" className="text-[13px] text-muted">
          This package is the same as {chosen.name} v{chosen.latest.n}, so importing it returns that version and makes no new one.
        </span>
      )}
    </Panel>
  );
}
