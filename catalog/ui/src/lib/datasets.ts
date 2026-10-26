// The Dataset Catalog's response shapes (catalog/datasets, catalog/README.md "API") and small helpers the pages
// share. A dataset's versions never change, so their files, references and comparisons are cached for good; only
// the dataset and version records are refreshed by the SSE `dataset` event (lib/events.ts).

import { api, ApiError, errorOf } from "./api";

/** The app name the catalog records for what this UI makes. */
export const APP = "dataset-catalog";

export type Use = { app: string; ref: string; created_at: string };

export type Version = {
  dataset_id: string;
  n: number;
  parent_n: number | null;
  parent_deleted: boolean;
  digest: string;
  format_version: number;
  label_types: string[];
  labels: string[];
  counts: { files: number; segments?: number; hours?: number };
  published_by: string | null;
  published_at: string;
  uses: Use[];
  manifest: Record<string, unknown> & { source?: Record<string, string>; scope?: string; review?: Record<string, number>; models?: string[]; normaliser?: string };
};

export type Dataset = {
  id: string;
  name: string;
  language: string | null;
  description: string | null;
  tags: string[];
  created_by: string | null;
  created_at: string;
  updated_at: string;
  latest: Version | null;
  versions?: Version[];
};

export type VersionFile = {
  path: string;
  sha256: string;
  duration: number;
  sample_rate: number;
  segments: number | null;
  speakers: string[] | null;
  labels: Record<string, unknown>;
};

export type Segment = { start: number; end: number; speaker: string | null; text: string | null; uid: string | null; labels: Record<string, unknown> };

export type FileRefs = {
  path: string;
  format: string | null;
  segments: Segment[] | null;
  labels: Record<string, unknown>;
  speakers: Record<string, Record<string, unknown>>;
};

export type DiffOp = { op: "equal" | "delete" | "insert" | "replace"; a: string; b: string };

export type Comparison = {
  a: number;
  b: number;
  files: { added: string[]; removed: string[]; same: number };
  changed: {
    path: string;
    audio_changed?: boolean;
    segments?: { added: Segment[]; removed: Segment[]; edited: { a: Segment; b: Segment; diff: DiffOp[] }[] };
    labels?: { a: { files: Record<string, unknown>; speakers: Record<string, unknown> }; b: { files: Record<string, unknown>; speakers: Record<string, unknown> } };
  }[];
  segments_compared: boolean;
  labels: { added: string[]; removed: string[] };
};

export type AudioRecord = {
  sha256: string;
  size: number;
  ext: string;
  duration: number | null;
  sample_rate: number | null;
  channels: number | null;
  state: "partial" | "ready";
  created_at: string;
};

export type Group = { files: number; bytes: number };

export type StoreStats = {
  root: string;
  audio: Group;
  in_versions: Group;
  held_only: Group;
  neither: Group;
  partial: Group;
  holds: { app: string; files: number; bytes: number }[];
  datasets: number;
  versions: number;
  disk: { total: number; free: number };
};

export type GcResult = { dry_run: boolean; files: number; bytes: number; sample: Pick<AudioRecord, "sha256" | "size" | "ext" | "state" | "created_at">[] };

export type ImportPreview = {
  ok: boolean;
  problems: string[];
  manifest: (Version["manifest"] & { name?: string; language?: string; label_types?: string[]; counts?: Version["counts"]; format_version?: number; audio_included?: string; description?: string; tags?: string[] }) | null;
  files: number;
  missing: string[];
  same_as_latest: boolean;
};

export function labelType(t: string): string {
  return { asr: "ASR", diar: "Diarisation" }[t] ?? t;
}

/** A version with audio only (format version 3): no references, so Concord Label can start labelling it. */
export function audioOnly(v: Version | null | undefined): boolean {
  return !!v && v.label_types.length === 0;
}

/** Whether Upload audio may add files: a dataset with no versions yet, or whose latest version has audio only. */
export function takesAudio(d: Dataset): boolean {
  return !d.latest || audioOnly(d.latest);
}

export function parentLabel(v: Version): string {
  if (v.parent_n === null) return "none";
  return v.parent_deleted ? `v${v.parent_n} (deleted)` : `v${v.parent_n}`;
}

export function appLabel(app: string | null | undefined): string {
  if (!app) return "unknown";
  return { label: "Concord Label", "concord-label": "Concord Label", bench: "Concord Bench", "concord-bench": "Concord Bench", [APP]: "Dataset Catalog" }[app] ?? app;
}

/** "3 files", "1 file". */
export function plural(n: number, word: string, many = `${word}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? word : many}`;
}

export function labelValue(v: unknown): string {
  if (v === null || v === undefined) return "none";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

export const datasetPath = (id: string) => `/datasets/${id}`;
export const versionPath = (id: string, n: number) => `/datasets/${id}/v/${n}`;
export const packageUrl = (id: string, n: number, audio: boolean) => `/api/datasets/${id}/versions/${n}/package${audio ? "?audio=true" : ""}`;
export const audioUrl = (sha: string) => `/api/store/audio/${sha}`;

/** A multipart POST with upload progress (0 to 1), which fetch can't report. For a reference package zip, which
 *  goes in one request. */
export function postForm<T>(path: string, fd: FormData, onProgress?: (f: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", `/api${path}`);
    x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    x.onload = () => {
      const r = new Response(x.responseText || null, { status: x.status, statusText: x.statusText, headers: { "Content-Type": "application/json" } });
      if (r.ok) resolve(JSON.parse(x.responseText) as T);
      else void errorOf(r).then(reject);
    };
    x.onerror = () => reject(new TypeError("The connection dropped before the catalog answered. Try again."));
    x.send(fd);
  });
}

const PAGE = 1000;

/** Every file of a version, a page of 1000 at a time. */
export async function allFiles(id: string, n: number): Promise<VersionFile[]> {
  const out: VersionFile[] = [];
  for (let offset = 0; ; offset += PAGE) {
    const r = await api.get<{ total: number; files: VersionFile[] }>(`/datasets/${id}/versions/${n}/files?offset=${offset}&limit=${PAGE}`);
    out.push(...r.files);
    if (out.length >= r.total || !r.files.length) return out;
  }
}

/** Value counts per file label, across a version's files: {"condition": [["clean", 12], ["noisy", 3]]}. */
export function labelStats(files: VersionFile[]): [string, [string, number][]][] {
  const by = new Map<string, Map<string, number>>();
  for (const f of files) {
    for (const [k, v] of Object.entries(f.labels ?? {})) {
      const m = by.get(k) ?? new Map<string, number>();
      const key = labelValue(v);
      m.set(key, (m.get(key) ?? 0) + 1);
      by.set(k, m);
    }
  }
  return [...by]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, m]) => [k, [...m].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]))]);
}

/** The error's message, with what the catalog lists alongside it (missing audio, problems, unreadable files). */
export function errorDetail(e: unknown): string[] {
  if (!(e instanceof ApiError)) return [];
  const b = e.body;
  const list = (k: string) => (Array.isArray(b[k]) ? (b[k] as unknown[]).map(String) : []);
  return [...list("problems"), ...list("unreadable"), ...list("missing")];
}

/** Where a person opens a file in Concord Label, when Label has registered with the catalog at an address a browser
 *  can reach. Label registers the URL other services use (POST /api/apps): on the host that is also where its pages
 *  are, but in a container it is a name on ai-net (http://concord-label:8620) that only other containers resolve. */
export function labelBase(apps: { name: string; base_url: string }[] | undefined): string | null {
  const app = (apps ?? []).find((a) => a.name === "label" || a.name === "concord-label");
  if (!app) return null;
  try {
    const host = new URL(app.base_url).hostname;
    const reachable = host === "localhost" || host.includes(".") || host.includes(":");
    return reachable ? app.base_url.replace(/\/+$/, "") : null;
  } catch {
    return null;
  }
}

export function labelFileUrl(base: string, datasetId: string, sha: string, t?: number): string {
  const q = new URLSearchParams({ catalog_dataset_id: datasetId });
  if (t !== undefined) q.set("t", t.toFixed(2));
  return `${base}/label/file/by-sha/${sha}?${q}`;
}
