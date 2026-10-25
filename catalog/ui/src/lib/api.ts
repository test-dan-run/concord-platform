// Thin fetch wrapper + the response shapes the pages use. The catalog's README is the contract
// (platform/catalog/README.md); field names follow catalog/registry.py and catalog/scheduler.py.

/** An error answer. `body` is the catalog's error object ({message, type, code, ...}), with the extra fields some
 *  routes add, e.g. `size` on a 409 from an upload or `missing` on a publish. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body: { type?: string; [k: string]: unknown } = {},
  ) {
    super(message);
  }
}

export async function errorOf(r: Response): Promise<ApiError> {
  let msg = r.statusText;
  let body = {};
  try {
    const j = await r.json();
    msg = j.error?.message ?? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
    body = j.error ?? {};
  } catch {
    /* not JSON */
  }
  return new ApiError(r.status, msg, body);
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  let payload: BodyInit | undefined;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const r = await fetch(`/api${path}`, { method, headers, body: payload });
  if (!r.ok) throw await errorOf(r);
  if (r.status === 204) return undefined as T;
  return (await r.json()) as T;
}

export const api = {
  get: <T>(p: string) => request<T>("GET", p),
  post: <T>(p: string, b?: unknown) => request<T>("POST", p, b),
  patch: <T>(p: string, b?: unknown) => request<T>("PATCH", p, b),
  put: <T>(p: string, b?: unknown) => request<T>("PUT", p, b),
  del: <T>(p: string) => request<T>("DELETE", p),
};

export type CheckLine = { name: string; status: "pass" | "warn" | "fail" | "info"; reason: string };

export type Capabilities = {
  build_key: string | null;
  status: "pending" | "running" | "verified" | "failed" | "stale";
  usable_as: "asr" | "chat-audio" | "segmenter" | "enhancer" | "itn" | null;
  tasks: string[];
  timing_level: string | null;
  output_detail: string[];
  protocol: { contract: string; api: string; routes: string[]; response_format?: string } | null;
  rtf: number | null;
  verified_at: string | null;
  report: {
    ok: boolean;
    lines: CheckLine[];
    sample: {
      transcript?: string;
      wer?: number;
      der?: number | null;
      si_sdr_input_db?: number;
      si_sdr_output_db?: number;
      length_diff_samples?: number;
      itn?: { input: string; output: string }[];
    };
  } | null;
};

export type Tensor = { name: string; datatype: string; shape: number[] };

export type ModelState = "running" | "stopped" | "starting" | "stopping" | "queued" | "failed" | "hosted";

export type ConnectionTest = { ok: boolean; lines: CheckLine[]; signature: Model["signature"]; tested_at?: string };

/** A hosted endpoint's settings (no key: only its last four characters). */
export type HostedInfo = {
  base_url: string;
  model: string;
  host: string;
  auth_type: "bearer" | "header" | "none";
  header_name: string | null;
  key_hint: string | null;
  timeout_s: number;
  max_concurrency: number;
  max_upload_mb: number;
  resolved_url: string;
  data_ack_at: string;
  created_at: string;
  next_check_at: string | null;
  last_test: ConnectionTest | null;
};

export type Model = {
  id: string;
  aliases: string[];
  registered: boolean;
  display_name: string;
  tags: string[];
  parent_id: string | null;
  registered_at: string | null;
  description: string | null;
  tasks: string[];
  kind: string;
  engine: string;
  contract: string;
  api: string;
  signature: { inputs: Tensor[]; outputs: Tensor[] } | null;
  gpu: { fraction: number; gb: number } | null;
  state: ModelState;
  state_detail: string | null;
  leased_by: string[];
  in_flight: number;
  build_key: string | null;
  build_pending: string | null;
  image: string | null;
  capabilities: Capabilities | null;
  cold_start_s: number;
  cold_start_measured: boolean;
  endpoint: { gateway_url: string; base_path: string };
  last_used: string | null;
  calls: number;
  errors?: number;
  served_name: string;
  hosted: HostedInfo | null;
};

export type Lease = {
  id: string;
  kind: "lease" | "start";
  holder: string;
  models: string[];
  state: "queued" | "starting" | "active" | "released" | "expired" | "failed" | "done";
  position: number | null;
  waiting_for: string | null;
  error: string | null;
  created_at: string | null;
  granted_at: string | null;
  ready_at: string | null;
  expires_at: string | null;
};

export type PlatformInfo = {
  gpu_total_gb: number;
  util_limit: number;
  reserve_gb: number;
  swaps?: number; // GPU model swaps counted by the scheduler
  used_fraction: number;
  gpu_models: { id: string; fraction: number }[];
  platform_error: string | null;
};

export type Usage = { id: string; calls: number; errors: number; in_flight: number; last_used: string | null };
export type App = { name: string; base_url: string };
export type IndexStat = { owner: string; outputs: number };

export type ExportModel = {
  id: string;
  display_name: string;
  registered: boolean;
  contract: string;
  api: string;
  build_key: string | null;
  image: string;
  image_size: number | null;
  weights_size: number;
  weights_files: number;
  weights_note: string | null;
  state: ModelState;
  verification: Capabilities["status"] | null;
  exportable: boolean;
  reason: string | null;
};

export type ExportJob = {
  id: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  phase: string | null;
  phase_index: number;
  phase_count: number;
  bytes_done: number;
  bytes_expected: number;
  bytes_written: number | null;
  error: string | null;
  failed_phase: string | null;
  request: { models: string[]; destination: "bundle" | "registry"; registry: string | null; images: boolean; weights: boolean; verification: boolean };
  name: string | null;
  available: boolean;
  created_at: string;
  finished_at: string | null;
};

/** A text normaliser: an ITN service, or a text.transform service whose manifest declares text normalisation.
 * text.transform alone is the general text-to-text API (redaction, transliteration, ...), so the task decides. */
export function isTextNormaliser(m: Model): boolean {
  if ((m.capabilities?.usable_as ?? probeFor(m)) === "itn") return true;
  return m.contract === "openai-inspired" && m.api === "text.transform" && m.tasks.includes("text-normalization");
}

/** Which capability probe applies to a service, before it is verified (catalog/verifier/probes.py). */
export function probeFor(m: Model): Capabilities["usable_as"] {
  if (m.contract === "openai" && m.api === "audio.transcriptions") return "asr";
  if (m.contract === "openai" && m.api === "chat.completions" && m.tasks.some((t) => ["speech-recognition", "audio-understanding", "audio-transcription", "audio"].includes(t))) return "chat-audio";
  if (m.contract === "oip-v2" && m.signature) {
    const ins = m.signature.inputs.map((t) => t.name);
    const outs = m.signature.outputs.map((t) => t.name);
    if (ins.includes("AUDIO") && ins.includes("SAMPLE_RATE") && outs.includes("SEGMENTS")) return "segmenter";
    if (ins.includes("AUDIO") && ins.includes("SAMPLE_RATE") && outs.includes("AUDIO")) return "enhancer";
  }
  if (m.contract === "openai-inspired" && m.api === "text.itn") return "itn";
  return null;
}

export type ModelCardSection = {
  key: "upstream" | "service";
  title: string;
  markdown: string;
  evaluation: string | null; // the upstream card's results, shown collapsed
  full_url: string | null; // the whole upstream card at the same revision
  origin: string;
  fetched_at?: string | null;
};

/** GET /api/models/{id}/card (catalog/cards.py): provenance, license, upstream metadata, checkpoint facts, docs. */
export type ModelCard = {
  id: string;
  description: string | null;
  tasks: string[];
  model_weights: boolean;
  source: {
    host: "huggingface" | "github" | "provider";
    repo: string;
    revision: string | null;
    pinned: boolean;
    origin: string;
    url: string;
    downloaded: string | null;
    subpath: string | null;
    version: string | null;
  } | null;
  license: { name: string | null; link: string | null; text: string | null; file: string | null };
  upstream: { base_model: string[]; languages: string[]; datasets: string[] };
  facts: { label: string; value: string }[];
  sections: ModelCardSection[];
  gaps: string[];
};

// GET /models/{id}/batch (catalog/batching.py): the current build's batch profile.
export type BatchProfile = {
  kind: "audio" | "kv";
  base: number;
  a: number;
  b: number;
  max_item_s: number | null;
  points: { n: number; length_s: number; peak_gib: number; seconds: number }[];
  oom: { length_s: number; stopped: string }[];
  method: string;
  kv_tokens: number | null;
  kv_gib: number | null;
  fixed_gib: number | null;
  max_model_len: number | null;
  max_concurrency: number | null;
  given_gib: number | null;
};

export type BatchView = {
  model: string;
  build_key: string | null;
  measuring: "pending" | "running" | null;
  status: "none" | "measured" | "failed";
  error: string | null;
  measured_at: string | null;
  profile?: BatchProfile;
  table?: { unit: "s" | "tokens"; budgets_gb: number[]; full_gb: number | null; rows: { length: number; batch: number[] }[] };
  // custom GPU images only: the limits the next start gets at the full budget, and the ones set by hand
  limits?: BatchLimits & { vram_gb: number; measured: BatchLimits; source: Record<keyof BatchLimits, "override" | "measured" | "default"> };
  override?: (BatchLimits & { set_at: string }) | null;
  // null when the service is not running; started_with is null when it was started outside the catalog
  running: {
    vram_gb: number | null;
    started_with: BatchLimits | null;
    batchers: (BatchLimits & { name: string; backoffs: number; lowered_len_s: number | null })[];
  } | null;
};

// null = not set: the image's own default
export type BatchLimits = { max_batch: number | null; max_len_s: number | null };
