// Form pieces the hosted Add model form and the hosted model page share: contract, endpoint, limits,
// authentication and the connection test's result lines.

import { useState } from "react";
import type { CheckLine, HostedInfo } from "../lib/api";
import * as I from "./icons";
import { cx } from "./ui";

export const HOSTED_CONTRACTS = [
  { value: "openai|audio.transcriptions", label: "openai / audio.transcriptions (ASR)", path: "/audio/transcriptions", url: "https://api.example.com/v1", model: "e.g. whisper-1", verify: "Conformance checks, then transcription of the 10 s check clip: word, segment or text timing, and WER." },
  { value: "openai|chat.completions", label: "openai / chat.completions (audio LLM or text)", path: "/chat/completions", url: "https://api.example.com/v1", model: "e.g. the provider’s model id", verify: "Conformance checks, then the check clip as an input_audio part. Without audio support it is verified as a text model." },
  { value: "openai|audio.speech", label: "openai / audio.speech (TTS)", path: "/audio/speech", url: "https://api.example.com/v1", model: "e.g. the provider’s TTS model", verify: "Conformance checks only: verified on its protocol, no audio capability probe." },
  { value: "openai|embeddings", label: "openai / embeddings", path: "/embeddings", url: "https://api.example.com/v1", model: "e.g. the provider’s embedding model", verify: "Conformance checks only: verified on its protocol." },
  { value: "openai-inspired|text.itn", label: "openai-inspired / text.itn (inverse text normalisation)", path: "/text/itn", url: "https://api.example.com/v1", model: "e.g. the provider’s ITN model", verify: "Conformance checks, then spoken-form phrases such as “twenty one” and “three thirty p m”: each written form must contain the expected digits, and spans must be valid." },
  { value: "oip-v2|infer", label: "oip-v2 / infer (VAD, diarisation, speech enhancement)", path: "", url: "https://inference.example.com", model: "e.g. silero-vad", verify: "Conformance checks, then infer on the check clip: SEGMENTS → VAD, SEGMENTS + SPEAKERS → diarisation, AUDIO → speech enhancement." },
];

export type Endpoint = {
  contract: string; // "contract|api"
  base_url: string;
  model: string;
  timeout_s: number;
  max_concurrency: number;
  max_upload_mb: number;
};

export type Auth = { auth_type: HostedInfo["auth_type"]; header_name: string; key: string };

export function contractOf(value: string) {
  return HOSTED_CONTRACTS.find((c) => c.value === value) ?? HOSTED_CONTRACTS[0];
}

/** Where an inference call ends up: the gateway drops a leading /v1 for OpenAI-style base URLs. */
export function resolvedUrl(e: Endpoint): string {
  const c = contractOf(e.contract);
  const base = e.base_url.trim().replace(/\/+$/, "") || c.url;
  return e.contract.startsWith("oip-v2") ? `${base}/v2/models/${e.model.trim() || "<model>"}/infer` : base + c.path;
}

export function body(e: Endpoint, a: Auth) {
  const [contract, apiName] = e.contract.split("|");
  return {
    contract,
    api: apiName,
    base_url: e.base_url.trim(),
    model: e.model.trim(),
    timeout_s: e.timeout_s,
    max_concurrency: e.max_concurrency,
    max_upload_mb: e.max_upload_mb,
    auth_type: a.auth_type,
    header_name: a.auth_type === "header" ? a.header_name.trim() : null,
    ...(a.key.trim() ? { key: a.key.trim() } : {}),
  };
}

const input = "min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg";
const field = "flex flex-col gap-1.5 text-xs text-muted";

export function EndpointFields({ value, onChange }: { value: Endpoint; onChange: (e: Endpoint) => void }) {
  const c = contractOf(value.contract);
  const set = (patch: Partial<Endpoint>) => onChange({ ...value, ...patch });
  return (
    <>
      <label className={field}>
        Contract
        <select value={value.contract} onChange={(e) => set({ contract: e.target.value })} className={input}>
          {HOSTED_CONTRACTS.map((x) => <option key={x.value} value={x.value}>{x.label}</option>)}
        </select>
      </label>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className={field}>
          Base URL
          <input type="url" value={value.base_url} onChange={(e) => set({ base_url: e.target.value })} placeholder={c.url} className={input} />
        </label>
        <label className={field}>
          Model name at the provider
          <input type="text" value={value.model} onChange={(e) => set({ model: e.target.value })} placeholder={c.model} className={input} />
        </label>
      </div>
      <div className="flex flex-wrap items-baseline gap-2 text-xs">
        <span className="text-muted">Calls go to</span>
        <span className="num break-all">{resolvedUrl(value)}</span>
      </div>
      <div className="grid gap-3 grid-cols-3">
        <label className={field}>
          Timeout (s)
          <input type="number" min={1} value={value.timeout_s} onChange={(e) => set({ timeout_s: Number(e.target.value) })} className={input} />
        </label>
        <label className={field}>
          Max concurrent calls
          <input type="number" min={1} value={value.max_concurrency} onChange={(e) => set({ max_concurrency: Number(e.target.value) })} className={input} />
        </label>
        <label className={field}>
          Max upload (MB)
          <input type="number" min={1} value={value.max_upload_mb} onChange={(e) => set({ max_upload_mb: Number(e.target.value) })} className={input} />
        </label>
      </div>
    </>
  );
}

export function AuthFields({ value, onChange, keyLabel = "API key", keyPlaceholder = "Paste the key" }: { value: Auth; onChange: (a: Auth) => void; keyLabel?: string; keyPlaceholder?: string }) {
  const [show, setShow] = useState(false);
  const set = (patch: Partial<Auth>) => onChange({ ...value, ...patch });
  return (
    <>
      <fieldset className="m-0 p-0 border-0 flex flex-col gap-1.5">
        <legend className="text-xs text-muted pb-1.5">How the key is sent</legend>
        <span className="flex flex-wrap gap-x-5 gap-y-1">
          {(
            [
              ["bearer", "Authorization: Bearer"],
              ["header", "Custom header"],
              ["none", "None"],
            ] as const
          ).map(([v, l]) => (
            <label key={v} className="flex items-center gap-2 min-h-8 text-[13px] cursor-pointer">
              <input type="radio" name="auth" checked={value.auth_type === v} onChange={() => set({ auth_type: v })} className="accent-primary" />
              {l}
            </label>
          ))}
        </span>
      </fieldset>
      {value.auth_type === "header" && (
        <label className={field}>
          Header name
          <input type="text" value={value.header_name} onChange={(e) => set({ header_name: e.target.value })} placeholder="api-key" className={input} />
        </label>
      )}
      {value.auth_type !== "none" && (
        <div className="flex flex-col gap-1.5">
          <label htmlFor="api-key" className="text-xs text-muted">{keyLabel}</label>
          <span className="flex gap-2">
            <input id="api-key" type={show ? "text" : "password"} autoComplete="off" value={value.key} onChange={(e) => set({ key: e.target.value })} placeholder={keyPlaceholder} className={cx(input, "flex-1 min-w-0 num")} />
            <button type="button" aria-label={show ? "Hide key" : "Show key"} onClick={() => setShow((s) => !s)} className="min-h-10 min-w-10 inline-flex items-center justify-center rounded-md border border-line bg-transparent text-muted hover:text-fg cursor-pointer">
              <I.Eye />
            </button>
          </span>
        </div>
      )}
    </>
  );
}

const GLYPH: Record<string, { c: string; t: string }> = {
  pass: { c: "text-good", t: "✓" },
  warn: { c: "text-warn", t: "!" },
  fail: { c: "text-bad", t: "✕" },
  info: { c: "text-muted", t: "i" },
};

export function TestLines({ lines }: { lines: CheckLine[] }) {
  return (
    <div role="status" className="flex flex-col gap-1.5">
      {lines.map((l, k) => (
        <span key={k} className="flex items-start gap-2 text-[13px]">
          <span className={cx("num w-4 text-center font-semibold shrink-0", GLYPH[l.status]?.c)} aria-label={l.status}>{GLYPH[l.status]?.t}</span>
          <span className="min-w-0 break-words">{l.reason}</span>
        </span>
      ))}
    </div>
  );
}
