// How apps call models through the catalog, and how models are registered, verified, scheduled and moved.
// Each section is data (title, searchable text, body), so the search box can filter and highlight them.

import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import * as I from "../components/icons";
import { Button, cx, PageHeader } from "../components/ui";

type Section = { id: string; title: string; text: string; body: (gw: string) => ReactNode };

const P = ({ children }: { children: ReactNode }) => <p className="m-0 text-[13px] text-muted leading-relaxed">{children}</p>;
const Code = ({ children }: { children: ReactNode }) => <span className="num text-fg text-[12px]">{children}</span>;
const Pre = ({ children }: { children: string }) => (
  <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{children}</pre>
);

// Code and tables shown in a section are part of its search text too, so each is written once here.
const QUICK = `from aicore import client, model_name   # AICORE_CATALOG={gateway}
c = client("whisper-large-v3")                  # openai.OpenAI via the gateway
c.audio.transcriptions.create(
    model=model_name("whisper-large-v3"),
    file=open("call.wav", "rb"),
)`;
const FIND = `GET /api/models?task=asr
GET /api/models?task=diar
GET /api/models?task=asr&tag=call-centre`;
const LOOKUP = `POST /api/index/lookup
{"app": "label", "keys": [{"build_key", "audio_sha256",
  "span_start", "span_end", "params_hash"}]}
→ {"hits": [null | {"owner", "output_id", "owner_url"}]}

POST {owner_url}/api/outputs/batch  {"ids": [812]}
→ {"outputs": [{"id": 812, "output", …}]}`;
const IMPORT = `python -m catalog import catalog-export-2026-10-23.tar   # on the target`;
const PROBES = [
  ["words[] / segments[] / text", "ASR with word times, segment times or text only"],
  ["SEGMENTS", "VAD"],
  ["SEGMENTS + SPEAKERS", "Diarisation"],
  ["AUDIO", "Speech enhancement"],
  ["output + spans (text.itn)", "Text normalisation (ITN): spoken form to written form"],
];

export const SECTIONS: Section[] = [
  {
    id: "quick-start",
    title: "Quick start",
    text: `Apps never talk to a model service directly. Every call goes through the catalog's gateway, which speaks the same protocol as the service, starts the model on demand and holds it while the call is open. ${QUICK} Without the client, point any OpenAI-compatible SDK at {gateway}/m/<model>/v1. OIP v2 models are called at {gateway}/v2/models/<model>/infer. Request shapes for each service type are on the model contract page.`,
    body: (gw) => (
      <>
        <P>Apps never talk to a model service directly. Every call goes through the catalog's gateway, which speaks the same protocol as the service, starts the model on demand and holds it while the call is open.</P>
        <Pre>{QUICK.replaceAll("{gateway}", gw)}</Pre>
        <P>
          Without the client, point any OpenAI-compatible SDK at <Code>{gw}/m/&lt;model&gt;/v1</Code>. OIP v2 models are called at <Code>{gw}/v2/models/&lt;model&gt;/infer</Code>. Request shapes for each service type are on the{" "}
          <Link to="/contract">model contract</Link> page.
        </P>
      </>
    ),
  },
  {
    id: "finding",
    title: "Finding models",
    text: `Look models up by what they do rather than hard-coding names, so a model can be swapped without changing the app. task matches the tasks verification observed (asr, diar, vad, enhance, itn) or the tasks the manifest declares; tag matches the tags people add in the catalog. Capabilities in the response come from verification of the exact build, not from the manifest. ${FIND} On the Models page, group by model type to see ASR, VAD / diarisation, speech enhancement and text normalisation (ITN) models together.`,
    body: () => (
      <>
        <P>
          Look models up by what they do rather than hard-coding names, so a model can be swapped without changing the app. <Code>task</Code> matches the tasks verification observed (asr, diar, vad, enhance, itn) or the tasks the manifest declares; <Code>tag</Code> matches the tags people add in the catalog. Capabilities in the response come from verification of the exact build, not from the manifest.
        </P>
        <Pre>{FIND}</Pre>
        <P>
          On the <Link to="/">Models</Link> page, group by model type to see ASR, VAD / diarisation, speech enhancement and text normalisation (ITN) models together.
        </P>
      </>
    ),
  },
  {
    id: "leases",
    title: "Leases and the queue",
    text: "A job that needs a model for a while takes a lease. Leases are kept alive with a heartbeat every 30 s and expire after three missed heartbeats. The scheduler never stops a leased model; only POST /api/models/<model>/stop?force=1, the owner's override, does. All apps share one first-come-first-served queue. A request behind the head of the queue passes only if it needs no new GPU memory. Models with no lease and no open calls are stopped when room is needed. Models not started on purpose are also stopped when idle: 30 s after their last lease ends (CATALOG_RELEASE_LINGER_S), or 600 s after the last call when the gateway started them for a call without a lease (CATALOG_GATEWAY_IDLE_S). Hosted models use no GPU, so they never wait in the queue. See what is waiting and what is held on Queue and leases.",
    body: () => (
      <>
        <P>
          A job that needs a model for a while takes a lease. Leases are kept alive with a heartbeat every 30 s and expire after three missed heartbeats. The scheduler never stops a leased model; only <Code>POST /api/models/&lt;model&gt;/stop?force=1</Code>, the owner's override, does.
        </P>
        <ul className="m-0 pl-5 text-[13px] text-muted leading-relaxed flex flex-col gap-1">
          <li>All apps share one first-come-first-served queue.</li>
          <li>A request behind the head of the queue passes only if it needs no new GPU memory.</li>
          <li>Models with no lease and no open calls are stopped when room is needed.</li>
          <li>
            Models not started on purpose are also stopped when idle: 30 s after their last lease ends (<Code>CATALOG_RELEASE_LINGER_S</Code>), or 600 s after the last call when the gateway started them for a call without a lease (<Code>CATALOG_GATEWAY_IDLE_S</Code>).
          </li>
          <li>Hosted models use no GPU, so they never wait in the queue.</li>
        </ul>
        <P>
          See what is waiting and what is held on <Link to="/queue">Queue and leases</Link>.
        </P>
      </>
    ),
  },
  {
    id: "registering",
    title: "Registering models",
    text: "A model must speak one of the catalog's contracts. There are two ways to add one from Add model: Platform service Runs on this host, on the GPU or, for CPU-only services, on the CPU. Pick a service from platform/services/*.yaml; protocol, image and footprint come from its manifest. Add tags and a display name. Hosted endpoint Already served online. Give its contract, base URL, model name and API key. The key is stored encrypted in the catalog and never shown again; the gateway adds it to each call. model_name() returns the provider's model name. Data sent to it leaves this network.",
    body: () => (
      <>
        <P>
          A model must speak one of the catalog's contracts. There are two ways to add one from <Link to="/add">Add model</Link>:
        </P>
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="border border-track rounded-md p-3 flex flex-col gap-1">
            <span className="text-[13px] font-medium">Platform service</span>
            <span className="text-xs text-muted leading-relaxed">
              Runs on this host, on the GPU or, for CPU-only services, on the CPU. Pick a service from <Code>platform/services/*.yaml</Code>; protocol, image and footprint come from its manifest. Add tags and a display name.
            </span>
          </div>
          <div className="border border-track rounded-md p-3 flex flex-col gap-1">
            <span className="text-[13px] font-medium">Hosted endpoint</span>
            <span className="text-xs text-muted leading-relaxed">
              Already served online. Give its contract, base URL, model name and API key. The key is stored encrypted in the catalog and never shown again; the gateway adds it to each call. <Code>model_name()</Code> returns the provider's model name. Data sent to it leaves this network.
            </span>
          </div>
        </div>
      </>
    ),
  },
  {
    id: "verification",
    title: "Builds and verification",
    text: `Every record belongs to one build. For a platform service the build key is image id + config hash + weights hash; for a hosted model it is contract + api + base URL + model name. A rebuilt image, new weights or a change to a manifest field in the config hash (engine, model, args, contract, api, signature, build; not port, alias, description, tasks or GPU size) make a new build, which is verified again automatically. Hosted models are also re-verified daily, because a provider can change the model behind the same name. Verification runs the contract's conformance checks, then a capability probe: audio models get a 10.56 s two-speaker check clip, and ITN models get spoken-form phrases. The probe decides what the build can do: What each probe result means. Returns, detected as: ${PROBES.flat().join(" ")} Pass and fail rules for each are on the model contract page.`,
    body: () => (
      <>
        <P>
          Every record belongs to one build. For a platform service the build key is image id + config hash + weights hash; for a hosted model it is contract + api + base URL + model name. A rebuilt image, new weights or a change to a manifest field in the config hash (engine, model, args, contract, api, signature, build; not port, alias, description, tasks or GPU size) make a new build, which is verified again automatically. Hosted models are also re-verified daily, because a provider can change the model behind the same name.
        </P>
        <P>Verification runs the contract's conformance checks, then a capability probe: audio models get a 10.56 s two-speaker check clip, and ITN models get spoken-form phrases. The probe decides what the build can do:</P>
        <div className="overflow-x-auto scroll-thin">
          <table className="w-full border-collapse text-[13px]">
            <caption className="sr-only">What each probe result means</caption>
            <thead>
              <tr className="text-[11px] text-muted uppercase tracking-wider text-left">
                <th scope="col" className="font-medium px-3 py-2 border-b border-line">Returns</th>
                <th scope="col" className="font-medium px-3 py-2 border-b border-line">Detected as</th>
              </tr>
            </thead>
            <tbody>
              {PROBES.map(([r, d]) => (
                <tr key={r} className="border-b border-track">
                  <td className="px-3 py-2 num text-xs">{r}</td>
                  <td className="px-3 py-2 text-xs">{d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <P>
          Pass and fail rules for each are on the <Link to="/contract">model contract</Link> page.
        </P>
      </>
    ),
  },
  {
    id: "outputs",
    title: "Output index",
    text: `Apps keep their own model outputs; the catalog keeps only where each one is. Before running a model on a file, an app can look it up in the index. When an app already has an output for the same key, the asking app copies it from that app instead of running the model again; a hit whose owner no longer has the output is computed again. ${LOOKUP} The key is the build key, the sha256 of the original audio, the span in seconds (−1, −1 for the whole file) and the app's hash of its request parameters. Because the key includes the build, a new build never matches earlier outputs. An app records its own outputs with POST /api/index/record. See Output index.`,
    body: () => (
      <>
        <P>
          Apps keep their own model outputs; the catalog keeps only where each one is. Before running a model on a file, an app can look it up in the index. When an app already has an output for the same key, the asking app copies it from that app instead of running the model again; a hit whose owner no longer has the output is computed again.
        </P>
        <Pre>{LOOKUP}</Pre>
        <P>
          The key is the build key, the sha256 of the original audio, the span in seconds (−1, −1 for the whole file) and the app's hash of its request parameters. Because the key includes the build, a new build never matches earlier outputs. An app records its own outputs with <Code>POST /api/index/record</Code>. See <Link to="/index">Output index</Link>.
        </P>
      </>
    ),
  },
  {
    id: "export",
    title: "Moving to another environment",
    text: `From Export, pick models and where they go: Bundle file: one .tar with images, weights, registration and verification records, for offline or air-gapped targets. Image layers shared between models are written once. Push to a registry: images are retagged and pushed. The bundle holds the rest: the manifest and each model's service.yaml, registration and verification records, and its weights unless you leave them out. An export runs in the background and shows its progress; a failed export names the step that failed and removes the partial bundle. ${IMPORT} The import loads the images, manifests, weights and registrations; a manifest on the target that differs from the bundle's is kept unless you pass --force. The target verifies each build it has no record of: a build it already verified keeps that record, and the bundle's verification.json is not imported. A matching build key keeps its existing outputs.`,
    body: () => (
      <>
        <P>
          From <Link to="/export">Export</Link>, pick models and where they go:
        </P>
        <ul className="m-0 pl-5 text-[13px] text-muted leading-relaxed flex flex-col gap-1">
          <li>
            <span className="text-fg">Bundle file</span>: one .tar with images, weights, registration and verification records, for offline or air-gapped targets. Image layers shared between models are written once.
          </li>
          <li>
            <span className="text-fg">Push to a registry</span>: images are retagged and pushed. The bundle holds the rest: the manifest and each model's <Code>service.yaml</Code>, registration and verification records, and its weights unless you leave them out.
          </li>
        </ul>
        <P>An export runs in the background and shows its progress; a failed export names the step that failed and removes the partial bundle.</P>
        <Pre>{IMPORT}</Pre>
        <P>
          The import loads the images, manifests, weights and registrations; a manifest on the target that differs from the bundle's is kept unless you pass <Code>--force</Code>. The target verifies each build it has no record of: a build it already verified keeps that record, and the bundle's <Code>verification.json</Code> is not imported. A matching build key keeps its existing outputs.
        </P>
      </>
    ),
  },
];

const esc = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** A window of the section text around the first match, split into plain and highlighted parts. */
function snippet(text: string, terms: string[]): { text: string; hit: boolean }[] {
  const low = text.toLowerCase();
  const at = Math.max(0, low.indexOf(terms[0]));
  let start = Math.max(0, at - 70);
  if (start > 0) {
    const sp = text.indexOf(" ", start);
    start = sp > 0 && sp < at ? sp + 1 : start;
  }
  let end = Math.min(text.length, at + 170);
  if (end < text.length) {
    const sp = text.lastIndexOf(" ", end);
    end = sp > at ? sp : end;
  }
  const cut = (start > 0 ? "… " : "") + text.slice(start, end) + (end < text.length ? " …" : "");
  const re = new RegExp(`(${terms.map(esc).join("|")})`, "gi");
  return cut.split(re).filter(Boolean).map((part) => ({ text: part, hit: terms.includes(part.toLowerCase()) }));
}

export function DocsPage() {
  const [q, setQ] = useState("");
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<{ public_url: string }>("/health"), staleTime: 60_000 });
  const gw = health.data?.public_url ?? window.location.origin;
  const raw = q.trim();
  const terms = raw.toLowerCase().split(/\s+/).filter(Boolean);
  const searching = terms.length > 0;
  const hay = (s: Section) => `${s.title} ${s.text.replaceAll("{gateway}", gw)}`.toLowerCase();
  const matches = SECTIONS.filter((s) => terms.every((t) => hay(s).includes(t)));
  const count = (s: Section) => terms.reduce((a, t) => a + hay(s).split(t).length - 1, 0);

  return (
    <>
      <PageHeader
        crumbs="Reference"
        title="Docs"
        subtitle="How apps call models through the catalog, and how models are registered, verified, scheduled and moved."
        actions={
          <label className="relative flex w-full sm:w-[280px]">
            <span className="sr-only">Search docs</span>
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-subtle">
              <I.Search size={15} />
            </span>
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search docs"
              className="w-full min-h-10 bg-canvas border border-line rounded-md pl-8 pr-2 text-[13px] text-fg"
            />
          </label>
        }
      />
      {searching && (
        <section aria-labelledby="results-title" aria-live="polite" className="bg-panel border border-track rounded-lg p-4 flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <h2 id="results-title" className="m-0 text-sm font-semibold">
              {matches.length ? `${matches.length} of ${SECTIONS.length} sections match “${raw}”` : "No results"}
            </h2>
            <Button size="sm" className="ml-auto" onClick={() => setQ("")}>Clear search</Button>
          </div>
          {matches.length > 0 ? (
            <>
              <div className="flex flex-col gap-2">
                {matches.map((s) => {
                  const n = count(s);
                  return (
                    <a key={s.id} href={`#${s.id}`} className="flex flex-col gap-1 border border-track rounded-md p-3 no-underline !text-fg hover:bg-track/40">
                      <span className="flex items-center gap-2 text-[13px] font-medium">
                        {s.title}
                        <span className="text-[11px] text-muted font-normal num">{n} match{n === 1 ? "" : "es"}</span>
                      </span>
                      <span className="text-xs text-muted leading-relaxed">
                        {snippet(s.text.replaceAll("{gateway}", gw), terms).map((part, k) =>
                          part.hit ? <mark key={k} className="bg-primary/25 text-fg rounded-sm">{part.text}</mark> : <span key={k}>{part.text}</span>,
                        )}
                      </span>
                    </a>
                  );
                })}
              </div>
              <span className="text-[11px] text-subtle">Only matching sections are shown below. Every word you type must appear in a section for it to match.</span>
            </>
          ) : (
            <div className="border border-dashed border-line rounded-md p-6 text-center text-[13px] text-muted flex flex-col gap-1">
              <span>No section mentions “{raw}”.</span>
              <span>
                Try a route such as <Code>/api/models</Code>, or a word such as lease, tag or build.
              </span>
            </div>
          )}
        </section>
      )}
      <div className="flex flex-col lg:flex-row gap-5 items-start">
        <nav aria-label="On this page" className="w-full lg:w-[200px] shrink-0 lg:sticky lg:top-5 flex flex-col gap-1">
          <span className="text-[11px] text-muted uppercase tracking-wider px-2.5 pb-1">On this page</span>
          {SECTIONS.map((s) => {
            const on = !searching || matches.includes(s);
            return (
              <a
                key={s.id}
                href={`#${s.id}`}
                aria-disabled={!on || undefined}
                className={cx("min-h-9 flex items-center px-2.5 rounded-md text-[13px] no-underline hover:bg-track", on ? "!text-muted hover:!text-fg" : "!text-subtle/60 pointer-events-none")}
              >
                {s.title}
              </a>
            );
          })}
        </nav>
        <div className="flex-1 min-w-0 flex flex-col gap-5">
          {SECTIONS.filter((s) => !searching || matches.includes(s)).map((s) => (
            <article key={s.id} id={s.id} aria-labelledby={`${s.id}-title`} className="bg-panel border border-track rounded-lg p-4 flex flex-col gap-3 scroll-mt-5">
              <h2 id={`${s.id}-title`} className="m-0 text-sm font-semibold">{s.title}</h2>
              {s.body(gw)}
            </article>
          ))}
        </div>
      </div>
    </>
  );
}
