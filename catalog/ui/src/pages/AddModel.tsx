import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, isTextNormaliser, probeFor, type ConnectionTest, type Model } from "../lib/api";
import * as I from "../components/icons";
import { AuthFields, body, contractOf, EndpointFields, TestLines, type Auth, type Endpoint } from "../components/hosted";
import { gpuLabel, StateCell } from "../components/model";
import { TagInput } from "../components/TagInput";
import { Button, Chip, cx, Empty, ErrorNote, Fact, PageHeader, Panel, Select, Spinner } from "../components/ui";

const GROUPS = [
  { key: "speech", title: "Speech models", note: "ASR, VAD, diarisation and speech enhancement: verified with a capability probe on the check clip" },
  { key: "itn", title: "Text normalisation", note: "ITN services (text.itn), verified with spoken-form phrases, and text.transform normalisers, verified on their protocol" },
  { key: "model", title: "Other models", note: "Chat, embeddings, TTS and others: verified on their protocol" },
  { key: "utility", title: "Utilities", note: "Non-AI services: verified on their protocol" },
];

function group(m: Model): string {
  if (isTextNormaliser(m)) return "itn"; // before the kind: text-normalizer is a utility
  const probe = probeFor(m);
  if (probe) return "speech"; // before the kind: wpe-dereverb is a utility with the enhancement signature
  if (m.kind === "utility") return "utility";
  return "model";
}

function SourceCard({ on, onClick, icon, title, note }: { on: boolean; onClick: () => void; icon: ReactNode; title: string; note: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={cx("flex-1 text-left flex items-center gap-3 min-h-16 px-4 py-3 rounded-lg border cursor-pointer text-fg", on ? "border-primary bg-primary/10" : "border-track bg-panel hover:bg-track/60")}
    >
      <span className={cx("inline-flex", on ? "text-primary" : "text-muted")}>{icon}</span>
      <span className="flex flex-col">
        <span className="text-[13px] font-medium">{title}</span>
        <span className="text-xs text-muted">{note}</span>
      </span>
    </button>
  );
}

export function AddModel() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const name = params.get("name");
  const all = useQuery({ queryKey: ["models", "all"], queryFn: () => api.get<Model[]>("/models?all=1") });
  const [q, setQ] = useState("");
  const services = (all.data ?? []).filter((m) => !m.hosted); // ?all=1 lists the hosted endpoints too
  const entry = services.find((m) => m.id === name) ?? null;
  const hosted = params.get("source") === "hosted";
  const pick = (source: "platform" | "hosted") => setParams(source === "hosted" ? { source } : {});

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/">Models</Link> / Add model</>}
        title="Add model"
        subtitle={<>Register a service the platform runs, or a model already served online. Either way it must speak a catalog contract, and capabilities are measured by verification. See the <Link to="/contract">model contract</Link>.</>}
      />
      <div role="group" aria-label="Where the model runs" className="flex flex-col sm:flex-row gap-3">
        <SourceCard on={!hosted} onClick={() => pick("platform")} icon={<I.Cpu size={20} />} title="Platform service" note="Runs here on the GPU, from platform/services/*.yaml" />
        <SourceCard on={hosted} onClick={() => pick("hosted")} icon={<I.Globe size={20} />} title="Hosted endpoint" note="Already served online: register its URL and API key" />
      </div>
      {hosted ? (
        <HostedForm
          models={(all.data ?? []).filter((m) => m.registered)}
          onAdded={(m) => {
            void qc.invalidateQueries();
            nav(`/models/${m.id}`);
          }}
        />
      ) : (
      <div className="flex flex-col xl:flex-row gap-5 items-start">
        <Panel className="w-full xl:w-[340px] shrink-0" title="Platform services" subtitle="From platform/services/*.yaml">
          <label className="relative flex">
            <span className="sr-only">Search services</span>
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-subtle"><I.Search size={15} /></span>
            <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search services" className="w-full min-h-9 bg-canvas border border-line rounded-md pl-8 pr-2 text-[13px] text-fg" />
          </label>
          {all.isLoading && <Spinner />}
          <ErrorNote error={all.error} />
          {GROUPS.map((g) => {
            const items = services.filter((m) => group(m) === g.key && (!q || `${m.id} ${m.aliases.join(" ")} ${m.description ?? ""} ${m.tasks.join(" ")}`.toLowerCase().includes(q.toLowerCase())));
            if (!items.length) return null;
            return (
              <div key={g.key} className="flex flex-col gap-1">
                <span className="text-[11px] text-muted uppercase tracking-wider" title={g.note}>{g.title}</span>
                {items.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    aria-pressed={m.id === name}
                    onClick={() => setParams({ name: m.id })}
                    title={m.description ?? undefined}
                    className={cx("text-left flex items-center gap-2.5 min-h-10 px-2.5 rounded-md border cursor-pointer", m.id === name ? "bg-active border-focus/60" : "bg-transparent border-transparent hover:bg-track")}
                  >
                    <span className="flex flex-col min-w-0 flex-1">
                      <span className="text-[13px] truncate">{m.id}</span>
                      <span className="text-[11px] text-subtle truncate num">{m.contract} / {m.api} · {gpuLabel(m)}</span>
                    </span>
                    {m.registered && <Chip tone="good">added</Chip>}
                  </button>
                ))}
              </div>
            );
          })}
        </Panel>
        <div className="flex-1 w-full min-w-0">
          {!entry ? (
            <Empty>Pick a platform service to see what its manifest says and add it to the catalog.</Empty>
          ) : (
            <EntryForm
              key={entry.id}
              entry={entry}
              models={(all.data ?? []).filter((m) => m.registered)}
              onAdded={(m) => {
                void qc.invalidateQueries();
                nav(`/models/${m.id}`);
              }}
            />
          )}
        </div>
      </div>
      )}
    </>
  );
}

function EntryForm({ entry, models, onAdded }: { entry: Model; models: Model[]; onAdded: (m: Model) => void }) {
  const [displayName, setDisplayName] = useState(entry.registered && entry.display_name !== entry.id ? entry.display_name : "");
  const [tags, setTags] = useState<string[]>(entry.registered ? entry.tags : []);
  const [parent, setParent] = useState<string>(entry.parent_id ?? "");
  const add = useMutation({
    mutationFn: () => api.post<Model>(`/models/${entry.id}/register`, { display_name: displayName || null, tags, parent_id: parent || null }),
    onSuccess: onAdded,
  });
  const probe = probeFor(entry);
  return (
    <div className="flex flex-col 2xl:flex-row gap-5 items-start">
      <div className="flex-1 w-full flex flex-col gap-5 min-w-0">
        <Panel title={entry.id} subtitle={entry.description ?? undefined} actions={<StateCell m={entry} compact />}>
          <div className="grid gap-x-6 sm:grid-cols-2">
            <Fact k="Contract / API" v={`${entry.contract} / ${entry.api}`} />
            <Fact k="Engine" v={entry.engine} />
            <Fact k="Aliases" v={entry.aliases.join(", ") || "—"} />
            <Fact k="GPU footprint" v={gpuLabel(entry)} />
            <Fact k="Image" v={<span className="text-[11px] break-all">{entry.image ?? "—"}</span>} />
            <Fact k="Build key" v={entry.build_key ?? entry.build_pending ?? "—"} />
          </div>
          {entry.signature && (
            <div className="flex flex-col gap-1">
              <span className="text-[11px] text-muted uppercase tracking-wider">OIP signature</span>
              <table className="text-xs border-collapse">
                <tbody>
                  {[...entry.signature.inputs.map((t) => ["in", t] as const), ...entry.signature.outputs.map((t) => ["out", t] as const)].map(([dir, t]) => (
                    <tr key={`${dir}${t.name}`} className="border-t border-track">
                      <td className="py-1 pr-3 text-muted">{dir}</td>
                      <td className="py-1 pr-3 num">{t.name}</td>
                      <td className="py-1 pr-3 num text-muted">{t.datatype}</td>
                      <td className="py-1 num text-muted">[{t.shape.join(", ")}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {entry.tasks.length > 0 && <span className="flex flex-wrap gap-1">{entry.tasks.map((t) => <Chip key={t}>{t}</Chip>)}</span>}
        </Panel>
        <Panel title="Details" subtitle="Optional. Tags are how apps find models (GET /api/models?tag=…).">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1.5 text-xs text-muted">
              Display name
              <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder={entry.id} className="min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg" />
            </label>
            <Select label="Derived from" value={parent} onChange={(e) => setParent(e.target.value)} className="!min-h-10 flex-1">
              <option value="">None</option>
              {models.filter((m) => m.id !== entry.id).map((m) => (
                <option key={m.id} value={m.id}>{m.display_name}</option>
              ))}
            </Select>
          </div>
          <TagInput value={tags} onChange={setTags} label="Tags" />
        </Panel>
      </div>
      <aside className="w-full 2xl:w-[300px] shrink-0 flex flex-col gap-4">
        <Panel title="Verification" subtitle="Runs after adding, on every new build, and on demand">
          <span className="text-[13px] text-muted leading-relaxed">
            {probe === "asr" && "Conformance checks, then transcription of the 10 s check clip: word, segment or text timing, and WER."}
            {probe === "chat-audio" && "Conformance checks, then the check clip as an input_audio part: is the answer its transcript?"}
            {probe === "segmenter" && "Conformance checks, then segments of the check clip: SPEAKERS means diarisation, otherwise VAD."}
            {probe === "enhancer" && "Conformance checks, then the check clip with babble noise at 5 dB SNR (a synthetic room response for dereverberation): length, rate, NaN, clipping and SI-SDR against the clean clip."}
            {probe === "itn" && "Conformance checks, then spoken-form phrases such as “twenty one” and “three thirty p m”: each written form must contain the expected digits."}
            {!probe && "The platform's conformance checks for its contract. No audio capability probe applies."}
          </span>
          <span className="text-[11px] text-subtle">It takes a lease like any job, so it waits its turn and may start the model ({gpuLabel(entry)}).</span>
        </Panel>
        <ErrorNote error={add.error} />
        <Button variant="primary" disabled={add.isPending} onClick={() => add.mutate()}>
          {entry.registered ? "Save changes" : "Add model and verify"}
        </Button>
      </aside>
    </div>
  );
}

function HostedForm({ models, onAdded }: { models: Model[]; onAdded: (m: Model) => void }) {
  const [ep, setEp] = useState<Endpoint>({ contract: "openai|audio.transcriptions", base_url: "", model: "", timeout_s: 120, max_concurrency: 4, max_upload_mb: 25 });
  const [auth, setAuth] = useState<Auth>({ auth_type: "bearer", header_name: "", key: "" });
  const [id, setId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [parent, setParent] = useState("");
  const [ack, setAck] = useState(false);
  const [tested, setTested] = useState<ConnectionTest | null>(null);
  const c = contractOf(ep.contract);
  const test = useMutation({ mutationFn: () => api.post<ConnectionTest>("/hosted/test", body(ep, auth)), onSuccess: setTested });
  const add = useMutation({
    mutationFn: () => api.post<Model>("/hosted", { ...body(ep, auth), id: id.trim(), display_name: displayName.trim() || null, tags, parent_id: parent || null, data_leaves_network: ack }),
    onSuccess: onAdded,
  });
  const untest = <T,>(f: (v: T) => void) => (v: T) => {
    f(v);
    setTested(null);
  };
  const hasKey = auth.auth_type === "none" || (!!auth.key.trim() && (auth.auth_type !== "header" || !!auth.header_name.trim()));
  const canTest = !!ep.base_url.trim() && !!ep.model.trim() && hasKey;
  const hint = !ep.base_url.trim()
    ? "Enter the base URL."
    : !ep.model.trim()
      ? "Enter the model name at the provider."
      : !hasKey
        ? "Enter the API key."
        : !tested?.ok
          ? "Test the connection first."
          : !id.trim()
            ? "Enter the catalog name."
            : !ack
              ? "Confirm that data may leave this network."
              : "";
  return (
    <div className="flex flex-col xl:flex-row gap-5 items-start">
      <div className="flex-1 w-full flex flex-col gap-5 min-w-0">
        <Panel title="Endpoint" subtitle="The service must speak one of the catalog's contracts. The gateway forwards calls to it unchanged.">
          <EndpointFields value={ep} onChange={untest(setEp)} />
          <span className="text-[11px] text-subtle">Calls past the concurrency limit wait in the gateway; files over the upload limit are rejected before they are sent.</span>
        </Panel>
        <Panel title="Authentication" subtitle="The key stays in the catalog's secret store. The gateway adds it to each call; apps never see it.">
          <AuthFields value={auth} onChange={untest(setAuth)} />
        </Panel>
        <Panel title="Details" subtitle="The catalog name is what apps call it: /m/<name>/v1 through the gateway, like any other model.">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1.5 text-xs text-muted">
              Catalog name
              <input value={id} onChange={(e) => setId(e.target.value.toLowerCase())} placeholder="e.g. hosted-asr-whisper-1" className="min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg num" />
            </label>
            <label className="flex flex-col gap-1.5 text-xs text-muted">
              Display name
              <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder={id || "optional"} className="min-h-10 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg" />
            </label>
            <Select label="Derived from" value={parent} onChange={(e) => setParent(e.target.value)} className="!min-h-10 flex-1">
              <option value="">None</option>
              {models.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
            </Select>
          </div>
          <TagInput value={tags} onChange={setTags} placeholder="e.g. asr, hosted, en" />
          <label className="flex items-start gap-2.5 text-[13px] cursor-pointer border border-warn/40 bg-warn/5 rounded-md p-3">
            <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} className="mt-0.5 accent-primary" />
            <span className="flex flex-col">
              <span>Data sent to this model leaves this network</span>
              <span className="text-xs text-muted">Audio and text go to the provider. Only route data that is cleared for it.</span>
            </span>
          </label>
        </Panel>
      </div>
      <aside className="w-full xl:w-[320px] shrink-0 flex flex-col gap-4">
        <Panel title="Connection">
          {test.isPending && <Spinner label="Testing" />}
          {tested && <TestLines lines={tested.lines} />}
          {!tested && !test.isPending && <span className="text-[13px] text-muted">Not tested yet. A test sends one small request with the key; it does not run the check clip.</span>}
          <ErrorNote error={test.error} />
          <Button disabled={!canTest || test.isPending} onClick={() => test.mutate()}>Test connection</Button>
        </Panel>
        <Panel title="Verification" subtitle="Runs after adding, daily, and on demand">
          <span className="text-[13px] text-muted leading-relaxed">{c.verify}</span>
          <span className="text-[11px] text-subtle">No lease or GPU. A hosted model has no image, so its build key is the URL, model name and contract. The provider can change the model behind the same name, so it is re-verified every day.</span>
        </Panel>
        <ErrorNote error={add.error} />
        <Button variant="primary" disabled={!!hint || add.isPending} onClick={() => add.mutate()}>Add model and verify</Button>
        {hint && <span className="text-xs text-muted text-center">{hint}</span>}
      </aside>
    </div>
  );
}
