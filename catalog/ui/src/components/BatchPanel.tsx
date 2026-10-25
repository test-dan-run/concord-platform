// Batching on a model's page: the batch limits the service starts with (measured, or set by hand), the largest
// batch for each input length against 16, 24 and 32 GB and the full budget, and how the build was measured
// (catalog/batching.py).

import { useState, type FormEvent, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type BatchLimits, type BatchView } from "../lib/api";
import { ago, num } from "../lib/format";
import { Button, Chip, ErrorNote, Fact, Panel, Spinner } from "./ui";

const SOURCE = { override: "set by you", measured: "measured", default: "image default" } as const;

function seconds(s: number): string {
  return `${num(s, Number.isInteger(s) ? 0 : 1)} s`;
}

/** "up to 24 inputs and 240 s of audio in one batch", with the image's default for a limit that is not set. */
function describe(l: BatchLimits): string {
  const items = l.max_batch === null ? "the image's default number of inputs" : `up to ${num(l.max_batch)} input${l.max_batch === 1 ? "" : "s"}`;
  const audio = l.max_len_s === null ? "the image's default amount of audio" : `${seconds(l.max_len_s)} of audio`;
  return `${items} and ${audio} in one batch`;
}

function same(a: BatchLimits, b: BatchLimits): boolean {
  return a.max_batch === b.max_batch && a.max_len_s === b.max_len_s;
}

function Heading({ children }: { children: ReactNode }) {
  return <h3 className="m-0 text-[13px] font-semibold">{children}</h3>;
}

function Note({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" }) {
  return <p className={`m-0 text-[13px] ${tone === "warn" ? "text-warn" : "text-muted"}`}>{children}</p>;
}

function Intro({ v }: { v: BatchView }) {
  if (v.profile?.kind === "kv" || !v.limits) // no limits: a vLLM service
    return (
      <Note>
        vLLM decides by itself how many requests to run together. It runs as many as fit in its KV cache, which is the part of GPU memory that
        holds the tokens of the requests in progress. The catalog does not set a batch limit for vLLM, so the table only shows how many requests fit.
      </Note>
    );
  return (
    <Note>
      The service runs several inputs on the GPU together as one batch, which is faster than running them one at a time. A larger batch needs more
      GPU memory, so the catalog measures how much memory this build uses, and it sets the batch limits each time it starts the service.
    </Note>
  );
}

const input = "min-h-9 w-44 bg-canvas border border-line rounded-md px-3 text-[13px] text-fg num";

function LimitsForm({ id, v, onDone }: { id: string; v: BatchView; onDone: () => void }) {
  const qc = useQueryClient();
  const l = v.limits!;
  const [items, setItems] = useState(v.override?.max_batch?.toString() ?? "");
  const [audio, setAudio] = useState(v.override?.max_len_s?.toString() ?? "");
  const done = () => {
    qc.invalidateQueries({ queryKey: ["batch", id] });
    onDone();
  };
  const save = useMutation({
    mutationFn: (b: BatchLimits) =>
      b.max_batch === null && b.max_len_s === null ? api.del(`/models/${id}/batch-override`) : api.put(`/models/${id}/batch-override`, b),
    onSuccess: done,
  });
  const reset = useMutation({ mutationFn: () => api.del(`/models/${id}/batch-override`), onSuccess: done });
  const submit = (e: FormEvent) => {
    e.preventDefault();
    save.mutate({ max_batch: items.trim() ? Number(items) : null, max_len_s: audio.trim() ? Number(audio) : null });
  };
  const hint = (x: number | null, unit = "") => (x === null ? "image default" : `measured ${num(x, Number.isInteger(x) ? 0 : 1)}${unit}`);
  return (
    <form onSubmit={submit} className="flex flex-col gap-3 border border-track rounded-md p-3">
      <div className="flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Most inputs in one batch
          <input type="number" min={1} max={4096} step={1} value={items} onChange={(e) => setItems(e.target.value)} placeholder={hint(l.measured.max_batch)} className={input} />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Most audio in one batch, in seconds
          <input type="number" min={0.1} max={6000} step="any" value={audio} onChange={(e) => setAudio(e.target.value)} placeholder={hint(l.measured.max_len_s, " s")} className={input} />
        </label>
      </div>
      <Note>
        Leave a box empty to use the measured value. Your limits stay in place when the image is rebuilt, and they apply the next time the service
        starts.
      </Note>
      <ErrorNote error={save.error ?? reset.error} />
      <div className="flex flex-wrap gap-2">
        <Button type="submit" variant="primary" size="sm" disabled={save.isPending}>Save</Button>
        <Button size="sm" onClick={onDone}>Cancel</Button>
        {v.override && <Button size="sm" variant="ghost" disabled={reset.isPending} onClick={() => reset.mutate()}>Go back to the measured limits</Button>}
      </div>
    </form>
  );
}

function Limits({ id, v }: { id: string; v: BatchView }) {
  const [editing, setEditing] = useState(false);
  const l = v.limits!;
  const r = v.running;
  const value = (key: keyof BatchLimits, x: number | null) => (
    <span className="inline-flex items-center gap-2">
      {x === null ? <span className="text-muted">Image default</span> : key === "max_len_s" ? seconds(x) : num(x)}
      {l.source[key] !== "default" && <Chip tone={l.source[key] === "override" ? "info" : "default"}>{SOURCE[l.source[key]]}</Chip>}
    </span>
  );
  const lowered = r?.batchers.filter((b) => b.lowered_len_s !== null) ?? [];
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Heading>Batch limits the service uses</Heading>
        {!editing && <Button size="sm" className="ml-auto" onClick={() => setEditing(true)}>Change limits</Button>}
      </div>
      <div>
        <Fact k="Most inputs in one batch" v={value("max_batch", l.max_batch)} />
        <Fact k="Most audio in one batch" v={value("max_len_s", l.max_len_s)} />
      </div>
      {editing && <LimitsForm id={id} v={v} onDone={() => setEditing(false)} />}
      <Note>
        The audio limit counts each input in a batch as long as the longest one, because the service pads shorter inputs to that length. With long
        inputs, the service reaches the audio limit first, so a batch holds fewer inputs.
      </Note>
      {l.source.max_batch === "default" && l.source.max_len_s === "default" ? (
        <Note>This build has not been measured and no limits are set, so the service uses the limits built into its image.</Note>
      ) : (
        <Note>The service gets these limits the next time it starts, with about {num(l.vram_gb, 1)} GB of GPU memory.</Note>
      )}
      {r && (r.started_with ? (
        <Note>
          The running service started with {describe(r.started_with)}.
          {!same(r.started_with, l) && " The limits above apply the next time it starts."}
        </Note>
      ) : (
        <Note>
          The running service was started outside the catalog, so the catalog did not set its limits.
          {r.batchers.length > 0 && ` It reports ${describe(r.batchers[0])}.`}
        </Note>
      ))}
      {lowered.map((b) => (
        <Note key={b.name} tone="warn">
          The service ran out of GPU memory {b.backoffs === 1 ? "once" : `${b.backoffs} times`} since it started, so it lowered its audio limit
          {r!.batchers.length > 1 ? ` for ${b.name}` : ""} to {seconds(b.lowered_len_s!)}.
        </Note>
      ))}
    </div>
  );
}

function Table({ v }: { v: BatchView }) {
  const t = v.table!;
  const kv = v.profile!.kind === "kv";
  return (
    <div className="flex flex-col gap-2">
      <Heading>{kv ? "Requests that fit in each amount of GPU memory" : "Largest batch for each amount of GPU memory"}</Heading>
      <Note>
        {kv
          ? "Each number is how many requests of that length fit in the KV cache with that much GPU memory. You can give it to vLLM as --max-num-seqs. A 0 means the model weights alone do not fit."
          : "Each number is the largest batch of inputs of that length that fits in that much GPU memory, with the predicted peak kept under 90% of the memory. A 0 means not even one input fits."}
        {t.full_gb !== null && " The last column is all the GPU memory the catalog lets one model use."}
      </Note>
      <table className="w-full text-[13px] num">
        <caption className="sr-only">{kv ? "Requests that fit in each amount of GPU memory" : "Largest batch that fits in each amount of GPU memory"}</caption>
        <thead>
          <tr className="text-muted text-xs">
            <th scope="col" className="text-left font-normal py-1">{kv ? "Request length" : "Input length"}</th>
            {t.budgets_gb.map((b) => (
              <th key={b} scope="col" className="text-right font-normal py-1">{b === t.full_gb ? `${b} GB (full budget)` : `${b} GB`}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {t.rows.map((r) => (
            <tr key={r.length} className="border-t border-track">
              <th scope="row" className="text-left font-normal py-1">{t.unit === "s" ? `${r.length} s` : `${r.length.toLocaleString("en-US")} tokens`}</th>
              {r.batch.map((n, i) => <td key={i} className="text-right py-1">{n}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HowMeasured({ v }: { v: BatchView }) {
  const p = v.profile!;
  if (p.kind === "kv")
    return (
      <Note>
        vLLM reported at startup that its KV cache holds {p.kv_tokens?.toLocaleString("en-US")} tokens in {p.kv_gib} GiB, out of the {p.given_gib} GiB of
        GPU memory it was given.
      </Note>
    );
  const lengths = [...new Set(p.points.map((x) => x.length_s))].sort((a, b) => a - b).map(seconds);
  return (
    <details className="text-[13px]">
      <summary className="cursor-pointer text-muted">How the catalog measured this build</summary>
      <div className="flex flex-col gap-2 mt-2">
        <Note>
          The catalog started the model and sent batches of 1, 2, 4 and more inputs of {lengths.join(", ")}. It recorded the peak GPU memory of{" "}
          {p.points.length} batches and fitted this formula to them, where N is the number of inputs and L is the length of each input in seconds.
        </Note>
        <pre className="m-0 text-[12px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">
          peak GiB ≈ {p.base.toFixed(2)} + {p.a.toFixed(4)} × N × L + {p.b.toFixed(6)} × N × L²
        </pre>
        {p.method && <Note>The memory readings came from {p.method}.</Note>}
        {p.oom.length > 0 && <Note>Why the catalog stopped sending larger batches at each length:</Note>}
        {p.oom.length > 0 && (
          <ul className="m-0 pl-5 text-muted">
            {p.oom.map((o) => <li key={o.length_s}>Inputs of {seconds(o.length_s)}: {o.stopped === "max batch" ? "reached the largest batch the catalog sends" : o.stopped}</li>)}
          </ul>
        )}
      </div>
    </details>
  );
}

export function BatchPanel({ id }: { id: string }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["batch", id],
    queryFn: () => api.get<BatchView>(`/models/${id}/batch`),
    refetchInterval: (s) => (s.state.data?.measuring ? 3000 : 30_000),
  });
  const measure = useMutation({ mutationFn: () => api.post(`/models/${id}/batch-profile`), onSuccess: () => qc.invalidateQueries({ queryKey: ["batch", id] }) });
  const v = q.data;
  const subtitle = !v ? undefined : v.status === "measured" ? <>Measured {ago(v.measured_at)} for build <span className="num">{v.build_key}</span></> : v.status === "failed" ? "The last measurement failed" : "Not measured for this build yet";
  return (
    <Panel
      title="Batching"
      subtitle={subtitle}
      actions={<Button size="sm" disabled={!!v?.measuring || measure.isPending} onClick={() => measure.mutate()}>{v?.status === "measured" ? "Measure again" : "Measure"}</Button>}
    >
      {q.isLoading && <Spinner />}
      <ErrorNote error={q.error ?? measure.error} />
      {v && <Intro v={v} />}
      {v?.measuring && <Spinner label={v.measuring === "running" ? "Measuring. The catalog starts the model and sends larger batches until the GPU runs out of memory." : "Waiting in the queue to measure"} />}
      {v?.status === "failed" && v.error && <Note tone="warn">{v.error}</Note>}
      {v?.limits && <Limits id={id} v={v} />}
      {v?.profile && v.table && (
        <>
          <Table v={v} />
          <HowMeasured v={v} />
        </>
      )}
    </Panel>
  );
}
