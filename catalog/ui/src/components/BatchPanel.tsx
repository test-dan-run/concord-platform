// Batching on a model's page: the recommended batch for each item length against 16, 24 and 32 GB and the full
// budget, from the build's measured batch profile (catalog/batching.py).

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type BatchView } from "../lib/api";
import { ago } from "../lib/format";
import { Button, ErrorNote, Panel, Spinner } from "./ui";

function formula(v: BatchView): string {
  const p = v.profile!;
  if (p.kind === "kv")
    return `vLLM KV cache: ${p.kv_tokens?.toLocaleString("en-US")} tokens in ${p.kv_gib} GiB of the ${p.given_gib} GiB it was given`;
  return `peak GiB ≈ ${p.base.toFixed(2)} + ${p.a.toFixed(4)} × N × L + ${p.b.toFixed(6)} × N × L² (L in seconds, ${p.points.length} measured batches)`;
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
      {v?.measuring && <Spinner label={v.measuring === "running" ? "Measuring: leasing the model, then sending larger and larger batches until memory runs out" : "Queued for measurement"} />}
      {v?.status === "failed" && v.error && <span className="text-[13px] text-warn">{v.error}</span>}
      {v?.profile && v.table && (
        <>
          <span className="text-[13px] text-muted">{formula(v)}</span>
          <table className="w-full text-[13px] num">
            <caption className="sr-only">Largest batch that fits each memory size</caption>
            <thead>
              <tr className="text-muted text-xs">
                <th scope="col" className="text-left font-normal py-1">Item length</th>
                {v.table.budgets_gb.map((b) => (
                  <th key={b} scope="col" className="text-right font-normal py-1">{b === v.table!.full_gb ? `${b} GB (full)` : `${b} GB`}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {v.table.rows.map((r) => (
                <tr key={r.length} className="border-t border-track">
                  <th scope="row" className="text-left font-normal py-1">{v.table!.unit === "s" ? `${r.length} s` : `${r.length.toLocaleString("en-US")} tokens`}</th>
                  {r.batch.map((n, i) => <td key={i} className="text-right py-1">{n === 0 ? "—" : n}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
          <span className="text-xs text-muted">
            {v.profile.kind === "kv"
              ? "Sequences that fit the KV cache at each memory size: a value for --max-num-seqs. — means the weights alone do not fit."
              : "The largest batch whose predicted peak stays under 90% of the memory. — means not even one item fits. The service starts with these limits for the memory it is given."}
          </span>
        </>
      )}
    </Panel>
  );
}
