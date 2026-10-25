// A hosted endpoint's page: endpoint, authentication, verification, Calling it, usage and registration.
// No State, Platform, Stop or Export controls: a hosted model has no container, image or GPU footprint.

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type ConnectionTest, type Model } from "../lib/api";
import { ago, rtf } from "../lib/format";
import { AuthFields, body, EndpointFields, TestLines, type Auth, type Endpoint } from "../components/hosted";
import { CapabilityChips, Report, VerifyChip } from "../components/model";
import { Button, Chip, ErrorNote, Fact, PageHeader, Panel, Spinner } from "../components/ui";
import { DetailTabs, useDetailTab } from "../components/DetailTabs";
import { ModelCard } from "../components/ModelCard";
import { Registration, usage } from "./ModelDetail";

function until(iso: string | null): string {
  if (!iso) return "";
  const h = (new Date(iso).getTime() - Date.now()) / 3_600_000;
  return h <= 0 ? "next daily check due now" : `next daily check in ${h < 1 ? `${Math.ceil(h * 60)} min` : `${Math.round(h)} h`}`;
}

export function HostedDetail({ m, models }: { m: Model; models: Model[] }) {
  const qc = useQueryClient();
  const nav = useNavigate();
  const h = m.hosted!;
  const c = m.capabilities;
  const done = () => qc.invalidateQueries();
  const [tested, setTested] = useState<ConnectionTest | null>(h.last_test);
  const test = useMutation({ mutationFn: () => api.post<ConnectionTest>(`/models/${m.id}/test`), onSuccess: setTested });
  const verify = useMutation({ mutationFn: () => api.post(`/models/${m.id}/verify`), onSuccess: done });
  const remove = useMutation({ mutationFn: () => api.del(`/hosted/${m.id}`), onSuccess: () => { void done(); nav("/"); } });
  const [ep, setEp] = useState<Endpoint>({ contract: `${m.contract}|${m.api}`, base_url: h.base_url, model: h.model, timeout_s: h.timeout_s, max_concurrency: h.max_concurrency, max_upload_mb: h.max_upload_mb });
  const [tab, setTab] = useDetailTab();
  const [auth, setAuth] = useState<Auth>({ auth_type: h.auth_type, header_name: h.header_name ?? "", key: "" });
  const saveEp = useMutation({
    mutationFn: () => {
      const b = body(ep, { auth_type: h.auth_type, header_name: h.header_name ?? "", key: "" });
      return api.patch<Model>(`/hosted/${m.id}`, { contract: b.contract, api: b.api, base_url: b.base_url, model: b.model, timeout_s: b.timeout_s, max_concurrency: b.max_concurrency, max_upload_mb: b.max_upload_mb });
    },
    onSuccess: done,
  });
  const saveAuth = useMutation({
    mutationFn: () => api.patch<Model>(`/hosted/${m.id}`, { auth_type: auth.auth_type, header_name: auth.auth_type === "header" ? auth.header_name.trim() : null, ...(auth.key.trim() ? { key: auth.key.trim() } : {}) }),
    onSuccess: () => {
      setAuth((a) => ({ ...a, key: "" }));
      void done();
    },
  });
  const endpointChanged = ep.contract !== `${m.contract}|${m.api}` || ep.base_url.trim().replace(/\/+$/, "") !== h.base_url || ep.model.trim() !== h.model;
  const authChanged = auth.auth_type !== h.auth_type || (auth.auth_type === "header" && auth.header_name.trim() !== (h.header_name ?? "")) || !!auth.key.trim();

  return (
    <>
      <PageHeader
        crumbs={<><Link to="/">Models</Link> / {m.id}</>}
        title={<span className="flex items-center gap-2.5">{m.display_name}<Chip tone="info">hosted</Chip></span>}
        subtitle={`${m.description}. Calls go through the gateway, which adds the stored key. Data sent to this model leaves this network.`}
        actions={
          <>
            <Button disabled={test.isPending} onClick={() => test.mutate()}>Test connection</Button>
            <Button variant="primary" onClick={() => verify.mutate()} disabled={["pending", "running"].includes(c?.status ?? "")}>Verify again</Button>
          </>
        }
      />
      <ErrorNote error={test.error ?? verify.error ?? remove.error} />
      {m.state_detail && <div role="alert" className="border border-bad/50 bg-bad/10 text-[13px] rounded-md px-3 py-2">{m.state_detail}</div>}
      <DetailTabs tab={tab} onChange={setTab} />
      {tab === "card" ? (
        <ModelCard id={m.id} />
      ) : (
      <div className="grid gap-5 xl:grid-cols-[1fr_340px] items-start">
        <div className="flex flex-col gap-5 min-w-0">
          <Panel
            title="Verification"
            subtitle={c?.build_key ? <>Build <span className="num">{c.build_key}</span> (URL + model name + contract){c.verified_at && <> · {ago(c.verified_at)}</>}{h.next_check_at && <> · {until(h.next_check_at)}</>}</> : "Not verified yet"}
            actions={<VerifyChip c={c} />}
          >
            {c?.status === "stale" && <span className="text-[13px] text-warn">This record is for an earlier endpoint. The current one ({m.build_key}) has not been verified yet.</span>}
            {["pending", "running"].includes(c?.status ?? "") && <Spinner label={c?.status === "running" ? "Verifying: probing the provider with the stored key" : "Queued for verification"} />}
            {c && c.report && <Report c={c} />}
            {c && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs text-muted">Detected:</span>
                <CapabilityChips c={c} />
              </div>
            )}
            <span className="text-[11px] text-subtle">Re-verified every day, because a provider can change the model behind the same name. A changed result shows here and on the Models page.</span>
          </Panel>
          <Panel title="Endpoint" subtitle="Changing the contract, base URL or model name makes a new build and verifies it again. Limits apply at once.">
            <EndpointFields value={ep} onChange={setEp} />
            <ErrorNote error={saveEp.error} />
            <span className="flex items-center gap-3">
              <Button variant="primary" disabled={saveEp.isPending} onClick={() => saveEp.mutate()}>Save endpoint</Button>
              {endpointChanged && <span className="text-xs text-warn">Saving makes a new build and verifies it.</span>}
            </span>
          </Panel>
          <Panel title="Calling it" subtitle="Through the catalog's gateway, like any platform model. Apps never see the key.">
            <Fact k="Gateway URL" v={<span className="text-[11px] break-all">{m.endpoint.gateway_url}{m.endpoint.base_path}</span>} />
            <pre className="m-0 text-[11px] num bg-canvas border border-track rounded-md p-2.5 whitespace-pre-wrap">{usage(m)}</pre>
          </Panel>
        </div>
        <div className="flex flex-col gap-5">
          <Panel title="Connection" subtitle={tested?.tested_at ? `Tested ${ago(tested.tested_at)} with the stored key` : "Not tested since the catalog started"}>
            {test.isPending && <Spinner label="Testing" />}
            {tested && <TestLines lines={tested.lines} />}
            <div>
              <Fact k="Calls" v={m.calls.toLocaleString("en-US")} />
              <Fact k="Errors" v={m.errors ? <span className="text-bad">{m.errors}</span> : 0} />
              <Fact k="In flight" v={`${m.in_flight} of ${h.max_concurrency}`} />
              <Fact k="Real-time factor" v={rtf(c?.rtf)} />
              <Fact k="Last used" v={ago(m.last_used)} />
            </div>
            <span className="text-[11px] text-subtle">No lease, queue or GPU. Calls past the concurrency limit wait in the gateway.</span>
          </Panel>
          <Panel title="Authentication" subtitle="The key is stored encrypted and never shown again.">
            <AuthFields value={auth} onChange={setAuth} keyLabel="Replace key" keyPlaceholder="Paste a new key" />
            {h.key_hint && <Fact k="Stored key" v={`•••• •••• ${h.key_hint}`} />}
            <ErrorNote error={saveAuth.error} />
            <span className="flex items-center gap-3">
              <Button disabled={!authChanged || saveAuth.isPending} onClick={() => saveAuth.mutate()}>{auth.key.trim() ? "Replace" : "Save"}</Button>
              {saveAuth.isPending && <Spinner label="Testing the new key" />}
            </span>
            <span className="text-[11px] text-subtle">A new key is tested before it replaces the old one. Replacing a key keeps the build and its verification.</span>
          </Panel>
          <Registration
            key={m.id}
            m={m}
            models={models}
            note="data leaving the network confirmed when added"
            onRemove={() => confirm(`Unregister ${m.id}? This removes the endpoint, its encrypted key and its registration.`) && remove.mutate()}
          />
        </div>
      </div>
      )}
    </>
  );
}
