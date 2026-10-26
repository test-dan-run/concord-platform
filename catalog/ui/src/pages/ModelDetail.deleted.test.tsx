// A model page open while the model is removed elsewhere (a hosted endpoint unregistered in another tab, a
// service file deleted): the "model" event refetches it, the catalog answers 404, and the page says so instead of
// showing the old record with its Start, Verify and Unregister buttons.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";
import type { HostedInfo, Model } from "../lib/api";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const HOSTED: HostedInfo = {
  base_url: "https://api.example.com/v1",
  model: "whisper-1",
  host: "api.example.com",
  auth_type: "bearer",
  header_name: null,
  key_hint: "abcd",
  timeout_s: 60,
  max_concurrency: 4,
  max_upload_mb: 25,
  resolved_url: "https://api.example.com/v1/audio/transcriptions",
  data_ack_at: "2026-10-24T00:00:00Z",
  created_at: "2026-10-24T00:00:00Z",
  next_check_at: null,
  last_test: null,
};

function model(p: Partial<Model> = {}): Model {
  return {
    id: "whisper",
    aliases: [],
    registered: true,
    display_name: "Whisper",
    tags: [],
    parent_id: null,
    registered_at: "2026-10-24T00:00:00Z",
    description: "Speech to text",
    tasks: ["asr"],
    kind: "asr",
    engine: "vllm",
    contract: "openai",
    api: "audio.transcriptions",
    signature: null,
    gpu: null,
    state: "stopped",
    state_detail: null,
    leased_by: [],
    in_flight: 0,
    build_key: null,
    build_pending: null,
    image: null,
    capabilities: null,
    cold_start_s: 30,
    cold_start_measured: false,
    endpoint: { gateway_url: "http://catalog/m/whisper", base_path: "/v1" },
    last_used: null,
    calls: 0,
    served_name: "whisper",
    hosted: null,
    ...p,
  };
}

const json = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status, headers: { "Content-Type": "application/json" } });

/** Answer fetch from `routes` ("METHOD /api/path"); every call is recorded. */
function mockApi(routes: Record<string, () => unknown>) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://catalog");
      const key = `${init.method ?? "GET"} ${url.pathname}`;
      calls.push(key);
      const route = routes[key] ?? (key === "GET /api/models" ? () => [] : undefined);
      if (!route) return json({ error: { message: `no route ${key}` } }, 404);
      const out = route();
      return out instanceof Response ? out : json(out);
    }),
  );
  return calls;
}

function open(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <RouterProvider router={createMemoryRouter(routes, { initialEntries: [path] })} />
    </QueryClientProvider>,
  );
  return qc;
}

/** What the "model" event does (lib/events.ts), then wait for the page's model to be fetched again. */
async function refetch(qc: QueryClient, calls: string[], key: string) {
  await act(async () => void (await qc.invalidateQueries({ queryKey: ["model"] })));
  await waitFor(() => expect(calls.filter((c) => c === key).length).toBe(2));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("a model removed while its page is open", () => {
  it("a hosted model's page says it is gone when the refetch answers 404", async () => {
    let gone = false;
    const calls = mockApi({
      "GET /api/models/hosted-asr": () =>
        gone ? json({ error: { message: "unknown service 'hosted-asr'; known: whisper", type: "not_found", code: 404 } }, 404) : model({ id: "hosted-asr", display_name: "Hosted ASR", hosted: HOSTED }),
    });
    const qc = open("/models/hosted-asr");
    expect(await screen.findByRole("heading", { name: /Hosted ASR/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Unregister" })).toBeTruthy();
    gone = true;
    await refetch(qc, calls, "GET /api/models/hosted-asr");
    expect(await screen.findByText(/unknown service 'hosted-asr'/)).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /Hosted ASR/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Unregister" })).toBeNull();
  });

  it("a platform model's page says it is gone when the refetch answers 404", async () => {
    let gone = false;
    const calls = mockApi({
      "GET /api/models/whisper": () => (gone ? json({ error: { message: "unknown service 'whisper'; known: ", type: "not_found", code: 404 } }, 404) : model()),
    });
    const qc = open("/models/whisper");
    expect(await screen.findByRole("heading", { name: "Whisper" })).toBeTruthy();
    gone = true;
    await refetch(qc, calls, "GET /api/models/whisper");
    expect(await screen.findByText(/unknown service 'whisper'/)).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Whisper" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Start" })).toBeNull();
  });

  it("keeps showing the model when a refetch fails for another reason", async () => {
    let down = false;
    const calls = mockApi({
      "GET /api/models/whisper": () => (down ? json({ error: { message: "the catalog is busy" } }, 503) : model()),
    });
    const qc = open("/models/whisper");
    expect(await screen.findByRole("heading", { name: "Whisper" })).toBeTruthy();
    down = true;
    await refetch(qc, calls, "GET /api/models/whisper");
    expect(screen.getByRole("heading", { name: "Whisper" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Start" })).toBeTruthy();
  });
});
