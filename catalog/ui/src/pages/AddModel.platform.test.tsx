// Add model's platform list against a mocked API: GET /models?all=1 also returns hosted endpoints, and those are
// not platform services.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HostedInfo, Model } from "../lib/api";
import { AddModel } from "./AddModel";

function model(p: Partial<Model>): Model {
  return {
    id: "m", aliases: [], registered: false, display_name: "m", tags: [], parent_id: null, registered_at: null, description: null, tasks: [], kind: "model",
    engine: "vllm", contract: "openai", api: "chat.completions", signature: null, gpu: { fraction: 0.5, gb: 16 }, state: "stopped", state_detail: null, leased_by: [],
    in_flight: 0, build_key: null, build_pending: null, image: null, capabilities: null, cold_start_s: 30, cold_start_measured: false,
    endpoint: { gateway_url: "", base_path: "" }, last_used: null, calls: 0, served_name: "m", hosted: null, ...p,
  };
}

const HOSTED = { base_url: "https://api.example.com/v1", model: "whisper-1", host: "api.example.com", auth_type: "bearer", header_name: null, key_hint: "abcd", timeout_s: 120, max_concurrency: 4, max_upload_mb: 25, resolved_url: "", data_ack_at: "", created_at: "", next_check_at: null, last_test: null } satisfies HostedInfo;

function mockModels(models: Model[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = new URL(String(input), "http://catalog");
      if (url.pathname === "/api/models") return new Response(JSON.stringify(models), { status: 200 });
      return new Response(JSON.stringify({ error: { message: `no route ${url.pathname}` } }), { status: 404 });
    }),
  );
}

function open(path: string) {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[path]}>
        <AddModel />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("AddModel platform services", () => {
  const platform = model({ id: "qwen3-8b", display_name: "qwen3-8b" });
  const hosted = model({ id: "hosted-asr", display_name: "hosted-asr", registered: true, contract: "openai", api: "audio.transcriptions", gpu: null, state: "hosted", hosted: HOSTED });

  it("lists platform services only, not hosted endpoints", async () => {
    mockModels([hosted, platform]);
    open("/add");
    expect(await screen.findByRole("button", { name: /qwen3-8b/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /hosted-asr/ })).toBeNull();
    expect(screen.queryByText("added")).toBeNull();
  });

  it("does not open a hosted endpoint as a platform service", async () => {
    mockModels([hosted, platform]);
    open("/add?name=hosted-asr");
    expect(await screen.findByRole("button", { name: /qwen3-8b/ })).toBeTruthy();
    expect(screen.getByText(/Pick a platform service/)).toBeTruthy();
  });
});
