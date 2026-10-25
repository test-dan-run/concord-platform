// The Batching panel against a mocked API: it shows the limits the service starts and runs with, and Change limits
// saves an override, or removes it when both boxes are left empty.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BatchView } from "../lib/api";
import { BatchPanel } from "./BatchPanel";

function view(p: Partial<BatchView>): BatchView {
  return {
    model: "wavlm-sv", build_key: "bk1", measuring: null, status: "measured", error: null, measured_at: "2026-10-25T00:00:00Z",
    profile: { kind: "audio", base: 1.2, a: 0.004, b: 0.0003, max_item_s: 60, points: [{ n: 1, length_s: 2, peak_gib: 1.3, seconds: 0.1 }], oom: [{ length_s: 2, stopped: "max batch" }], method: "torch", kv_tokens: null, kv_gib: null, fixed_gib: null, max_model_len: null, max_concurrency: null, given_gib: 29 },
    table: { unit: "s", budgets_gb: [16, 24, 32, 29.76], full_gb: 29.76, rows: [{ length: 5, batch: [40, 64, 90, 80] }] },
    limits: { vram_gb: 29.76, max_batch: 128, max_len_s: 600, measured: { max_batch: 128, max_len_s: 600 }, source: { max_batch: "measured", max_len_s: "measured" } },
    override: null,
    running: null,
    ...p,
  };
}

function mockApi(v: BatchView) {
  const calls: { key: string; body?: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://catalog");
      const key = `${init.method ?? "GET"} ${url.pathname}`;
      calls.push({ key, body: init.body ? JSON.parse(String(init.body)) : undefined });
      if (key === "GET /api/models/wavlm-sv/batch") return new Response(JSON.stringify(v), { status: 200 });
      if (key === "PUT /api/models/wavlm-sv/batch-override") return new Response(JSON.stringify({ ...JSON.parse(String(init.body)), set_at: "now" }), { status: 200 });
      if (key === "DELETE /api/models/wavlm-sv/batch-override") return new Response(null, { status: 204 });
      return new Response(JSON.stringify({ error: { message: `no route ${key}` } }), { status: 404 });
    }),
  );
  return calls;
}

function show() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BatchPanel id="wavlm-sv" />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("BatchPanel", () => {
  it("shows the next start's limits and what the running service started with", async () => {
    mockApi(view({
      override: { max_batch: 16, max_len_s: null, set_at: "now" },
      limits: { vram_gb: 29.76, max_batch: 16, max_len_s: 600, measured: { max_batch: 128, max_len_s: 600 }, source: { max_batch: "override", max_len_s: "measured" } },
      running: { vram_gb: 29, started_with: { max_batch: 128, max_len_s: 600 }, batchers: [{ name: "embed", max_batch: 128, max_len_s: 600, backoffs: 2, lowered_len_s: 300 }] },
    }));
    show();
    expect(await screen.findByText("set by you")).toBeTruthy();
    expect(screen.getByText("measured")).toBeTruthy();
    expect(screen.getByText(/started with up to 128 inputs and 600 s of audio in one batch\. The limits above apply the next time it starts\./)).toBeTruthy();
    expect(screen.getByText(/ran out of GPU memory 2 times since it started, so it lowered its audio limit to 300 s/)).toBeTruthy();
  });

  it("saves an override, and removes it when both boxes are empty", async () => {
    const calls = mockApi(view({}));
    show();
    fireEvent.click(await screen.findByRole("button", { name: "Change limits" }));
    expect(screen.getByPlaceholderText("measured 128")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Most inputs in one batch"), { target: { value: "32" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(calls.some((c) => c.key === "PUT /api/models/wavlm-sv/batch-override")).toBe(true));
    expect(calls.find((c) => c.key.startsWith("PUT"))!.body).toEqual({ max_batch: 32, max_len_s: null });

    fireEvent.click(await screen.findByRole("button", { name: "Change limits" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(calls.some((c) => c.key === "DELETE /api/models/wavlm-sv/batch-override")).toBe(true));
  });
});
