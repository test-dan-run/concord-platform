// The Batching panel says "The limits above apply the next time it starts" only when the running service started
// with other limits than a start with its memory gets now (running.limits), not merely because the limits above
// are at the full budget.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { BatchView } from "../lib/api";
import { BatchPanel } from "./BatchPanel";

function view(running: BatchView["running"]): BatchView {
  return {
    model: "wavlm-sv", build_key: "bk1", measuring: null, status: "measured", error: null, measured_at: "2026-10-25T00:00:00Z",
    limits: { vram_gb: 29.76, max_batch: 128, max_len_s: 600, measured: { max_batch: 128, max_len_s: 600 }, source: { max_batch: "measured", max_len_s: "measured" } },
    override: null,
    running,
  };
}

function show(v: BatchView) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(v), { status: 200 })));
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

describe("BatchPanel running limits", () => {
  it("does not say the limits change when the service started with the limits for its memory", async () => {
    const limits = { vram_gb: 28.92, max_batch: 120, max_len_s: 560 };
    show(view({ vram_gb: 28.92, started_with: { max_batch: 120, max_len_s: 560 }, limits, batchers: [] }));
    expect(await screen.findByText(/started with up to 120 inputs and 560 s of audio in one batch\./)).toBeTruthy();
    expect(screen.queryByText(/apply the next time it starts/)).toBeNull();
  });

  it("says so when the limits for its memory changed after it started", async () => {
    const limits = { vram_gb: 28.92, max_batch: 6, max_len_s: 560 };
    show(view({ vram_gb: 28.92, started_with: { max_batch: 120, max_len_s: 560 }, limits, batchers: [] }));
    expect(await screen.findByText(/started with up to 120 inputs and 560 s of audio in one batch\. The limits above apply the next time it starts\./)).toBeTruthy();
  });
});
