// Compare versions with many changed files: the page renders the first 50, like the added and removed lists,
// and "Show all" renders the rest.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";
import type { Comparison, Dataset, Version } from "../lib/datasets";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const STATS = { root: "/s", audio: { files: 0, bytes: 0 }, in_versions: { files: 0, bytes: 0 }, held_only: { files: 0, bytes: 0 }, neither: { files: 0, bytes: 0 }, partial: { files: 0, bytes: 0 }, holds: [], datasets: 0, versions: 0, disk: { total: 10, free: 5 } };

function version(n: number): Version {
  return { dataset_id: "d1", n, parent_n: n > 1 ? n - 1 : null, parent_deleted: false, digest: "x", format_version: 2, label_types: ["asr"], labels: [], counts: { files: 1, segments: 1, hours: 0.1 }, published_by: "label", published_at: "2026-10-24T00:00:00Z", uses: [], manifest: {} };
}

const DATASET: Dataset = { id: "d1", name: "Calls", language: "en", description: null, tags: [], created_by: "label", created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:00Z", latest: version(2), versions: [version(2), version(1)] };

const seg = (text: string) => ({ start: 0, end: 1, speaker: null, text, uid: null, labels: {} });
const COMPARISON: Comparison = {
  a: 1,
  b: 2,
  files: { added: [], removed: [], same: 0 },
  changed: Array.from({ length: 120 }, (_, i) => ({ path: `call-${i}.wav`, segments: { added: [seg(`new ${i}`)], removed: [], edited: [] } })),
  segments_compared: true,
  labels: { added: [], removed: [] },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Compare versions", () => {
  it("renders 50 changed files until Show all is pressed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = new URL(String(input), "http://catalog").pathname;
        const out = { "/api/datasets/d1": DATASET, "/api/datasets/d1/compare": COMPARISON, "/api/store/stats": STATS, "/api/apps": [] }[path];
        return new Response(JSON.stringify(out ?? { error: { message: `no route ${path}` } }), { status: out ? 200 : 404, headers: { "Content-Type": "application/json" } });
      }),
    );
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <RouterProvider router={createMemoryRouter(routes, { initialEntries: ["/datasets/d1/compare?a=1&b=2"] })} />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("call-0.wav")).toBeTruthy();
    expect(screen.getAllByText(/^call-\d+\.wav$/).length).toBe(50);
    expect(screen.getByText("120 files")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show all 120" }));
    expect(screen.getAllByText(/^call-\d+\.wav$/).length).toBe(120);
    expect(screen.queryByRole("button", { name: /Show all/ })).toBeNull();
  });
});
