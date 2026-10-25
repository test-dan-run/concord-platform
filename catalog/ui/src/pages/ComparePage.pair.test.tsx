// Compare versions after "Show all": picking another pair of versions shows that pair's first 50 files again,
// even when its comparison is already cached and the page doesn't wait for it.

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

const DATASET: Dataset = { id: "d1", name: "Calls", language: "en", description: null, tags: [], created_by: "label", created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:00Z", latest: version(3), versions: [version(3), version(2), version(1)] };

const seg = (text: string) => ({ start: 0, end: 1, speaker: null, text, uid: null, labels: {} });

/** A comparison of v1 with v`b` that changed `count` files, named `v<b>-<i>.wav`. */
function comparison(b: number, count: number): Comparison {
  return {
    a: 1,
    b,
    files: { added: [], removed: [], same: 0 },
    changed: Array.from({ length: count }, (_, i) => ({ path: `v${b}-${i}.wav`, segments: { added: [seg(`new ${i}`)], removed: [], edited: [] } })),
    segments_compared: true,
    labels: { added: [], removed: [] },
  };
}

const COMPARE: Record<string, Comparison> = { "1-2": comparison(2, 120), "1-3": comparison(3, 80) };

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Compare versions", () => {
  it("caps the changed files again when another pair is picked", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = new URL(String(input), "http://catalog");
        const pair = `${url.searchParams.get("a")}-${url.searchParams.get("b")}`;
        const out = { "/api/datasets/d1": DATASET, "/api/datasets/d1/compare": COMPARE[pair], "/api/store/stats": STATS, "/api/apps": [] }[url.pathname];
        return new Response(JSON.stringify(out ?? { error: { message: `no route ${url.pathname}` } }), { status: out ? 200 : 404, headers: { "Content-Type": "application/json" } });
      }),
    );
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <RouterProvider router={createMemoryRouter(routes, { initialEntries: ["/datasets/d1/compare?a=1&b=2"] })} />
      </QueryClientProvider>,
    );
    const to = (n: number) => fireEvent.change(screen.getByLabelText("To"), { target: { value: String(n) } });
    const shown = (b: number) => screen.queryAllByText(new RegExp(`^v${b}-\\d+\\.wav$`)).length;

    expect(await screen.findByText("v2-0.wav")).toBeTruthy();
    to(3);
    expect(await screen.findByText("v3-0.wav")).toBeTruthy();
    expect(shown(3)).toBe(50);
    fireEvent.click(screen.getByRole("button", { name: "Show all 80" }));
    expect(shown(3)).toBe(80);

    // v1 to v2 was fetched first, so its comparison is cached and shows at once
    to(2);
    expect(await screen.findByText("v2-0.wav")).toBeTruthy();
    expect(shown(2)).toBe(50);
    expect(screen.getByRole("button", { name: "Show all 120" })).toBeTruthy();
  });
});
