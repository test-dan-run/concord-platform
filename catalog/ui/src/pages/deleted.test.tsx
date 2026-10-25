// A dataset page open while its version or dataset is deleted elsewhere: the "dataset" event refetches it, the
// catalog answers 404, and the page says it is gone instead of showing the old record and its download links.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";
import type { Dataset, Version } from "../lib/datasets";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const STATS = { root: "/s", audio: { files: 0, bytes: 0 }, in_versions: { files: 0, bytes: 0 }, held_only: { files: 0, bytes: 0 }, neither: { files: 0, bytes: 0 }, partial: { files: 0, bytes: 0 }, holds: [], datasets: 0, versions: 0, disk: { total: 10, free: 5 } };

function version(p: Partial<Version> = {}): Version {
  return { dataset_id: "d1", n: 1, parent_n: null, parent_deleted: false, digest: "x".repeat(64), format_version: 2, label_types: ["asr"], labels: [], counts: { files: 1, segments: 1, hours: 0.1 }, published_by: "label", published_at: "2026-10-24T00:00:00Z", uses: [], manifest: {}, ...p };
}

function dataset(p: Partial<Dataset> = {}): Dataset {
  return { id: "d1", name: "Calls", language: "en", description: null, tags: [], created_by: "label", created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:00Z", latest: version(), versions: [version()], ...p };
}

const json = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status, headers: { "Content-Type": "application/json" } });
const gone = (message: string) => json({ error: { message, type: "not_found", code: 404 } }, 404);

/** Answer fetch from `routes` ("METHOD /api/path"); every call is recorded. */
function mockApi(routes: Record<string, () => unknown>) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://catalog");
      const key = `${init.method ?? "GET"} ${url.pathname}`;
      calls.push(key);
      const route = routes[key] ?? (key === "GET /api/store/stats" ? () => STATS : key === "GET /api/apps" || key === "GET /api/datasets" ? () => [] : undefined);
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

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("a deleted dataset or version", () => {
  it("VersionPage says the version is gone when the refetch answers 404", async () => {
    let deleted = false;
    const calls = mockApi({
      "GET /api/datasets/d1": () => dataset(),
      "GET /api/datasets/d1/versions/1": () => (deleted ? gone("dataset d1 has no version 1") : version()),
      "GET /api/datasets/d1/versions/1/files": () => ({ total: 0, offset: 0, limit: 1000, files: [] }),
    });
    const qc = open("/datasets/d1/v/1");
    expect(await screen.findByRole("heading", { name: "Calls v1" })).toBeTruthy();
    expect(screen.getByRole("link", { name: /With audio/ })).toBeTruthy();
    deleted = true;
    // what the "dataset" event does (lib/events.ts)
    await act(async () => void (await qc.invalidateQueries({ queryKey: ["dataset"] })));
    await waitFor(() => expect(calls.filter((c) => c === "GET /api/datasets/d1/versions/1").length).toBe(2));
    expect(await screen.findByText(/has no version 1/)).toBeTruthy();
    expect(screen.getByText(/may have been deleted/)).toBeTruthy();
    expect(screen.queryByRole("link", { name: /With audio/ })).toBeNull();
  });

  it("VersionPage keeps showing the version when a refetch fails for another reason", async () => {
    let down = false;
    const calls = mockApi({
      "GET /api/datasets/d1": () => dataset(),
      "GET /api/datasets/d1/versions/1": () => (down ? json({ error: { message: "the catalog is busy" } }, 503) : version()),
      "GET /api/datasets/d1/versions/1/files": () => ({ total: 0, offset: 0, limit: 1000, files: [] }),
    });
    const qc = open("/datasets/d1/v/1");
    expect(await screen.findByRole("heading", { name: "Calls v1" })).toBeTruthy();
    down = true;
    await act(async () => void (await qc.invalidateQueries({ queryKey: ["dataset"] })));
    await waitFor(() => expect(calls.filter((c) => c === "GET /api/datasets/d1/versions/1").length).toBe(2));
    expect(screen.getByRole("heading", { name: "Calls v1" })).toBeTruthy();
    expect(screen.queryByText(/may have been deleted/)).toBeNull();
  });

  it("DatasetDetail says the dataset is gone when the refetch answers 404", async () => {
    let deleted = false;
    const calls = mockApi({ "GET /api/datasets/d1": () => (deleted ? gone("no dataset 'd1'") : dataset()) });
    const qc = open("/datasets/d1");
    expect(await screen.findByRole("heading", { name: "Calls" })).toBeTruthy();
    deleted = true;
    await act(async () => void (await qc.invalidateQueries({ queryKey: ["dataset"] })));
    await waitFor(() => expect(calls.filter((c) => c === "GET /api/datasets/d1").length).toBe(2));
    expect(await screen.findByRole("heading", { name: "Dataset not found" })).toBeTruthy();
    expect(screen.getByText("no dataset 'd1'")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Delete dataset" })).toBeNull();
  });
});
