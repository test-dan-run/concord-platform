// The Dataset Catalog's pages against a mocked API: filters reach the server, deleting a version lists the apps'
// datasets made from it, and Upload audio makes a dataset, removing it again when the version can't be made.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";
import { labelBase, labelFileUrl, labelStats, type Dataset, type Version, type VersionFile } from "../lib/datasets";
import { uploads } from "../lib/upload";
import { compareSummary } from "./DatasetDetail";
import { suggestName } from "./UploadPage";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const STATS = { root: "/s", audio: { files: 0, bytes: 0 }, in_versions: { files: 0, bytes: 0 }, held_only: { files: 0, bytes: 0 }, neither: { files: 0, bytes: 0 }, partial: { files: 0, bytes: 0 }, holds: [], datasets: 0, versions: 0, disk: { total: 10, free: 5 } };

function version(p: Partial<Version>): Version {
  return { dataset_id: "d1", n: 1, parent_n: null, parent_deleted: false, digest: "x", format_version: 2, label_types: ["asr"], labels: [], counts: { files: 2, segments: 4, hours: 0.5 }, published_by: "label", published_at: "2026-10-24T00:00:00Z", uses: [], manifest: {}, ...p };
}

function dataset(p: Partial<Dataset>): Dataset {
  return { id: "d1", name: "Calls", language: "en", description: null, tags: [], created_by: "label", created_at: "2026-10-24T00:00:00Z", updated_at: "2026-10-24T00:00:00Z", latest: version({}), ...p };
}

type Route = (url: URL, init: RequestInit) => unknown;

/** Answer fetch from `routes` ("METHOD /api/path"); every call is recorded. */
function mockApi(routes: Record<string, Route>) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://catalog");
      const key = `${init.method ?? "GET"} ${url.pathname}`;
      calls.push(`${key}${url.search}`);
      const route = routes[key] ?? (key === "GET /api/store/stats" ? () => STATS : key === "GET /api/apps" ? () => [] : undefined);
      if (!route) return new Response(JSON.stringify({ error: { message: `no route ${key}` } }), { status: 404 });
      const out = await route(url, init);
      if (out instanceof Response) return out;
      return out === undefined ? new Response(null, { status: 204 }) : new Response(JSON.stringify(out), { status: 200, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}

function open(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  uploads.clear();
});

describe("Datasets page", () => {
  it("lists datasets and sends the filters to the catalog", async () => {
    const calls = mockApi({
      "GET /api/datasets": (u) =>
        u.searchParams.get("language") === "ms"
          ? [dataset({ id: "d2", name: "Malay calls", language: "ms" })]
          : [dataset({}), dataset({ id: "d2", name: "Malay calls", language: "ms" }), dataset({ id: "d3", name: "Raw audio", latest: version({ label_types: [] }) })],
    });
    open("/datasets");
    expect(await screen.findByRole("link", { name: "Calls" })).toBeTruthy();
    expect(screen.getByText("3 datasets")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Language"), { target: { value: "ms" } });
    await waitFor(() => expect(calls).toContain("GET /api/datasets?language=ms"));
    await waitFor(() => expect(screen.queryByRole("link", { name: "Calls" })).toBeNull());
    expect(screen.getByText("1 of 3 datasets")).toBeTruthy();
    // "Audio only" has no server filter: the page keeps datasets whose latest version has no references.
    fireEvent.change(screen.getByLabelText("Language"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Labels"), { target: { value: "none" } });
    await waitFor(() => expect(screen.getByText("1 of 3 datasets")).toBeTruthy());
    expect(screen.getByRole("link", { name: "Raw audio" })).toBeTruthy();
  });
});

describe("Dataset page", () => {
  it("lists the apps' datasets made from a version before deleting it", async () => {
    const v2 = version({ n: 2, parent_n: 1, uses: [{ app: "bench", ref: "15", created_at: "2026-10-24T00:00:00Z" }] });
    const calls = mockApi({
      "GET /api/datasets/d1": () => dataset({ latest: v2, versions: [v2, version({})] }),
      "GET /api/datasets": () => [],
      "GET /api/datasets/d1/compare": () => ({ a: 1, b: 2, files: { added: ["b.wav"], removed: [], same: 1 }, changed: [], segments_compared: true, labels: { added: [], removed: [] } }),
      "DELETE /api/datasets/d1/versions/2": () => undefined,
    });
    open("/datasets/d1");
    expect(await screen.findByText(/1 file added/)).toBeTruthy();
    fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]);
    const dialog = screen.getByRole("alertdialog", { name: "Delete v2?" });
    expect(dialog.textContent).toContain("Concord Bench dataset 15");
    expect(calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Delete v2" }));
    await waitFor(() => expect(calls).toContain("DELETE /api/datasets/d1/versions/2"));
  });
});

describe("Upload audio page", () => {
  const sha = "b".repeat(64);
  const record = { sha256: sha, size: 4, ext: ".wav", duration: 2, sample_rate: 16000, channels: 1, state: "ready", created_at: "" };

  it("makes a new dataset, and removes it again when the version can't be made", async () => {
    let fail = true;
    const calls = mockApi({
      "GET /api/datasets": () => [],
      "POST /api/store/missing": () => ({ missing: [] }), // the store has the file already
      [`POST /api/store/audio/${"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}/commit`]: () => ({ ...record, sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" }),
      "POST /api/datasets": () => dataset({ id: "new", latest: null }),
      "POST /api/datasets/new/versions/audio": () =>
        fail ? new Response(JSON.stringify({ error: { message: "ffprobe can't read 1 files", type: "unreadable_audio", unreadable: ["calls/a.wav"] } }), { status: 422 }) : version({ dataset_id: "new", n: 1, label_types: [] }),
      "DELETE /api/datasets/new": () => undefined,
      "GET /api/datasets/new/versions/1": () => version({ dataset_id: "new", n: 1, label_types: [] }),
      "GET /api/datasets/new": () => dataset({ id: "new" }),
      "GET /api/datasets/new/versions/1/files": () => ({ total: 0, files: [] }),
    });
    const router = open("/datasets/upload");
    const file = new File(["test"], "a.wav");
    Object.defineProperty(file, "webkitRelativePath", { value: "calls/a.wav" });
    fireEvent.change(screen.getByLabelText("Choose files"), { target: { files: [file] } });
    expect(await screen.findByText(/Already in the store/)).toBeTruthy();
    expect((screen.getByLabelText("Name") as HTMLInputElement).value).toBe("calls");
    expect(screen.getByText(/Version 1 with 1 file/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Make dataset" }));
    expect(await screen.findByText(/ffprobe can't read 1 files/)).toBeTruthy();
    await waitFor(() => expect(calls).toContain("DELETE /api/datasets/new"));
    fail = false;
    fireEvent.click(screen.getByRole("button", { name: "Make dataset" }));
    await waitFor(() => expect(router.state.location.pathname).toBe("/datasets/new/v/1"));
    const made = (vi.mocked(fetch).mock.calls.find(([u, i]) => String(u).endsWith("/versions/audio") && i?.method === "POST")![1]!.body as string);
    expect(JSON.parse(made)).toEqual({ files: [{ path: "calls/a.wav", sha256: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" }], parent: null, app: "dataset-catalog" });
    expect(uploads.getSnapshot()).toEqual([]);
  });

  it("offers only datasets whose latest version has audio only", async () => {
    mockApi({ "GET /api/datasets": () => [dataset({ id: "refs", name: "Labelled" }), dataset({ id: "raw", name: "Raw", latest: version({ label_types: [] }) }), dataset({ id: "empty", name: "Empty", latest: null })] });
    open("/datasets/upload?dataset=refs");
    expect(await screen.findByText(/Labelled v1 has references/)).toBeTruthy();
    const options = [...(screen.getByLabelText("Dataset") as HTMLSelectElement).options].map((o) => o.textContent);
    expect(options).toEqual(["Pick one", "Labelled", "Raw (v1, 2 files)", "Empty (no versions)"]);
  });
});

describe("helpers", () => {
  it("suggests a dataset name from the folder or zip", () => {
    expect(suggestName([{ path: "calls/a.wav" }, { path: "calls/b.wav" }])).toBe("calls");
    expect(suggestName([{ path: "batch.zip" }])).toBe("batch");
    expect(suggestName([{ path: "a.wav" }, { path: "b.wav" }])).toBe("");
  });

  it("summarises a comparison", () => {
    expect(compareSummary({ a: 1, b: 2, files: { added: ["x"], removed: ["y", "z"], same: 0 }, changed: [], segments_compared: true, labels: { added: ["noise"], removed: [] } })).toBe(
      "1 file added, 2 files removed, 1 label added",
    );
  });

  it("links to Label only at an address a browser can reach", () => {
    expect(labelBase([{ name: "label", base_url: "http://127.0.0.1:8620/" }])).toBe("http://127.0.0.1:8620");
    expect(labelBase([{ name: "label", base_url: "http://localhost:8620" }])).toBe("http://localhost:8620");
    expect(labelBase([{ name: "label", base_url: "http://concord-label:8620" }])).toBeNull();
    expect(labelBase([{ name: "bench", base_url: "http://127.0.0.1:8610" }])).toBeNull();
    expect(labelFileUrl("http://h:1", "d1", "ab", 2.5)).toBe("http://h:1/label/file/by-sha/ab?catalog_dataset_id=d1&t=2.50");
  });

  it("counts file label values", () => {
    const f = (labels: Record<string, unknown>) => ({ labels }) as VersionFile;
    expect(labelStats([f({ c: "clean" }), f({ c: "noisy" }), f({ c: "clean", ch: "phone" })])).toEqual([
      ["c", [["clean", 2], ["noisy", 1]]],
      ["ch", [["phone", 1]]],
    ]);
  });
});
