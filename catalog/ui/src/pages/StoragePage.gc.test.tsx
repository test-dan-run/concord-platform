// Storage clean-up against a mocked API: "Delete N files" sends the dry run's N, so the catalog can refuse to
// delete more than the person saw, and the page then checks again.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const STATS = { root: "/s", audio: { files: 3, bytes: 30 }, in_versions: { files: 0, bytes: 0 }, held_only: { files: 0, bytes: 0 }, neither: { files: 3, bytes: 30 }, partial: { files: 0, bytes: 0 }, holds: [], datasets: 0, versions: 0, disk: { total: 10, free: 5 } };
const CHANGED = "clean-up found more files than the check showed: check again";

const audio = (i: number) => ({ sha256: String(i).repeat(64), size: 10, ext: ".wav", state: "stored", created_at: "2026-10-20T00:00:00Z" });
const result = (dry_run: boolean, n: number) => ({ dry_run, files: n, bytes: 10 * n, sample: Array.from({ length: n }, (_, i) => audio(i + 1)) });
const json = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status, headers: { "Content-Type": "application/json" } });

/** A store with `files()` deletable files; `strict` answers like a catalog that checks expect_files. */
function mockStore(files: () => number, strict: boolean) {
  const gc: Record<string, unknown>[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const path = new URL(String(input), "http://catalog").pathname;
      if (path === "/api/store/stats") return json(STATS);
      if (path === "/api/apps") return json([]);
      if (path !== "/api/store/gc") return json({ error: { message: `no route ${path}` } }, 404);
      const body = JSON.parse(String(init.body));
      gc.push(body);
      if (body.dry_run) return json(result(true, files()));
      if (strict && typeof body.expect_files === "number" && files() > body.expect_files)
        return json({ error: { message: CHANGED, type: "gc_changed", code: 409 } }, 409);
      return json(result(false, files()));
    }),
  );
  return gc;
}

function open() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <RouterProvider router={createMemoryRouter(routes, { initialEntries: ["/datasets/storage"] })} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Storage clean-up", () => {
  it("sends the dry run's file count with the delete", async () => {
    const gc = mockStore(() => 2, true);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check what clean-up would delete" }));
    fireEvent.click(await screen.findByRole("button", { name: "Delete 2 files" }));
    expect(await screen.findByText("Deleted 2 audio files, 20 B.")).toBeTruthy();
    expect(gc).toEqual([{ dry_run: true }, { dry_run: false, expect_files: 2 }]);
  });

  it("shows the catalog's refusal and checks again when more files would go", async () => {
    let n = 2;
    const gc = mockStore(() => n, true);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check what clean-up would delete" }));
    const del = await screen.findByRole("button", { name: "Delete 2 files" });
    n = 3; // a version was deleted after the dry run
    fireEvent.click(del);
    expect(await screen.findByText(CHANGED)).toBeTruthy();
    expect(await screen.findByRole("button", { name: "Delete 3 files" })).toBeTruthy();
    expect(gc).toEqual([{ dry_run: true }, { dry_run: false, expect_files: 2 }, { dry_run: true }]);
    expect(screen.queryByText(/^Deleted /)).toBeNull();
  });

  it("works as before with a catalog that ignores expect_files", async () => {
    let n = 2;
    mockStore(() => n, false);
    open();
    fireEvent.click(await screen.findByRole("button", { name: "Check what clean-up would delete" }));
    const del = await screen.findByRole("button", { name: "Delete 2 files" });
    n = 3;
    fireEvent.click(del);
    expect(await screen.findByText("Deleted 3 audio files, 30 B.")).toBeTruthy();
  });
});
