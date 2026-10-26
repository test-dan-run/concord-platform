// Import package while an import runs: a zip dropped on the page, or a change of "Import as", must not hide the
// running import or let a second one start beside it.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { routes } from "../app/routes";

vi.mock("../lib/events", () => ({ useCatalogEvents: () => false }));

const STATS = { root: "/s", audio: { files: 0, bytes: 0 }, in_versions: { files: 0, bytes: 0 }, held_only: { files: 0, bytes: 0 }, neither: { files: 0, bytes: 0 }, partial: { files: 0, bytes: 0 }, holds: [], datasets: 0, versions: 0, disk: { total: 10, free: 5 } };
const PREVIEW = { ok: true, problems: [], manifest: { name: "pkg", label_types: ["asr"], counts: { files: 1 } }, files: 1, missing: [], same_as_latest: false };

// A fake XMLHttpRequest for postForm: each send() waits until the test answers it.
const sent: { x: FakeXHR; preview: boolean }[] = [];
class FakeXHR {
  upload: { onprogress?: (e: { lengthComputable: boolean; loaded: number; total: number }) => void } = {};
  status = 0;
  statusText = "";
  responseText = "";
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  open() {}
  send(fd: FormData) {
    sent.push({ x: this, preview: fd.get("preview") === "true" });
  }
  answer(status: number, body: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

beforeEach(() => {
  sent.length = 0;
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input), "http://catalog").pathname;
      const out = path === "/api/store/stats" ? STATS : path === "/api/apps" || path === "/api/datasets" ? [] : null;
      return new Response(JSON.stringify(out ?? { error: { message: `no route ${path}` } }), { status: out ? 200 : 404, headers: { "Content-Type": "application/json" } });
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/** Open Import package, drop a.zip, answer its check, and start the import. */
async function importing() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <RouterProvider router={createMemoryRouter(routes, { initialEntries: ["/datasets/import"] })} />
    </QueryClientProvider>,
  );
  const zone = (await screen.findByText("Drop a reference package zip here")).parentElement!;
  fireEvent.drop(zone, { dataTransfer: { files: [new File(["a"], "a.zip")] } });
  await waitFor(() => expect(sent.length).toBe(1));
  await act(async () => sent[0].x.answer(200, PREVIEW));
  const button = (await screen.findByRole("button", { name: "Import as a new dataset" })) as HTMLButtonElement;
  await waitFor(() => expect(button.disabled).toBe(false));
  fireEvent.click(button);
  await waitFor(() => expect(sent.length).toBe(2));
  expect(screen.getByText("Importing")).toBeTruthy();
  return zone;
}

describe("Import package while an import runs", () => {
  it("ignores a zip dropped on the page", async () => {
    const zone = await importing();
    fireEvent.drop(zone, { dataTransfer: { files: [new File(["b"], "b.zip")] } });
    await act(async () => {});
    expect(sent.length).toBe(2); // b.zip is not checked
    expect(screen.getByText("a.zip")).toBeTruthy();
    expect(screen.getByText("Importing")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Import as a new dataset" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps showing the import when Import as changes", async () => {
    await importing();
    fireEvent.click(screen.getByRole("button", { name: "Next version" }));
    await act(async () => {});
    expect(screen.getByText("Importing")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Import as the next version" }) as HTMLButtonElement).disabled).toBe(true);
    expect(sent.filter((s) => !s.preview).length).toBe(1);
  });
});
