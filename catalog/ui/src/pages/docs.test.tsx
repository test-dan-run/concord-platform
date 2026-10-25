// Docs and Model contract against the catalog: every call they show is a route catalog/app.py serves, the
// filters and paths are the ones that work, and the Docs search finds what each section shows.

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractPage } from "./ContractPage";
import { DocsPage, SECTIONS } from "./DocsPage";
import { ExportPage } from "./ExportPage";

vi.mock("@tanstack/react-query", async (orig) => ({
  ...(await orig<typeof import("@tanstack/react-query")>()),
  useQuery: () => ({ data: undefined, isLoading: false, error: null }),
  useMutation: () => ({ mutate: () => {}, isPending: false, error: null }),
  useQueryClient: () => ({ invalidateQueries: () => {} }),
}));

const GW = "http://gateway:8700";

/** Every text node under `el`, joined with spaces (textContent runs block elements together). */
function words(el: Element): string {
  const parts: string[] = [];
  const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  while (walk.nextNode()) parts.push(walk.currentNode.textContent ?? "");
  return parts.join(" ");
}

const tokens = (t: string) => new Set(t.toLowerCase().match(/[a-z0-9]+/g) ?? []);

function page(ui: React.ReactNode): string {
  const { container } = render(<MemoryRouter>{ui}</MemoryRouter>);
  return words(container);
}

/** The catalog's routes as (method, regex), from the decorators in catalog/app.py. */
function routes(): [string, RegExp][] {
  const src = readFileSync(`${import.meta.dirname}/../../../app.py`, "utf8");
  return [...src.matchAll(/@app\.(get|post|put|patch|delete)\("([^"]+)"/g)].map(([, m, path]) => [
    m.toUpperCase(),
    new RegExp(`^${path.replace(/\{[^}]+\}/g, "[^/]+")}$`),
  ]);
}

afterEach(cleanup);

describe("the calls the docs show", () => {
  it("are routes the catalog serves", () => {
    const served = routes();
    expect(served.length).toBeGreaterThan(30);
    const text = [page(<DocsPage />), page(<ContractPage />)].join(" ");
    const calls = [...text.matchAll(/\b(GET|POST|PUT|PATCH|DELETE) (\/api\/[^\s?"]+)/g)].map(([, m, p]) => [m, p.replace(/<[^>]+>/g, "x")]);
    expect(calls.length).toBeGreaterThan(3);
    for (const [m, p] of calls) expect(served.some(([sm, re]) => sm === m && re.test(p)), `${m} ${p}`).toBe(true);
  });

  it("use the task filter, the index lookup and the gateway path", () => {
    const docs = page(<DocsPage />);
    expect(docs).toContain("GET /api/models?task=asr");
    expect(docs).toContain("GET /api/models?task=diar");
    expect(docs).not.toMatch(/\?tag=(asr|diarisation)/);
    expect(docs).toContain("POST /api/index/lookup");
    expect(docs).toMatch(/\/api\/outputs\/batch\s+\{"ids"/);
    cleanup();
    expect(page(<ContractPage />)).toContain("POST /m/<model>/v1/text/itn");
  });

  it("say a registry export still writes the records and weights", () => {
    const docs = SECTIONS.find((s) => s.id === "export")!.text;
    expect(docs).not.toContain("holds only the manifest");
    expect(docs).toContain("service.yaml");
    expect(page(<ExportPage />)).not.toContain("holds only the manifest");
  });
});

describe("Docs search", () => {
  it("matches the words each section shows, and only those", () => {
    for (const s of SECTIONS) {
      const { container } = render(<MemoryRouter>{s.body(GW)}</MemoryRouter>);
      const shown = tokens(words(container));
      const searched = tokens(s.text.replaceAll("{gateway}", GW));
      expect([...shown].filter((w) => !searched.has(w)), `${s.id}: shown, not searchable`).toEqual([]);
      expect([...searched].filter((w) => !shown.has(w)), `${s.id}: searchable, not shown`).toEqual([]);
      cleanup();
    }
  });

  it("suggests only searches that find something", () => {
    render(<MemoryRouter><DocsPage /></MemoryRouter>);
    const box = screen.getByRole("searchbox", { name: "Search docs" });
    fireEvent.change(box, { target: { value: "zzzz" } });
    expect(screen.getByText(/Try /).textContent).not.toContain("model name");
    for (const q of ["/api/models", "lease", "tag", "build"]) {
      fireEvent.change(box, { target: { value: q } });
      expect(screen.getByRole("heading", { name: /sections? match/ }), q).toBeTruthy();
    }
  });
});
