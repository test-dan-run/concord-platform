// Add model's hosted form against a mocked API: a connection test counts only for the inputs it ran with, so an
// edit while the test is still out leaves the form untested.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ConnectionTest } from "../lib/api";
import { AddModel } from "./AddModel";

const PASSED: ConnectionTest = { ok: true, lines: [{ name: "reach", status: "pass", reason: "The endpoint answered" }], signature: null };

/** Answer GET /api/models with nothing and hold each POST /api/hosted/test until its release is called. */
function mockApi() {
  const tests: { body: Record<string, unknown>; release: () => void }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = new URL(String(input), "http://catalog");
      const key = `${init.method ?? "GET"} ${url.pathname}`;
      if (key === "GET /api/models") return new Response("[]", { status: 200 });
      if (key === "POST /api/hosted/test") {
        await new Promise<void>((release) => tests.push({ body: JSON.parse(String(init.body)), release }));
        return new Response(JSON.stringify(PASSED), { status: 200 });
      }
      return new Response(JSON.stringify({ error: { message: `no route ${key}` } }), { status: 404 });
    }),
  );
  return tests;
}

function open() {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={["/add?source=hosted"]}>
        <AddModel />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const type = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });
const testButton = () => screen.getByRole("button", { name: "Test connection" }) as HTMLButtonElement;

async function startTest() {
  const tests = mockApi();
  open();
  type("Base URL", "https://api.example.com/v1");
  type("Model name at the provider", "whisper-1");
  type("API key", "sk-test");
  fireEvent.click(testButton());
  await waitFor(() => expect(tests.length).toBe(1));
  return tests;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("AddModel hosted form", () => {
  it("accepts a connection test for the inputs it ran with", async () => {
    const tests = await startTest();
    tests[0].release();
    expect(await screen.findByText("The endpoint answered")).toBeTruthy();
    expect(screen.getByText("Enter the catalog name.")).toBeTruthy();
  });

  it("drops a connection test whose inputs changed while it ran", async () => {
    const tests = await startTest();
    expect(tests[0].body.base_url).toBe("https://api.example.com/v1");
    type("Base URL", "https://other.example.com/v1");
    tests[0].release();
    await waitFor(() => expect(testButton().disabled).toBe(false));
    expect(screen.queryByText("The endpoint answered")).toBeNull();
    expect(screen.getByText("Test the connection first.")).toBeTruthy();
  });
});
