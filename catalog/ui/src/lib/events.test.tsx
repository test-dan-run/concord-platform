// The catalog's server-sent events against a fake EventSource: each event refreshes the queries that show what it
// changed, a moment after it arrives.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCatalogEvents } from "./events";

class FakeEventSource {
  static last: FakeEventSource;
  handlers = new Map<string, () => void>();
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    FakeEventSource.last = this;
  }
  addEventListener(type: string, fn: () => void) {
    this.handlers.set(type, fn);
  }
  close() {}
}

/** The query keys refreshed after a `type` event. */
function refreshedBy(type: string): unknown[][] {
  vi.useFakeTimers();
  vi.stubGlobal("EventSource", FakeEventSource);
  const qc = new QueryClient();
  const keys: unknown[][] = [];
  vi.spyOn(qc, "invalidateQueries").mockImplementation(async (f) => void keys.push([...(f?.queryKey ?? [])]));
  renderHook(() => useCatalogEvents(), { wrapper: ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider> });
  FakeEventSource.last.handlers.get(type)!();
  vi.advanceTimersByTime(400);
  return keys;
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useCatalogEvents", () => {
  it("refreshes the Batching panel when a model starts or stops", () => {
    expect(refreshedBy("model")).toEqual([["models"], ["model"], ["platform"], ["batch"]]);
  });
});
