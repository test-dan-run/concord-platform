// The event stream reconnects (the catalog ends the stream of a client that fell behind, and EventSource opens
// it again): events sent while it was down are lost, so every query the events refresh is refreshed then.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
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

function connect() {
  vi.useFakeTimers();
  vi.stubGlobal("EventSource", FakeEventSource);
  const qc = new QueryClient();
  const keys: string[] = [];
  vi.spyOn(qc, "invalidateQueries").mockImplementation(async (f) => void keys.push(String(f?.queryKey?.[0])));
  const hook = renderHook(() => useCatalogEvents(), { wrapper: ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider> });
  const es = FakeEventSource.last;
  const refreshed = () => {
    vi.advanceTimersByTime(400);
    return keys.splice(0).sort();
  };
  return { es, hook, refreshed };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useCatalogEvents after a reconnect", () => {
  it("refreshes lease, export and dataset queries as well as the models", () => {
    const { es, hook, refreshed } = connect();
    act(() => es.onopen!());
    expect(hook.result.current).toBe(true);
    expect(refreshed()).toEqual([]); // the first open: the pages have just loaded
    act(() => es.onerror!());
    expect(hook.result.current).toBe(false);
    act(() => es.onopen!());
    expect(hook.result.current).toBe(true);
    expect(refreshed()).toEqual(["batch", "dataset", "datasets", "exports", "leases", "model", "models", "platform", "store-stats"]);
    act(() => es.onopen!()); // open again without a drop in between: nothing was missed
    expect(refreshed()).toEqual([]);
  });
});
