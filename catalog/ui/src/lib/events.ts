// Live updates: the catalog's server-sent events invalidate the cached queries they affect, so pages
// refresh within a second of a model, lease or dataset changing instead of polling hard.

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

/** The query families each event refreshes. */
const EVENTS: Record<string, string[]> = {
  // A model started or stopped: the Batching panel shows the limits the running service started with.
  model: ["models", "model", "platform", "batch"],
  lease: ["leases", "models", "model"],
  queue: ["leases", "platform"],
  export: ["exports"],
  // A dataset was made, edited, published to or deleted. Version files, references and comparisons never
  // change, so their queries ("version-files", "refs", "compare") aren't touched.
  dataset: ["datasets", "dataset", "store-stats"],
};

export function useCatalogEvents(): boolean {
  const qc = useQueryClient();
  const [live, setLive] = useState(false);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const touched = new Set<string>();
    const flush = () => {
      for (const k of touched) void qc.invalidateQueries({ queryKey: [k] });
      touched.clear();
      timer = null;
    };
    const touch = (...keys: string[]) => {
      keys.forEach((k) => touched.add(k));
      timer ??= setTimeout(flush, 400);
    };
    const es = new EventSource("/api/events");
    // Events sent while the stream was down are lost, and a new stream starts with a model event per model only,
    // so after a reconnect every query the events refresh is refreshed.
    let dropped = false;
    es.onopen = () => {
      setLive(true);
      if (dropped) touch(...Object.values(EVENTS).flat());
      dropped = false;
    };
    es.onerror = () => {
      setLive(false);
      dropped = true;
    };
    for (const [type, keys] of Object.entries(EVENTS)) es.addEventListener(type, () => touch(...keys));
    return () => {
      es.close();
      if (timer) clearTimeout(timer);
    };
  }, [qc]);
  return live;
}
