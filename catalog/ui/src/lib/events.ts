// Live updates: the catalog's server-sent events invalidate the cached queries they affect, so pages
// refresh within a second of a model, lease or dataset changing instead of polling hard.

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

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
    es.onopen = () => setLive(true);
    es.onerror = () => setLive(false);
    es.addEventListener("model", () => touch("models", "model", "platform"));
    es.addEventListener("lease", () => touch("leases", "models", "model"));
    es.addEventListener("queue", () => touch("leases", "platform"));
    es.addEventListener("export", () => touch("exports"));
    // A dataset was made, edited, published to or deleted. Version files, references and comparisons never
    // change, so their queries ("version-files", "refs", "compare") aren't touched.
    es.addEventListener("dataset", () => touch("datasets", "dataset", "store-stats"));
    return () => {
      es.close();
      if (timer) clearTimeout(timer);
    };
  }, [qc]);
  return live;
}
